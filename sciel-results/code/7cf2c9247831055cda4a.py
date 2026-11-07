"""Optimizer using population-based CMA-ES with adaptive restarts for BBOB f18."""
import numpy as np
import time

class Optimizer:
    """CMA-ES with adaptive restarts for ill-conditioned functions."""
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def _clip(self, x):
        return np.clip(x, -5.0, 5.0)
    
    def _cma_es_step(self, mean, sigma, C, sqrtC, weights, mu_eff, rng, func, budget_remaining, deadline):
        """One CMA-ES iteration. Returns (new_mean, new_sigma, new_C, new_sqrtC, best_x, best_f, calls_used)."""
        d = self.dim
        lam = 2 * (4 + int(np.log(d)))  # population size
        mu = lam // 2  # number of recombination weights
        
        # Compute weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        w = np.concatenate([w, np.zeros(lam - mu)])
        
        # Compute sqrtC via Cholesky or eigendecomposition
        # We'll maintain sqrtC as the matrix square root of C
        try:
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            sqrtC = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        except np.linalg.LinAlgError:
            sqrtC = np.eye(d)
        
        # Sample population
        population = []
        for i in range(lam):
            z = rng.standard_normal(d)
            x = mean + sigma * (sqrtC @ z)
            x = self._clip(x)
            population.append(x)
        
        # Evaluate population
        values = []
        for x in population:
            if budget_remaining <= 0 or time.time() > deadline:
                break
            f = func(x)
            values.append(f)
            budget_remaining -= 1
        
        if len(values) == 0:
            return mean, sigma, C, sqrtC, None, np.inf, 0
        
        # Sort by fitness
        idx = np.argsort(values)
        sorted_idx = idx[:len(values)]
        sorted_values = np.array(values)[sorted_idx]
        
        # Weighted recombination
        new_mean = np.zeros(d)
        for i in range(min(mu, len(sorted_idx))):
            new_mean += w[i] * population[sorted_idx[i]]
        
        # Update covariance matrix
        diff = np.array([population[sorted_idx[i]] - mean for i in range(min(mu, len(sorted_idx)))])
        C_new = C + sigma**2 * (2.0 / mu_eff) * (diff.T @ (w[:min(mu, len(sorted_idx))] * diff))
        
        # Symmetrize
        C_new = (C_new + C_new.T) / 2.0
        
        # Update step size (1/5 rule)
        if len(sorted_idx) >= mu:
            h = np.sqrt((mu_eff - 2 + 2.0 / (mu_eff - 1)) / ((d + 2) * mu_eff))
            ps = (1 - c_sigma) * ps if hasattr(self, '_ps') else np.zeros(d)
            ps = (1 - c_sigma) * ps + c_sigma / np.sqrt(mu_eff) * (sqrtC.T @ new_mean - mean) / sigma
            cs = (mu_eff + 2) / (d + mu_eff + 5)
            hsig = 1.4 + 2.0 / (d + 1)
            sigma = sigma * np.exp(cs / (d + 2) * (np.linalg.norm(ps) / np.sqrt(mu_eff) - hsig))
        else:
            sigma *= 0.95
        
        sigma = np.clip(sigma, 1e-10, 50.0)
        
        # Find best in this generation
        best_idx = sorted_idx[0] if len(sorted_idx) > 0 else 0
        best_x = population[best_idx]
        best_f = sorted_values[0] if len(sorted_values) > 0 else np.inf
        
        calls_used = len(values)
        
        return new_mean, sigma, C_new, sqrtC, best_x, best_f, calls_used
    
    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0
        
        # Initial random sampling
        n_initial = min(10, budget // 10)
        for i in range(n_initial):
            if total_calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4.5, 4.5, size=dim)
            f = func(x)
            total_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # CMA-ES with adaptive restarts
        max_restarts = 5
        for restart in range(max_restarts):
            if total_calls >= budget or time.time() > deadline - 10:
                break
            
            remaining = budget - total_calls
            if remaining < 30:
                break
            
            # Initialize CMA-ES
            mean = best_x.copy()
            sigma = 1.0
            C = np.eye(dim)
            
            # Run CMA-ES for a portion of the remaining budget
            cma_budget = min(remaining, max(50, budget // (max_restarts - restart)))
            
            no_improve_count = 0
            prev_best = best_f
            
            while total_calls < budget and time.time() < deadline:
                if no_improve_count > 20:
                    break
                
                mean, sigma, C, _, bx, bf, calls_used = self._cma_es_step(
                    mean, sigma, C, None, None, None, rng, func, 
                    budget - total_calls, deadline
                )
                total_calls += calls_used
                
                if bf < best_f:
                    best_f = bf
                    best_x = bx.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # Check for convergence
                if sigma < 1e-6 or np.allclose(C, np.eye(dim) * np.trace(C) / dim, atol=1e-4):
                    break
            
            # If we didn't improve much, restart from a new random point
            if best_f == prev_best and restart < max_restarts - 1:
                x_new = rng.uniform(-4.0, 4.0, size=dim)
                f_new = func(x_new)
                total_calls += 1
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
        
        return best_x, best_f
