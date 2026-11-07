"""Optimizer using lightweight CMA-ES for BBOB f18 (Schaffers F7 ill-conditioned)."""
import numpy as np
import time

class Optimizer:
    """Lightweight CMA-ES with adaptive restarts for ill-conditioned functions."""
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def _clip(self, x):
        return np.clip(x, -5.0, 5.0)
    
    def _cma_es(self, func, x0, sigma, budget, rng, deadline):
        """Run CMA-ES from initial point x0 with step size sigma.
        
        Returns (best_x, best_f, calls_used).
        """
        dim = self.dim
        # Population size: 4 + floor(3 * ln(dim)) for dim=5 -> 4+5=9
        pop_size = 4 + int(3 * np.log(dim)) if dim > 1 else 2
        pop_size = min(pop_size, 12)  # Cap for budget efficiency
        
        mean = x0.copy()
        C = np.eye(dim)
        C_eigvals = np.ones(dim)
        C_eigvecs = np.eye(dim)
        
        # Step sizes
        cc = (4 + dim / 3) / (dim + 4 + 2 / dim)
        cmu = min(1 - cc, 2 / ((dim + 1.3) ** 2 + cc))
        cs = (dim + 1.7) / (dim + 9)
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1) + cs
        sigma = sigma
        sigma_history = []
        
        # Update parameters
        cc = (4 + dim / 3) / (dim + 4 + 2 / dim)
        cmu = min(1 - cc, 2 / ((dim + 1.3) ** 2 + cc))
        cs = (dim + 1.7) / (dim + 9)
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1) + cs
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        best_x = mean.copy()
        best_f = func(mean)
        calls = 1
        
        # Initial population
        for _ in range(pop_size):
            if calls >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            
            # Sample from N(mean, sigma^2 * C)
            z = rng.standard_normal(dim)
            # Use eigendecomposition for sampling
            x_sample = self._clip(mean + sigma * (C_eigvecs @ (C_eigvals ** 0.5 * z)))
            fx = func(x_sample)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x_sample.copy()
        
        while calls < budget:
            if time.time() > deadline - 5.0:
                break
            
            # Generate population
            offspring = []
            for _ in range(pop_size):
                if calls >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                
                z = rng.standard_normal(dim)
                x_sample = self._clip(mean + sigma * (C_eigvecs @ (C_eigvals ** 0.5 * z)))
                fx = func(x_sample)
                calls += 1
                offspring.append((x_sample, fx))
                if fx < best_f:
                    best_f = fx
                    best_x = x_sample.copy()
            
            if len(offspring) < 2:
                break
            
            # Sort by fitness
            offspring.sort(key=lambda x: x[1])
            lambda_ = len(offspring)
            mu = max(1, lambda_ // 2)
            
            # Weights (linear)
            weights = np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1 / (weights ** 2).sum()
            
            # Update mean
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * np.array([o[0] for o in offspring[:mu]]), axis=0)
            
            # Update evolution paths
            y = (mean - old_mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y
            if np.linalg.norm(ps) / chi_n < (1.4 + cc) * np.sqrt(1 + (1 - cs) / cs):
                pc = (1 - cc) * pc + cc * y
            
            # Update covariance matrix
            delta_sig = 1.0
            for i in range(mu):
                w_i = weights[i]
                y_i = (offspring[i][0] - old_mean) / sigma
                C = (1 - ds - cmu * mueff + cmu * mueff) * C + \
                    (ds - 2 + cmu * mueff) * np.outer(pc, pc) + \
                    cmu * w_i * (np.outer(y_i, y_i) + np.outer(pc, pc))
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Eigendecomposition (only every few iterations for speed)
            try:
                C_eigvals, C_eigvecs = np.linalg.eigh(C)
                C_eigvals = np.clip(C_eigvals, 1e-20, None)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                C_eigvals = np.ones(dim)
                C_eigvecs = np.eye(dim)
            
            # Sigma adaptation (CSA)
            sigma_history.append(sigma)
            if len(sigma_history) > 10:
                sigma = sigma * np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 20.0)
            
            # Check for convergence
            if np.linalg.norm(mean - old_mean) < 1e-8 * sigma:
                break
        
        return best_x, best_f, calls
    
    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # Initial random sampling to find a good starting point
        n_initial = min(10, budget // 5)
        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0
        
        for i in range(n_initial):
            if total_calls >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            if i == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-4.0, 4.0, size=dim)
            fx = func(x0)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()
        
        # Run CMA-ES from the best initial point
        remaining = budget - total_calls
        if remaining > 10:
            sigma = 2.0  # Reasonable initial step for [-5,5]^5
            x_cma, f_cma, calls_cma = self._cma_es(func, best_x, sigma, remaining, rng, deadline)
            total_calls += calls_cma
            if f_cma < best_f:
                best_f = f_cma
                best_x = x_cma.copy()
        
        # If we still have budget, try another CMA-ES from a different start
        if total_calls < budget - 20 and time.time() < deadline - 10.0:
            x_start = rng.uniform(-3.0, 3.0, size=dim)
            fx = func(x_start)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x_start.copy()
            
            remaining = budget - total_calls
            if remaining > 10:
                sigma = 1.5
                x_cma, f_cma, calls_cma = self._cma_es(func, best_x, sigma, remaining, rng, deadline)
                total_calls += calls_cma
                if f_cma < best_f:
                    best_f = f_cma
                    best_x = x_cma.copy()
        
        return best_x, best_f
