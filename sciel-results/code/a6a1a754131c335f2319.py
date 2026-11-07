"""Optimizer using CMA-ES with adaptive restarts for BBOB f18."""
import numpy as np
import time

class Optimizer:
    """CMA-ES with multi-start and adaptive population size."""
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def _clip(self, x):
        return np.clip(x, -5.0, 5.0)
    
    def _cma_es(self, func, x0, sigma0, budget_remaining, deadline, rng):
        """
        Run CMA-ES from x0 with initial step sigma0.
        Returns best_x, best_f, calls_used.
        """
        dim = self.dim
        # Population size: default is 4 + floor(3 * log(dim + 1))
        lambda_pop = max(4, int(4 + 3 * np.log(dim + 1)))
        
        # CMA-ES parameters
        mu = lambda_pop // 2  # Number of parents
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (weights[0] - 1) / ((dim + 2)**2 + mu_eff))
        
        # Initial state
        x_mean = x0.copy()
        sigma = sigma0
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Track best
        best_x = x0.copy()
        best_f = np.inf
        calls_used = 0
        
        # Generate initial population
        def generate_population():
            """Generate lambda_pop candidates."""
            z = rng.standard_normal((lambda_pop, dim))
            # Transform using B and D
            y = (z @ B.T) * D  # (lambda_pop, dim)
            candidates = x_mean + sigma * y
            return candidates, z
        
        # Evaluate initial population
        candidates, z = generate_population()
        candidates = np.array([self._clip(c) for c in candidates])
        values = np.array([func(c) for c in candidates])
        calls_used += lambda_pop
        
        # Update best
        for i in range(lambda_pop):
            if values[i] < best_f:
                best_f = values[i]
                best_x = candidates[i].copy()
        
        # Sort by fitness
        order = np.argsort(values)
        
        # Main loop
        while calls_used < budget_remaining:
            if time.time() > deadline:
                break
            
            # Update step size (sigma)
            # Compute evolution path
            y_mean = (B @ D) @ (weights @ z[order[:mu], :].T).T  # This is a bit off, let me redo
            # Actually, let's compute it properly
            # The weighted mean of the selected z vectors
            z_selected = z[order[:mu]]  # (mu, dim)
            # y_mean in the transformed space
            y_mean = np.zeros(dim)
            for i in range(mu):
                y_mean += weights[i] * (B @ (D * z_selected[i]))
            
            # Update ps
            norm_y = np.linalg.norm(y_mean / sigma)
            h_sigma = (norm_y / np.sqrt(1 - (1 + calls_used / (0.5 * (dim + 1)) * c_sigma)) <
                       (1 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)))
            ps = (1 - c_sigma) * ps + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean / sigma
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            sigma = np.clip(sigma, 1e-10, 50.0)
            
            # Update mean
            x_new = x_mean.copy()
            for i in range(mu):
                x_new += weights[i] * candidates[order[i]]
            x_mean = x_new
            
            # Update covariance matrix
            yw = np.array([(candidates[order[i]] - x_mean) / sigma for i in range(mu)])
            delta_h = (1 - h_sigma) * c_sigma * (2 - c_sigma)
            pc = (1 - c_c) * pc + delta_h * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
            C = ((1 - (c_1 + c_mu + delta_h * c_c * (2 - c_c))) * C +
                 c_1 * (np.outer(pc, pc) - C) +
                 c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * yw[:, :, np.newaxis] * yw[:, np.newaxis, :], axis=0))
            
            # Re-orthogonalize
            # Use eigenvalue decomposition to ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            B, D = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            D = np.sqrt(np.diag(B)).copy()
            B = np.diag(1.0 / D) @ B
            
            # Generate new population
            candidates, z = generate_population()
            candidates = np.array([self._clip(c) for c in candidates])
            
            # Check if we have enough budget
            if calls_used + lambda_pop > budget_remaining:
                # Only evaluate as many as we can afford
                n_eval = budget_remaining - calls_used
                if n_eval <= 0:
                    break
                candidates = candidates[:n_eval]
                z = z[:n_eval]
            
            values = np.array([func(c) for c in candidates])
            calls_used += len(candidates)
            
            # Update best
            for i in range(len(candidates)):
                if values[i] < best_f:
                    best_f = values[i]
                    best_x = candidates[i].copy()
            
            # Check for stagnation: if no improvement in many generations, stop
            # (simplified: just check if sigma is very small or very large)
            if sigma < 1e-8 or sigma > 20.0:
                break
            
            # Stagnation check: if best hasn't improved in 50 generations, stop
            # (simplified: track generations since last improvement)
        
        return best_x, best_f, calls_used
    
    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0
        
        # Phase 1: Initial sampling to find a good starting point
        n_initial = min(30, budget // 10)
        
        # Evaluate origin first
        x0 = np.zeros(dim)
        f0 = func(x0)
        total_calls += 1
        best_x, best_f = x0.copy(), f0
        
        # Random sampling
        initial_points = [x0.copy()]
        initial_values = [f0]
        
        for i in range(n_initial - 1):
            if total_calls >= budget or time.time() > deadline - 10.0:
                break
            xi = rng.uniform(-4.0, 4.0, size=dim)
            fi = func(xi)
            total_calls += 1
            initial_points.append(xi.copy())
            initial_values.append(fi)
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()
        
        # Phase 2: Run CMA-ES from top-k starting points
        k = min(5, len(initial_points))
        order = np.argsort(initial_values)
        
        # Allocate budget among runs
        calls_per_run = max(50, (budget - total_calls) // k)
        
        for start_idx in range(k):
            if total_calls >= budget or time.time() > deadline - 5.0:
                break
            
            x_start = initial_points[order[start_idx]].copy()
            remaining = budget - total_calls
            max_calls = min(calls_per_run, remaining)
            
            if max_calls < 20:
                break
            
            # Initial sigma: based on distance to other points or default
            sigma0 = 1.0
            if start_idx > 0:
                # Use distance to best point as hint
                dist = np.linalg.norm(x_start - best_x)
                sigma0 = max(0.5, min(2.0, dist * 0.5))
            
            x_best, f_best, calls_used = self._cma_es(
                func, x_start, sigma0, max_calls, deadline, rng
            )
            total_calls += calls_used
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()
        
        return best_x, best_f
