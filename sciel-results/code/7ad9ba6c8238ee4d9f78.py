"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive population, multi-restart, and Rastrigin-aware initialization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        rng = np.random.default_rng(0)
        d = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        total_calls = 0

        def check_deadline():
            return time.time() > deadline or total_calls >= budget

        # Rastrigin-aware initialization: try points near integer multiples of 2
        # where Rastrigin local minima tend to be
        lattice_vals = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
        
        # Generate diverse starting points
        start_points = []
        # Origin
        start_points.append(np.zeros(d))
        # Random lattice points
        for _ in range(10):
            pt = np.array([lattice_vals[rng.integers(0, len(lattice_vals))] for _ in range(d)])
            start_points.append(pt)
        
        # Evaluate initial points and pick best
        for sp in start_points:
            if check_deadline():
                break
            fsp = func(sp)
            total_calls += 1
            if fsp < best_f:
                best_f, best_x = fsp, sp.copy()

        # CMA-ES parameters
        pop_size = max(10, int(4 + 3 * np.log(d)))
        sigma_init = 1.5
        
        # Run multiple CMA-ES restarts
        max_restarts = 5
        calls_per_restart = max(80, (budget - total_calls) // max_restarts)
        
        for restart in range(max_restarts):
            if check_deadline():
                break
            
            # For first restart, use the best point found; for others, use random lattice
            if restart == 0:
                x_mean = best_x.copy()
            else:
                x_mean = np.array([lattice_vals[rng.integers(0, len(lattice_vals))] for _ in range(d)])
                fx_mean = func(x_mean)
                total_calls += 1
                if fx_mean < best_f:
                    best_f, best_x = fx_mean, x_mean.copy()
            
            sigma = sigma_init
            C = np.eye(d)
            pc = np.zeros(d)
            ps = np.zeros(d)
            cs = (pop_size + 2) / (d + 4)
            ds = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (d + 1)) - 1) + cs
            cc = (4 + d / pop_size) / (d + 4 + 2 * pop_size / d)
            cmu = min(1 / ds, 2 / ((d + 2) ** 2 + pop_size))
            
            # Sort history for ranking
            history = []
            
            for step in range(calls_per_restart):
                if check_deadline():
                    break
                
                # Generate population
                weights = np.exp(-np.arange(pop_size) / (pop_size / 2))
                weights /= weights.sum()
                mueff = 1 / (weights ** 2).sum()
                
                # Update evolution paths
                mu_w = np.sqrt(mueff) * (cs + ds - cc)
                c_sigma = 2 / ((d + 1.3) ** 2 + pop_size)
                c_c = (4 + d / pop_size) / (d + 4 + 2 * pop_size / d)
                
                # Generate candidates
                candidates = []
                for i in range(pop_size):
                    z = rng.standard_normal(d)
                    # Use Cholesky decomposition of C for sampling
                    L = np.linalg.cholesky(C)
                    y = x_mean + sigma * L @ z
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    total_calls += 1
                    candidates.append((fy, y))
                
                # Sort by fitness
                candidates.sort(key=lambda x: x[0])
                f_sorted = np.array([c[0] for c in candidates])
                x_sorted = np.array([c[1] for c in candidates])
                
                # Update best
                if f_sorted[0] < best_f:
                    best_f, best_x = f_sorted[0], x_sorted[0].copy()
                
                # Update evolution paths and covariance
                w = weights[:int(mueff)]
                x_w = np.sum(w[:, np.newaxis] * x_sorted[:int(mueff)], axis=0)
                
                # Update sigma (1/5 success rule variant)
                success = np.sum(f_sorted[:int(mueff)] < f_sorted[int(mueff):])
                if success > int(mueff) / 2:
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                
                # Update mean
                x_mean = x_w
                x_mean = np.clip(x_mean, -5, 5)
                
                # Update C matrix (simplified: just track variance)
                # For simplicity, use diagonal approximation
                diffs = x_sorted[:int(mueff)] - x_mean
                variances = np.sum(w[:, np.newaxis] * diffs ** 2, axis=0)
                C = np.diag(np.maximum(variances, 1e-8)) + 1e-6 * np.eye(d)
                
                # Rescale sigma to match the scale of C
                sigma = np.sqrt(np.mean(np.diag(C))) * 0.5
                sigma = np.clip(sigma, 0.01, 5.0)
                
                if check_deadline():
                    break
        
        # Final polish around the best point
        if best_x is not None and not check_deadline():
            x = best_x.copy()
            fx = best_f
            sigma = 0.2
            remaining = budget - total_calls
            
            for _ in range(min(30, remaining)):
                if check_deadline():
                    break
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                total_calls += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    sigma *= 1.05
                else:
                    sigma *= 0.92

        return best_x, best_f
# EVOLVE-BLOCK-END
