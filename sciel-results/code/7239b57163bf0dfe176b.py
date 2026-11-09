"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid CMA-ES with global restarts for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 300s kill

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        call_count = 0

        def _evaluate(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global Sampling to find the best basin
        # Use a mix of structured and random points
        n_init = min(20, budget // 10)
        
        # Generate initial points: corners, center, and random
        init_points = []
        # Center
        init_points.append(np.zeros(dim))
        # Random points in [-4, 4]
        for _ in range(n_init - 1):
            init_points.append(rng.uniform(-4.0, 4.0, size=dim))
        
        # Evaluate initial points
        for pt in init_points:
            if call_count >= budget - 10:
                break
            if time.time() > deadline:
                break
            _evaluate(pt)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            call_count += 1

        # Phase 2: CMA-ES with Restarts
        # CMA-ES Parameters
        lambda_ = 8  # Smaller population for more generations
        mu_ = lambda_ // 2
        
        # Weights
        w = np.zeros(mu_)
        for i in range(mu_):
            w[i] = np.log(mu_ + 0.5) - np.log(i + 1)
        w /= w.sum()
        
        mu_eff = 1.0 / np.sum(w ** 2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * min(mu_ / (dim + 2) - 1, 1) + c_sigma
        c_c = (4 + dim / c_sigma) / (dim + 4 + 2 / c_sigma)
        c_1 = 2 / (dim ** 2 + 6)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (dim ** 2 + 2))
        
        # Chi_n constant for sigma update
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Restart logic
        max_stagnation = 10  # Generations without improvement before restart
        stagnation_count = 0
        prev_best = best_f
        sigma_init = 1.5
        restart_count = 0
        max_restarts = 3
        
        while call_count < budget and time.time() < deadline:
            if restart_count > max_restarts:
                break
            
            # Initialize CMA-ES state
            x_mean = best_x.copy()
            sigma = sigma_init
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Eigen decomposition cache
            D = np.ones(dim)
            B = np.eye(dim)
            
            # Run CMA-ES until stagnation or budget exhausted
            while call_count < budget and time.time() < deadline:
                if stagnation_count >= max_stagnation:
                    break
                
                # Generate population
                offspring = []
                for i in range(lambda_):
                    if call_count >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (B @ (D * z))
                    f_y = _evaluate(y)
                    offspring.append((y, f_y, z))
                
                if len(offspring) < mu_:
                    break
                    
                # Sort by fitness
                offspring.sort(key=lambda x: x[1])
                selected = offspring[:mu_]
                
                # Check for improvement
                if offspring[0][1] < prev_best - 1e-10:
                    prev_best = offspring[0][1]
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                
                # Update mean
                old_mean = x_mean.copy()
                x_mean = np.sum([w[i] * y for i, (y, f, z) in enumerate(selected)], axis=0)
                
                # Update evolution paths
                C_inv_sqrt = B @ np.diag(1.0 / np.maximum(D, 1e-10)) @ B.T
                delta_mean = x_mean - old_mean
                
                p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * (C_inv_sqrt @ delta_mean)
                
                # Update sigma
                z_mean = np.sum([w[i] * z for i, (y, f, z) in enumerate(selected)], axis=0)
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * z_mean
                
                norm_p_sigma = np.linalg.norm(p_sigma)
                sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                # Update Covariance Matrix
                C = (1 - c_1 - c_mu) * C
                C += c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_sigma) ** 2) * np.eye(dim))
                
                for i in range(mu_):
                    diff = selected[i][0] - x_mean
                    C += c_mu * w[i] * np.outer(diff, diff)
                
                C = (C + C.T) / 2  # Symmetrize
                
                # Ensure positive semi-definite
                try:
                    eigvals_C, _ = np.linalg.eigh(C)
                    if np.min(eigvals_C) < 1e-12:
                        C = C + (1e-10 - np.min(eigvals_C)) * np.eye(dim)
                except:
                    C = np.eye(dim)
                    
                # Update eigen decomposition for next iteration
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(eigvals, 1e-12))
                    B = eigvecs
                except:
                    pass
            
            # If we broke due to stagnation, restart from a new point
            if stagnation_count >= max_stagnation and restart_count < max_restarts:
                restart_count += 1
                # New starting point: random or current best with larger sigma
                if rng.random() < 0.5:
                    best_x = rng.uniform(-4.0, 4.0, size=dim)
                sigma_init = 2.0
                # Reset stagnation
                stagnation_count = 0
                prev_best = best_f

        return best_x, best_f
# EVOLVE-BLOCK-END
