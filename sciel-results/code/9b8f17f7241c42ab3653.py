# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.perf_counter()
        deadline = t_start + 280.0  # Safety margin

        rng = np.random.default_rng(42)
        
        # Track best solution
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return float('inf')
            if time.perf_counter() > deadline:
                return float('inf')
            calls_made += 1
            try:
                f_val = float(func(x))
            except Exception:
                return float('inf')
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Radial Sweep Initialization
        # Sample points along axes and random directions to find the needle
        directions = []
        for i in range(dim):
            for sign in [1, -1]:
                d = np.zeros(dim)
                d[i] = sign
                directions.append(d)
        
        for _ in range(10):
            d = rng.standard_normal(dim)
            norm = np.linalg.norm(d)
            if norm > 1e-10:
                d /= norm
                directions.append(d)
                
        radii = [0.5, 1.0, 2.0, 3.0, 4.0]
        
        for d in directions:
            for r in radii:
                if calls_made >= budget or time.perf_counter() > deadline:
                    break
                x = np.clip(r * d, -5, 5)
                f = evaluate(x)
                if f == float('inf'):
                    break

        if calls_made >= budget or time.perf_counter() > deadline:
            return best_x, best_f

        # Phase 2: CMA-ES implementation
        # CMA-ES is well-suited for the Discus function's anisotropic landscape
        
        # Initialize CMA-ES state
        m = best_x.copy()
        sigma = 1.0
        population_size = max(4, int(4 + 3 * np.log(dim)))
        
        # CMA-ES parameters
        c_sigma = (pop + 2) / ((dim + 8) * (pop + 1.5))
        d_sigma = 1 + 2 * max(0, np.sqrt((pop - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + pop / dim) / (dim + 4 + 2 * pop / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + pop)
        c_mu = min(1 - c_1, 2 * ((pop / 2 + 1) - dim - 2) / ((dim + 2) ** 2 + pop))
        
        # Weighting scheme
        mu = pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff / 2 + 1) - dim - 2) / ((dim + 2) ** 2 + mu_eff))
        
        # Covariance matrix
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)  # Initially identity
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Track progress for restart
        best_f_start = best_f
        no_improve_count = 0
        max_no_improve = 50
        
        while calls_made < budget and time.perf_counter() < deadline:
            # Generate population
            offspring = []
            for i in range(population_size):
                z = rng.standard_normal(dim)
                x_new = m + sigma * z
                x_new = np.clip(x_new, -5, 5)
                offspring.append((x_new, z))
            
            # Evaluate offspring
            offspring_f = []
            for x_new, z in offspring:
                if calls_made >= budget or time.perf_counter() > deadline:
                    break
                f_val = evaluate(x_new)
                offspring_f.append((f_val, x_new, z))
            
            if len(offspring_f) < population_size:
                break
            
            # Sort by fitness
            offspring_f.sort(key=lambda t: t[0])
            
            # Update mean
            new_m = np.zeros(dim)
            for i in range(mu):
                new_m += weights[i] * offspring_f[i][1]
            m = new_m
            
            # Update sigma
            old_sigma = sigma
            # Compute C^{-1/2}(m - m_old)
            delta_m = (m - (m - sigma * np.linalg.multi_dot([C_inv_sqrt, p_sigma]))) / sigma
            h_sigma = (np.linalg.norm(delta_m) / 
                       np.sqrt(1 - (1 - c_sigma) ** (2 * population_size)))
            h_sigma = min(1.0, h_sigma)
            
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * h_sigma * delta_m
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = max(1e-8, min(sigma, 1e3))
            
            # Update covariance matrix
            y = (m - (m - sigma * np.linalg.multi_dot([C_inv_sqrt, p_sigma]))) / sigma
            p_c = (1 - c_c) * p_c + c_c * y
            
            # Rank-1 and rank-mu updates
            C = (1 + c_1 * c_mu) * C - c_1 * np.outer(y, y)
            for i in range(mu):
                z_i = offspring_f[i][2]
                C += c_mu * np.outer(z_i, z_i)
            
            # Recompute C_inv_sqrt using eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Check for stagnation
            if best_f < best_f_start:
                best_f_start = best_f
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Restart if stagnating
            if no_improve_count > max_no_improve:
                # Perturb mean and increase sigma
                m = m + sigma * rng.standard_normal(dim)
                m = np.clip(m, -5, 5)
                sigma = max(sigma * 2.0, 0.5)
                C = np.eye(dim)
                C_inv_sqrt = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                no_improve_count = 0
                best_f_start = best_f
        
        return best_x, best_f
# EVOLVE-BLOCK-END
