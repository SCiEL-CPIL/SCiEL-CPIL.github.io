import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        
        def count_calls(x):
            nonlocal calls_made, best_x, best_f
            calls_made += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Custom lightweight CMA-ES implementation
        # This avoids the overhead of the cma library and ensures strict budget control.
        
        def cma_es_light(x0, sigma0, max_fvals, seed=42):
            nonlocal calls_made, best_x, best_f
            rng = np.random.RandomState(seed)
            
            # Initialize
            x = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            cs = np.zeros(dim)
            
            # Strategy parameters
            mu = max(1, int(np.floor(dim / 2)))  # Number of weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_s = (2 + mu_eff / (dim + 2)) / (dim + 2 + 2 * mu_eff / dim)
            c_sigma = (mu_eff / 2 + 1) / (dim + mu_eff / 2 + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_s = (2 + mu_eff / (dim + 2)) / (dim + 2 + 2 * mu_eff / dim)
            
            chi_n = np.sqrt(dim) * (1 - 1. / (4. * dim) + 1. / (21. * dim**2))
            
            # Initial evaluation
            f_x = count_calls(x)
            
            while calls_made < max_fvals:
                if time.time() > deadline - 5:
                    break
                
                # Generate lambda samples
                lam = 2 * mu
                ps = np.zeros(dim)
                
                # Create population
                ys = []
                xs = []
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    y = sigma * np.linalg.cholesky(C) @ z
                    x_new = x + y
                    xs.append(x_new)
                    ys.append(y)
                    
                # Evaluate
                fs = np.array([count_calls(xi) for xi in xs])
                
                # Sort
                sort_idx = np.argsort(fs)
                xs = np.array(xs)[sort_idx]
                ys = np.array(ys)[sort_idx]
                fs = fs[sort_idx]
                
                # Update mean
                x_old = x
                x = np.sum(weights[:, np.newaxis] * xs[:mu], axis=0)
                
                # Update covariance matrix
                y_w = np.sum(weights[:, np.newaxis] * ys[:mu], axis=0)
                C = (1 - c_c * (2 - c_c)) * C + c_c * (2 - c_c) * np.outer(pc, pc) + c_s * (1 - c_s) * np.outer(ys[:mu].T @ weights, y_w)
                
                # Update path covariance
                pc = (1 - c_c) * pc + c_c * (x - x_old) / sigma
                
                # Update step size
                norm_y_w = np.linalg.norm(y_w)
                h_sigma = 1 if (np.sqrt((1 - (1 - c_c)**2) * (1 + (2 - c_c) * c_c * mu_eff / dim)) / (1 - c_c) * chi_n) < 1.4 + 2 / (dim + 1) else 0
                cs = (1 - c_sigma) * cs + c_sigma * (h_sigma * norm_y_w / chi_n - 1)
                sigma = sigma * np.exp(c_sigma / c_sigma * (norm_y_w / chi_n - 1))
                
                # Check convergence
                if np.ptp(fs[:mu]) < 1e-10:
                    break
                if np.linalg.norm(x - x_old) / sigma < 1e-8:
                    break
                
                # Coarse-to-fine strategy: shrink sigma aggressively if we are still far from "convergence"
                # This is a heuristic to speed up the "search" phase for Sphere
                if calls_made < max_fvals * 0.2:
                    sigma *= 0.95
                elif calls_made < max_fvals * 0.5:
                    sigma *= 0.99

        # Strategy:
        # 1. Initial broad search with large sigma to locate the basin.
        # 2. If not converged, restart with smaller sigma centered at best_x.
        
        # Start 1: Broad search
        x0 = np.zeros(dim)
        sigma0 = 3.0  # Covers most of [-5, 5]
        max_fvals_1 = int(budget * 0.4)
        
        cma_es_light(x0, sigma0, max_fvals_1, seed=42)
        
        # Start 2: Focused search if we haven't converged
        if calls_made < budget:
            x0 = best_x.copy()
            sigma0 = 0.5
            max_fvals_2 = int(budget * 0.4)
            cma_es_light(x0, sigma0, max_fvals_2, seed=123)
            
        # Start 3: Fine-tuning
        if calls_made < budget:
            x0 = best_x.copy()
            sigma0 = 0.1
            max_fvals_3 = budget - calls_made
            cma_es_light(x0, sigma0, max_fvals_3, seed=999)

        return best_x, best_f
