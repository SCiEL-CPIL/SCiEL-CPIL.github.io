import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        
        def time_left():
            return deadline - time.time()

        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def eval_func(x):
            nonlocal calls_used
            x = np.clip(x, -5, 5)
            f_val = func(x)
            calls_used += 1
            return f_val

        # Phase 1: Coarse Grid Search
        # 3 points per dimension: -4, 0, 4
        # 3^5 = 243 points.
        grid_points = []
        coords = np.linspace(-4.0, 4.0, 3)
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            grid_points.append(np.array([coords[i], coords[j], coords[k], coords[l], coords[m]]))
        
        # Evaluate grid
        for gp in grid_points:
            if calls_used >= budget or time_left() < 10:
                break
            f_val = eval_func(gp)
            if f_val < best_f:
                best_f = f_val
                best_x = gp.copy()
        
        if best_x is None:
            # Fallback if grid failed or budget was tiny
            best_x = np.zeros(dim)
            best_f = eval_func(best_x)
            
        # Phase 2: Local Refinement with CMA-ES
        # Remaining budget
        remaining = budget - calls_used
        
        # If we have enough budget for CMA-ES
        if remaining > 20 and time_left() > 10:
            # CMA-ES Parameters
            lam = 10  # Smaller population to fit budget
            mu = lam // 2
            
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w**2)
            
            m = best_x.copy()
            sigma = 1.0  # Start with moderate sigma
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            gen = 0
            max_gens = 25  # Limit generations to ensure we don't run out
            
            while gen < max_gens and calls_used < budget and time_left() > 5:
                # Check if we have enough budget for this generation
                if calls_used + lam > budget:
                    break
                
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)
                
                z = np.random.randn(lam, dim)
                y = m + sigma * (z @ L.T)
                
                fy = np.array([eval_func(yp) for yp in y])
                
                if calls_used >= budget:
                    break
                
                # Update best
                min_idx = np.argmin(fy)
                if fy[min_idx] < best_f:
                    best_f = fy[min_idx]
                    best_x = y[min_idx].copy()
                
                # Sort
                idx = np.argsort(fy)
                y_sorted = y[idx]
                
                # Update mean
                m_new = np.sum(w[:, np.newaxis] * y_sorted[:mu], axis=0)
                
                # Update p_sigma
                c_sigma = (2 + math.log(mu_eff + 2)) / (dim + 4)
                p_sigma = (1 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (L.T @ (m_new - m))
                
                # Update sigma
                sigma *= math.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / math.sqrt(mu_eff) - 1))
                
                # Update p_c
                c_c = (4 + math.log(dim) - math.log(mu_eff)) / (dim + 4)
                p_c = (1 - c_c) * p_c + math.sqrt(c_c * (2 - c_c) * mu_eff) * (m_new - m) / sigma
                
                # Update C
                c1 = 2 / ((dim + 1.3)**2 + mu_eff)
                cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
                
                C = (1 + c1 + cmu) * C - c1 * np.outer(p_c, p_c)
                
                y_inv = (y_sorted[:mu] - m) / sigma
                C += cmu * (w[:, np.newaxis] * (y_inv * y_inv)).sum(axis=0)
                
                C = (C + C.T) / 2
                
                # Ensure PD
                try:
                    eigvals = np.linalg.eigvalsh(C)
                    if np.any(eigvals < 1e-10):
                        C = np.eye(dim)
                        p_sigma = np.zeros(dim)
                        p_c = np.zeros(dim)
                        sigma = 1.0
                except:
                    C = np.eye(dim)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    sigma = 1.0

                m = m_new
                gen += 1
        
        # Phase 3: Final Polishing with Nelder-Mead (if budget remains)
        remaining = budget - calls_used
        if remaining > 10 and time_left() > 5 and best_x is not None:
             def nm_func(x):
                 nonlocal calls_used
                 x = np.clip(x, -5, 5)
                 f_val = func(x)
                 calls_used += 1
                 return f_val

             simplex = [best_x.copy()]
             for i in range(dim):
                 x_temp = best_x.copy()
                 step = 0.1
                 x_temp[i] += step
                 simplex.append(x_temp)
             
             f_values = [nm_func(x) for x in simplex]
             
             max_iter = (budget - calls_used) // (dim + 1)
             for _ in range(max_iter):
                 if calls_used >= budget or time_left() < 1:
                     break
                     
                 order = np.argsort(f_values)
                 simplex = [simplex[i] for i in order]
                 f_values = [f_values[i] for i in order]
                 
                 centroid = np.mean(simplex[:-1], axis=0)
                 
                 x_r = centroid + 1.0 * (centroid - simplex[-1])
                 x_r = np.clip(x_r, -5, 5)
                 f_r = nm_func(x_r)
                 
                 if f_r < f_values[0]:
                     x_e = centroid + 2.0 * (centroid - simplex[-1])
                     x_e = np.clip(x_e, -5, 5)
                     f_e = nm_func(x_e)
                     if f_e < f_r:
                         simplex[-1] = x_e
                         f_values[-1] = f_e
                     else:
                         simplex[-1] = x_r
                         f_values[-1] = f_r
                 elif f_r < f_values[-2]:
                     simplex[-1] = x_r
                     f_values[-1] = f_r
                 else:
                     x_c = centroid + 0.5 * (simplex[-1] - centroid)
                     x_c = np.clip(x_c, -5, 5)
                     f_c = nm_func(x_c)
                     if f_c < f_values[-1]:
                         simplex[-1] = x_c
                         f_values[-1] = f_c
                     else:
                         for i in range(1, dim + 1):
                             simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                             f_values[i] = nm_func(simplex[i])
                             
                 min_idx = np.argmin(f_values)
                 if f_values[min_idx] < best_f:
                     best_f = f_values[min_idx]
                     best_x = simplex[min_idx].copy()

        return best_x, best_f
