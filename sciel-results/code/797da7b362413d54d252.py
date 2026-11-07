import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        # State variables
        x_best = np.random.uniform(-5, 5, dim)
        f_best = func(x_best)
        evals = 1

        # CMA-ES parameters
        # For dim=5, lambda=5 is a good balance
        lam = 5
        mu = lam // 2  # Number of parents
        # Weights for recombination
        weights = np.ones(mu)
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Initial step size
        sigma = 2.0
        # Initial covariance
        C = np.eye(dim)
        # Pre-compute inverse sqrt of C for sampling (approximation for speed, or just use cholesky)
        # For dim=5, cholesky is fast
        try:
            B = np.linalg.cholesky(C)
        except:
            B = np.eye(dim)
        
        # Evolution path for sigma adaptation
        p_sigma = np.zeros(dim)
        # Evolution path for covariance adaptation
        p_c = np.zeros(dim)
        
        # Constants
        c_sigma = (mu_eff + 2) / (dim + 4)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Threshold for stagnation
        stagnation_counter = 0
        best_f_prev = f_best
        threshold = 1e-12
        
        def nelder_mead(func, x0, f0, budget_remaining, deadline):
            # Simple Nelder-Mead implementation
            dim = len(x0)
            # Initial simplex
            simplex = [x0.copy() for _ in range(dim + 1)]
            f_simplex = [f0] * (dim + 1)
            
            # Expand simplex
            for i in range(dim):
                simplex[i+1] = x0.copy()
                simplex[i+1][i] += 0.05 if x0[i] < 5 else -0.05
                f_simplex[i+1] = func(simplex[i+1])
            
            evals_local = dim + 1
            best_idx = np.argmin(f_simplex)
            
            max_iter = budget_remaining
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                # Sort
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                # Centroid
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + (centroid - simplex[-1])
                f_ref = func(x_ref)
                evals_local += 1
                
                if f_ref < f_simplex[0]:
                    # Expansion
                    x_exp = centroid + 2 * (x_ref - centroid)
                    f_exp = func(x_exp)
                    evals_local += 1
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_simplex[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_simplex[-1] = f_ref
                elif f_ref < f_simplex[-2]:
                    simplex[-1] = x_ref
                    f_simplex[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + 0.5 * (simplex[-1] - centroid)
                    f_con = func(x_con)
                    evals_local += 1
                    if f_con < f_simplex[-1]:
                        simplex[-1] = x_con
                        f_simplex[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            f_simplex[i] = func(simplex[i])
                            evals_local += 1
                
                best_idx = np.argmin(f_simplex)
                if f_simplex[best_idx] < f_best:
                    f_best = f_simplex[best_idx]
                    x_best = simplex[best_idx].copy()
                    
                # Convergence check
                if np.max(f_simplex) - np.min(f_simplex) < 1e-10:
                    break
                    
            return x_best, f_best, evals_local

        # Main Loop
        while evals < budget:
            if time.time() > deadline - 10:
                break
                
            # Check if we should switch to local search
            # Heuristic: if sigma is small and we've made some progress
            if evals > 100 and sigma < 0.1 and (budget - evals) > 20:
                x_polished, f_polished, evals_used = nelder_mead(func, x_best, f_best, budget - evals, deadline)
                evals += evals_used
                if f_polished < f_best:
                    f_best = f_polished
                    x_best = x_polished
                break

            # Generate samples
            z = np.random.randn(lam, dim)
            # Transform to correlated normal
            # y = x + sigma * B @ z.T
            ys = x_best + sigma * (z @ B.T)
            # Clip to box
            ys = np.clip(ys, -5, 5)
            
            # Evaluate
            fs = np.array([func(y) for y in ys])
            evals += lam
            
            # Update best
            min_idx = np.argmin(fs)
            if fs[min_idx] < f_best:
                f_best = fs[min_idx]
                x_best = ys[min_idx].copy()
            
            # CMA-ES Update
            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]
            
            # Recombination
            x_new = np.sum(weights[:, None] * ys_sorted[:mu], axis=0)
            
            # Update covariance
            # v = (x_new - x_best) / sigma
            v = (x_new - x_best) / sigma
            
            # Update p_sigma
            c_sigma_sqrt = np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
            B_inv = np.linalg.inv(B)
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma_sqrt * (B_inv @ v)
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * v
            
            # Hessian-like term
            delta_sigma = np.sqrt((1 + 2 * max(0, (mu_eff - 1) / (dim + 1) - 1)) * (np.linalg.norm(p_c)**2 / c_c - 1))
            
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c * p_c.T + (delta_sigma - 1) * C) + c_mu * sum(w_i * y_i * y_i.T)
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (delta_sigma - 1) * C)
            
            # Sum of outer products of selected parents
            sum_outer = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (ys_sorted[i] - x_best) / sigma
                sum_outer += weights[i] * np.outer(y_i, y_i)
                
            C += c_mu * sum_outer
            
            # Ensure positive definite
            try:
                B = np.linalg.cholesky(C)
            except:
                # If not PD, add identity
                C += 1e-6 * np.eye(dim)
                try:
                    B = np.linalg.cholesky(C)
                except:
                    B = np.eye(dim)
                    C = np.eye(dim)
                    sigma = 1.0 # Reset
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
            
            x_best = x_new
            
            # Stagnation check
            if f_best == best_f_prev:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                best_f_prev = f_best
                
            if stagnation_counter > 30:
                # Restart with new center, keep sigma small
                x_best = np.random.uniform(-5, 5, dim)
                f_best = func(x_best)
                evals += 1
                sigma = 1.0
                C = np.eye(dim)
                B = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                stagnation_counter = 0
                best_f_prev = f_best

        return x_best, f_best
