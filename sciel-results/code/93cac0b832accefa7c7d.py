"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid CMA-ES with Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        deadline = time.time() + 290  # Safety margin
        rng = np.random.default_rng()

        def cmaes(x0, budget, sigma0):
            """Simple CMA-ES implementation."""
            x = x0.copy()
            fx = func(x)
            calls = 1
            sigma = sigma0
            n = dim
            # Population size
            lam = max(4, int(4 + 3 * np.log(n)))
            # Step size adaptation
            c_sigma = (lam / 2 + 1) / (n + 2 + lam / 2)
            d_sigma = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (n + 1)) - 1) + c_sigma
            # Mean update
            c1 = (2 / ((n + 1.3) ** 2 + lam))
            cmu = min(1 - c1, 2 * ((lam / 2 - 2 + 2 / lam) / ((n + 2) ** 2 + lam)))
            cs = (lam / 2 + 1) / (n + 4)
            c1 = 1 / (n + 1.3)
            cmu = min(1 - c1, 2 * ((lam / 2 - 2 + 2 / lam) / ((n + 2) ** 2 + lam)))
            cc = (4 + n / lam) / (n + 4 + 2 * lam / n)
            damps = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (n + 1)) - 1) + cs
            # Covariance matrix adaptation
            C = np.eye(n)
            C_inv = np.eye(n)
            C_sqrt = np.eye(n)
            C_inv_sqrt = np.eye(n)
            # Evolution paths
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            
            for _ in range(budget // lam):
                if time.time() > deadline:
                    break
                
                # Generate population
                Y = []
                for i in range(lam):
                    z = rng.standard_normal(n)
                    y = x + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    Y.append(y)
                
                # Evaluate
                F = []
                for y in Y:
                    fy = func(y)
                    calls += 1
                    F.append(fy)
                
                # Sort
                idx = np.argsort(F)
                Y = [Y[i] for i in idx]
                F = [F[i] for i in idx]
                
                # Update mean
                w = np.log(np.log(lam + 1) / 2) - np.log(np.arange(1, lam + 1))
                w = w / w.sum()
                mu_eff = 1 / (w ** 2).sum()
                new_x = np.sum([w[i] * Y[i] for i in range(lam)], axis=0)
                
                # Update sigma
                y_sigma = (new_x - x) / (sigma + 1e-32)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) / chi_n < (1.4 + 2 / (n + 1)) * (1 - c_sigma)
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Update C
                p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mu_eff) * y_sigma
                delta_h = (1 - h_sigma) * cc * (1 - cc)
                C = (1 - c1 - cmu) * C + c1 * np.outer(y_sigma, y_sigma) + cmu * np.sum([w[i] ** 2 * np.outer(Y[i] - x, Y[i] - x) for i in range(lam)], axis=0) / (sigma ** 2 + 1e-32)
                
                # Recompute square root
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
                
                x = new_x
                fx = F[0]
                
            return x, fx

        def nelder_mead(x0, budget, initial_step=0.5):
            """Nelder-Mead simplex method."""
            n = dim
            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(n):
                v = x0.copy()
                v[i] += initial_step
                simplex.append(np.clip(v, -5, 5))
            
            fvals = [func(s) for s in simplex]
            calls = n + 1
            
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma_nm = 0.5  # Shrink
            
            for _ in range(budget - calls):
                if time.time() > deadline:
                    break
                
                # Sort
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Centroid
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls += 1
                
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                        f_c = f_r
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                        f_c = fvals[-1]
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls += 1
                    
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls += 1
                
                if calls >= budget:
                    break
            
            # Find best
            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx]

        # Strategy:
        # 1. Run CMA-ES with 2 random starts, using 40% of budget each.
        # 2. Polish the best with Nelder-Mead using remaining 20%.
        
        if total_budget < 20:
            x_init = rng.uniform(-5, 5, dim)
            return nelder_mead(x_init, total_budget)

        n_cma = 2
        budget_cma = total_budget // 5  # 20% each = 40% total
        budget_nm = total_budget - n_cma * budget_cma  # Remaining 60%
        
        best_x = None
        best_f = float('inf')
        
        for i in range(n_cma):
            if time.time() > deadline:
                break
            
            x_init = rng.uniform(-5, 5, dim)
            x_run, f_run = cmaes(x_init, budget_cma, sigma0=2.0)
            
            if f_run < best_f:
                best_f = f_run
                best_x = x_run
        
        # Polish with Nelder-Mead
        if time.time() < deadline and budget_nm > 0:
            x_final, f_final = nelder_mead(best_x, budget_nm, initial_step=0.3)
            if f_final < best_f:
                best_f = f_final
                best_x = x_final
        
        return best_x, best_f
# EVOLVE-BLOCK-END
