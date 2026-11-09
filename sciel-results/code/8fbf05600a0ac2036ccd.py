import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        n = self.dim
        total_budget = self.budget
        
        # Split budget between CMA-ES and Nelder-Mead
        cma_budget = int(total_budget * 0.6)
        nm_budget = total_budget - cma_budget
        
        # CMA-ES parameters
        lambda_pop = max(4 + 3 * int(np.log(n)), 8)
        # For n=5, 4+3*2=10
        if lambda_pop > cma_budget:
            lambda_pop = max(4, cma_budget // 2)
        
        x_mean = np.zeros(n)
        sigma = 2.0  # Start with larger sigma to cover the box [-5, 5]^5
        
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        
        x_best = x_mean.copy()
        f_best = func(x_best)
        calls_used = 1
        
        rng = np.random.default_rng(42)
        
        # CMA-ES Loop
        while calls_used + lambda_pop <= cma_budget:
            if time.time() > deadline:
                break
            
            # Generate population
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(n)
            
            Z = rng.standard_normal((lambda_pop, n))
            Y = x_mean + sigma * (Z @ L.T)
            
            # Evaluate
            F = np.empty(lambda_pop)
            for i in range(lambda_pop):
                F[i] = func(Y[i])
                calls_used += 1
                if F[i] < f_best:
                    f_best = F[i]
                    x_best = Y[i].copy()
            
            # Sort
            idx = np.argsort(F)
            Y_sorted = Y[idx]
            F_sorted = F[idx]
            
            # Update Mean
            x_mean_new = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)
            
            # Update Covariance and Sigma
            y = (x_mean_new - x_mean) / sigma if sigma > 1e-16 else np.zeros(n)
            norm_y = np.linalg.norm(y)
            
            # Update evolution paths
            if norm_y > 1e-16:
                y_hat = y / norm_y
            else:
                y_hat = np.zeros(n)
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_hat * norm_y
            
            # h_sigma calculation
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used // lambda_pop + 1))) 
                   < (1 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1)))
            
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y
            
            # Update C
            Y_steps = (Y_sorted[:mu] - x_mean) / sigma
            rank_mu = np.sum(weights[:, None, None] * np.outer(Y_steps, Y_steps), axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - hsig) * c_c * (2 - c_c) * C) + c_mu * rank_mu
            C = (C + C.T) / 2
            
            # Update Sigma
            sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            
            x_mean = x_mean_new
            
            # Convergence check
            if sigma < 1e-10:
                break
        
        # Nelder-Mead Refinement
        if time.time() < deadline - 5:
            # Initialize simplex around x_best
            simplex = [x_best.copy()]
            delta = 0.1
            for i in range(n):
                point = x_best.copy()
                point[i] += delta
                simplex.append(point)
            
            # Evaluate simplex
            f_values = [func(p) for p in simplex]
            calls_used += n + 1
            for i in range(n + 1):
                if f_values[i] < f_best:
                    f_best = f_values[i]
                    x_best = simplex[i].copy()
            
            # Nelder-Mead iterations
            max_nm_iter = nm_budget // (n + 1)
            for _ in range(max_nm_iter):
                if time.time() > deadline:
                    break
                if calls_used >= total_budget:
                    break
                
                # Sort simplex
                indices = np.argsort(f_values)
                simplex = [simplex[i] for i in indices]
                f_values = [f_values[i] for i in indices]
                
                # Centroid of best n points
                centroid = np.mean(simplex[:n], axis=0)
                
                # Reflection
                xr = centroid + (centroid - simplex[-1])
                fr = func(xr)
                calls_used += 1
                if fr < f_best:
                    f_best = fr
                    x_best = xr.copy()
                
                if fr < f_values[0]:
                    # Expansion
                    xe = centroid + 2 * (centroid - simplex[-1])
                    fe = func(xe)
                    calls_used += 1
                    if fe < f_best:
                        f_best = fe
                        x_best = xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        f_values[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_values[-1] = fr
                elif fr < f_values[-2]:
                    # Replace worst with reflection
                    simplex[-1] = xr
                    f_values[-1] = fr
                else:
                    # Contraction
                    if fr < f_values[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                        fc = func(xc)
                        calls_used += 1
                        if fc < f_best:
                            f_best = fc
                            x_best = xc.copy()
                        if fc < fr:
                            simplex[-1] = xc
                            f_values[-1] = fc
                        else:
                            break
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (centroid - simplex[-1])
                        fc = func(xc)
                        calls_used += 1
                        if fc < f_best:
                            f_best = fc
                            x_best = xc.copy()
                        if fc < f_values[-1]:
                            simplex[-1] = xc
                            f_values[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                f_values[i] = func(simplex[i])
                                calls_used += 1
                                if f_values[i] < f_best:
                                    f_best = f_values[i]
                                    x_best = simplex[i].copy()
                            if calls_used >= total_budget:
                                break
        
        return x_best, f_best
