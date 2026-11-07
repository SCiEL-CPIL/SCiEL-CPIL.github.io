import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_ok():
            return time.time() < deadline

        # --- Latin Hypercube Sampling for initial diverse starts ---
        def lhc_sample(n, d):
            """Generate n points in [0,1]^d using Latin Hypercube Sampling."""
            points = np.zeros((n, d))
            for j in range(d):
                perm = np.random.permutation(n)
                points[:, j] = (perm + np.random.uniform(size=n)) / n
            return points

        # --- Nelder-Mead with bounds ---
        def nelder_mead(func, x0, max_iter=200, tol=1e-8):
            """Bounded Nelder-Mead optimization."""
            nonlocal calls, best_x, best_f
            dim = len(x0)
            # Initial simplex
            simplex = [x0.copy()]
            f0 = func(x0)
            calls += 1
            update_best(x0, f0)
            simplex_f = [f0]
            
            # Expand simplex
            for i in range(dim):
                x_new = x0.copy()
                step = 0.1 if x0[i] >= 0 else -0.1
                x_new[i] += step
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                calls += 1
                update_best(x_new, f_new)
                simplex.append(x_new.copy())
                simplex_f.append(f_new)
            
            simplex = np.array(simplex)
            simplex_f = np.array(simplex_f)
            
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            
            for iteration in range(max_iter):
                if not time_ok() or calls >= budget:
                    break
                
                # Sort
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]
                
                # Check convergence
                if simplex_f[-1] - simplex_f[0] < tol:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls += 1
                update_best(x_r, f_r)
                
                if simplex_f[0] <= f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                elif f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls += 1
                    update_best(x_e, f_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls += 1
                    update_best(x_c, f_c)
                    
                    if f_c < simplex_f[-1]:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = func(simplex[i])
                            calls += 1
                            update_best(simplex[i], simplex_f[i])
                            if not time_ok() or calls >= budget:
                                break
                        if not time_ok() or calls >= budget:
                            break
            
            # Return best point found
            return simplex[np.argmin(simplex_f)].copy(), simplex_f.min()

        # --- CMA-ES for global search ---
        def cma_es(x_start, sigma_start, budget_cma):
            """CMA-ES optimization."""
            nonlocal calls, best_x, best_f
            
            lam = max(4 + int(3 * np.log(dim)), 8)
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            chi_n = np.sqrt(dim) * (1 - 1. / (4. * dim) + 1. / (21. * dim ** 2))
            
            c_sigma = (lam / 2 + 1) / (dim + 4 + lam / 2)
            c_c = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + lam / 2)
            
            x_mean = x_start.copy()
            sigma = sigma_start
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C_cholesky = np.eye(dim)
            
            while calls < budget_cma and time_ok():
                # Generate population
                Y = np.zeros((lam, dim))
                for i in range(lam):
                    z = np.random.randn(dim)
                    Y[i] = x_mean + sigma * (C_cholesky @ z)
                    Y[i] = np.clip(Y[i], -5, 5)
                
                # Evaluate
                F = np.zeros(lam)
                for i in range(lam):
                    if calls >= budget_cma or not time_ok():
                        F[i] = np.inf
                        continue
                    F[i] = func(Y[i])
                    calls += 1
                    update_best(Y[i], F[i])
                
                if calls >= budget_cma or not time_ok():
                    break
                
                # Sort
                order = np.argsort(F)
                Y = Y[order]
                F = F[order]
                
                # Update mean
                x_new = np.sum(weights[:, np.newaxis] * Y[:mu], axis=0)
                
                # Update step-size
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (C_cholesky @ (x_new - x_mean) / sigma)
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1))) < (1 + 2 / (dim + 1)) * chi_n
                
                # Update covariance
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma
                C = (1 - c_1) * C + c_1 * (np.outer((x_new - x_mean) / sigma, (x_new - x_mean) / sigma) + (1 - hsig) * c_c * (2 - c_c) * np.outer(pc, pc))
                C = (C + C.T) / 2
                C += 1e-10 * np.eye(dim)
                
                try:
                    C_cholesky = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    C_cholesky = np.eye(dim)
                
                x_mean = x_new
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)
            
            return x_mean.copy()

        # --- Phase 1: Global exploration with LHS + Nelder-Mead ---
        # Use about 60% of budget for global exploration
        n_starts = 12
        budget_per_start = max(20, int(budget * 0.6 / n_starts))
        
        # Generate LHS points
        lhs_points = lhc_sample(n_starts, dim)
        # Map from [0,1]^d to [-5,5]^d
        lhs_points = lhs_points * 10 - 5
        
        # Also add some random points for diversity
        n_random = 8
        random_points = np.random.uniform(-5, 5, size=(n_random, dim))
        
        all_starts = np.vstack([lhs_points, random_points])
        budget_per_start = max(15, int(budget * 0.6 / len(all_starts)))
        
        local_best = []
        for i in range(len(all_starts)):
            if not time_ok() or calls >= budget:
                break
            
            x0 = all_starts[i]
            # Run Nelder-Mead from this start
            x_best_local, f_best_local = nelder_mead(func, x0, max_iter=50, tol=1e-6)
            local_best.append((f_best_local, x_best_local))
        
        # --- Phase 2: Select best starts and polish with CMA-ES ---
        # Sort local results by fitness
        local_best.sort(key=lambda x: x[0])
        
        # Use remaining budget for CMA-ES on top candidates
        remaining_budget = budget - calls
        n_cma_starts = min(3, len(local_best))
        
        for i in range(n_cma_starts):
            if not time_ok() or calls >= budget:
                break
            
            f_local, x_local = local_best[i]
            # Run CMA-ES from this point with small sigma
            sigma_cma = 0.5 if i == 0 else 1.0
            budget_cma = max(50, int(remaining_budget / (n_cma_starts - i)))
            x_cma_best = cma_es(x_local, sigma_cma, budget_cma)
            
            # Evaluate the CMA-ES result
            if time_ok() and calls < budget:
                f_cma = func(x_cma_best)
                calls += 1
                update_best(x_cma_best, f_cma)
        
        # --- Phase 3: Final polish with Nelder-Mead on the global best ---
        if time_ok() and calls < budget:
            x_polish, f_polish = nelder_mead(func, best_x, max_iter=100, tol=1e-10)
        
        return best_x, best_f
