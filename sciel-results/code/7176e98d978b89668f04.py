"""Combined optimizer: global sampling + CMA-ES + local polish for BBOB f6."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Global-to-local optimizer: stratified/random sampling then CMA-ES + Nelder-Mead."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        t_start = time.time()
        DEADLINE = 300.0
        SAFE = 5.0

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x_clipped = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x_clipped))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        def time_left():
            return (DEADLINE - SAFE) - (time.time() - t_start)

        # ---- Phase 1: global exploration ----
        # Stratified grid over [-4, 4]^dim (2 points per axis)
        n_global = max(10, min(80, budget // 5))
        
        grid = []
        for i in range(2 ** dim):
            pts = []
            for d in range(dim):
                bit = (i >> d) & 1
                pts.append(-4.0 + bit * 8.0)
            grid.append(pts)
        grid = np.array(grid)
        
        # Include origin
        grid = np.vstack([grid, np.zeros(dim)])
        
        for p in grid:
            if time_left() <= 0 or calls >= n_global:
                break
            eval_func(p)
        
        # Random fill to reach n_global
        while calls < n_global and time_left() > 0:
            p = rng.uniform(-4.0, 4.0, dim)
            eval_func(p)

        if calls >= budget or time_left() <= 10:
            return best_x, best_f

        # ---- Phase 2: CMA-ES ----
        def cmaes(x0, sigma0, max_evals):
            nonlocal calls, best_f, best_x
            
            if max_evals < 20:
                return
            
            n = dim
            lambda_pop = 4 + int(3 * np.log(n))
            mu = lambda_pop // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights ** 2)
            
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            
            x_mean = x0.copy()
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            sigma = sigma0
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            
            def compute_sqrt_C(C):
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                return eigvecs @ np.diag(np.sqrt(eigvals))
            
            sqrt_C = compute_sqrt_C(C)
            
            n_evals = 0
            stagnant_generations = 0
            best_in_cma = best_f
            best_x_in_cma = best_x.copy()
            
            while n_evals < max_evals and time_left() > 5.0:
                z = rng.standard_normal((lambda_pop, n))
                offsprings = x_mean + sigma * (z @ sqrt_C.T)
                
                fitness = np.zeros(lambda_pop)
                for i in range(lambda_pop):
                    if n_evals >= max_evals or time_left() <= 5.0:
                        break
                    fitness[i] = eval_func(offsprings[i])
                    n_evals += 1
                
                indices = np.argsort(fitness)
                x_sorted = offsprings[indices]
                f_sorted = fitness[indices]
                
                if f_sorted[0] < best_in_cma:
                    best_in_cma = f_sorted[0]
                    best_x_in_cma = x_sorted[0].copy()
                    stagnant_generations = 0
                else:
                    stagnant_generations += 1
                
                x_mean_new = x_mean + sigma * sum(weights[i] * x_sorted[i] for i in range(mu))
                
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (sqrt_C.T @ (x_mean_new - x_mean))
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (n_evals + 1) / lambda_pop)) < (c1 + 1) * chi_n + 0.3:
                    pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * (x_mean_new - x_mean) / sigma
                
                diff = x_sorted[:mu] - x_mean
                C = (1 + c1 * cmu - c1) * C + c1 * np.outer(pc, pc) + cmu * (diff * weights[:, None]).T @ diff
                
                C = (C + C.T) / 2
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-10:
                    C += 1e-10 * np.eye(n)
                
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 1e10)
                
                x_mean = x_mean_new
                sqrt_C = compute_sqrt_C(C)
                
                if stagnant_generations > 25:
                    break
                if sigma < 1e-12:
                    break
            
            if best_in_cma < best_f:
                best_f = best_in_cma
                best_x = best_x_in_cma.copy()
            return best_x_in_cma, best_in_cma

        remaining_budget = budget - calls
        if remaining_budget > 50:
            cma_budget = int(remaining_budget * 0.75)
            if cma_budget > 30:
                cmaes(best_x.copy(), 1.0, cma_budget)

        # ---- Phase 3: Local polish with Nelder-Mead ----
        remaining_budget = budget - calls
        if remaining_budget > 20 and time_left() > 10.0:
            try:
                def objective(z):
                    return eval_func(z)
                
                res = minimize(
                    objective,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining_budget,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
