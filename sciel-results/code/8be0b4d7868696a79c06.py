import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 270.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        lb = np.full(dim, -5.0)
        ub = np.full(dim, 5.0)

        call_count = [0]
        best_f = [np.inf]
        best_x = [np.zeros(dim)]
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            if call_count[0] >= budget:
                raise BudgetExceeded()
            call_count[0] += 1
            x = np.asarray(x, dtype=float)
            fx = func(x)
            if np.isnan(fx) or np.isinf(fx):
                fx = np.inf
            if fx < best_f[0]:
                best_f[0] = fx
                best_x[0] = x.copy()
            return fx

        class BudgetExceeded(Exception):
            pass

        # CMA-ES implementation (proven effective, pop=12, standard sigma, early restart)
        def cma_es(func_wrapper, dim, lb, ub, max_evals, seed=42):
            rng_local = np.random.default_rng(seed)
            
            # Initial point at origin (center of box)
            x = np.zeros(dim)
            
            # Initial step size
            sigma = 1.0
            
            # Population size: reduced to 12 for stability
            lambda_ = 12
            mu = lambda_ // 2
            
            # Evolution path weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = int(1 / np.sum(weights**2))
            
            # Learning rates
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 2 / mueff) / ((dim + 2) ** 2 + mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
            
            # State variables
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C = np.eye(dim)
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            # Evaluate initial point
            f0 = func_wrapper(x)
            
            gen = 0
            no_improve = 0
            best_f_local = f0
            best_x_local = x.copy()
            
            while call_count[0] < max_evals and time.time() < deadline:
                gen += 1
                
                # Generate population
                D = np.sqrt(np.diag(C))
                B = np.linalg.inv(np.linalg.cholesky(C + 1e-10 * np.eye(dim)))
                
                y = np.zeros((lambda_, dim))
                fvals = np.zeros(lambda_)
                
                for i in range(lambda_):
                    if call_count[0] >= max_evals or time.time() > deadline:
                        break
                    zi = rng_local.standard_normal(dim)
                    xi = x + sigma * (B @ (D * zi))
                    xi = np.clip(xi, lb, ub)
                    fvals[i] = func_wrapper(xi)
                    y[i] = zi
                
                # Sort by function value
                order = np.argsort(fvals)
                fvals_sorted = fvals[order]
                y_sorted = y[order]
                
                # Update best
                if fvals_sorted[0] < best_f_local:
                    best_f_local = fvals_sorted[0]
                    best_x_local = np.clip(x + sigma * (B @ (D * y_sorted[0])), lb, ub)
                    no_improve = 0
                else:
                    no_improve += 1
                
                # Restart if stagnated for 12 generations
                if no_improve >= 12:
                    break
                
                # Update mean
                x = x + sigma * np.sum(weights[:, np.newaxis] * y_sorted[:mu], axis=0)
                x = np.clip(x, lb, ub)
                
                # Update covariance matrix adaptation
                ymean = np.sum(weights[:, np.newaxis] * y_sorted[:mu], axis=0)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C @ (sigma * ymean / sigma))
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (gen + 1)) / chiN < (1 + cc) * 1.4 + 1 / dim
                pc = (1 - cc) * pc + hsig * cc * ymean
                
                delta = hsig * cc * (1 - cc) * ymean
                C = (1 + c1 * (np.outer(pc, pc) - C) + cmu * np.sum(weights[:mu, np.newaxis, np.newaxis] * np.outer(y_sorted[:mu], y_sorted[:mu]), axis=0) - (c1 + cmu) * C)
                
                # Update step size
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))
                sigma = max(1e-10, min(sigma, 10.0))
            
            return best_x_local.copy(), best_f_local

        # Phase 1: Run CMA-ES with most of the budget
        cma_budget = int(budget * 0.75)
        
        try:
            cx, cf = cma_es(wrapped_func, dim, lb, ub, cma_budget, seed=42)
            if cx is not None:
                if cf < best_f[0]:
                    best_f[0] = cf
                    best_x[0] = cx.copy()
        except (TimeoutError, BudgetExceeded):
            pass
        except Exception:
            pass

        # Phase 2: Local refinement with Nelder-Mead if budget remains
        if best_x[0] is not None and call_count[0] < budget:
            try:
                from scipy.optimize import minimize
                remaining = budget - call_count[0]
                maxiter = min(200, remaining // 2)
                if maxiter > 10:
                    res = minimize(
                        lambda x: wrapped_func(x),
                        best_x[0],
                        method='Nelder-Mead',
                        options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                    )
            except (TimeoutError, BudgetExceeded):
                pass
            except Exception:
                pass

        # Phase 3: Final local search with small perturbations
        if call_count[0] < budget and best_x[0] is not None:
            try:
                for _ in range(20):
                    if call_count[0] >= budget or time.time() > deadline:
                        break
                    # Small random perturbations around best
                    scale = max(0.01, (best_f[0] - (-100)) / 200 * 0.1)
                    x_trial = best_x[0] + rng.normal(0, scale, dim)
                    x_trial = np.clip(x_trial, lb, ub)
                    f = wrapped_func(x_trial)
            except (TimeoutError, BudgetExceeded):
                pass
            except Exception:
                pass

        return best_x[0], best_f[0]
