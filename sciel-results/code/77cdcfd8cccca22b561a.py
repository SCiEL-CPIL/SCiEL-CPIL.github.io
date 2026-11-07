"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

class Optimizer:
    """Differential Evolution global search with L-BFGS-B polish for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        time_limit = 285.0  # Safety margin for 300s limit
        calls_made = 0
        best_x = None
        best_f = np.inf
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                raise StopIteration
            if time.time() - start_time > time_limit:
                raise StopIteration
            calls_made += 1
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def check_time():
            return time.time() - start_time < time_limit

        # Phase 1: Differential Evolution for global exploration
        # Rosenbrock has a narrow valley. DE is good at navigating this if population is large enough.
        # We use a smaller budget for DE to ensure we have time for polish.
        de_budget = 350
        
        # Bounds for DE
        bounds = [(-5.0, 5.0)] * dim
        
        # Custom objective for DE that tracks calls
        def de_objective(x):
            return safe_call(x)

        try:
            # maxiter controls number of generations. 
            # popsize=15 means 15*dim=75 individuals.
            # We cap maxfun to stay within de_budget.
            res_de = differential_evolution(
                de_objective, 
                bounds, 
                seed=0, 
                maxiter=100, 
                maxfun=de_budget,
                tol=1e-8,
                popsize=15,
                init='latinhypercube',
                polish=False, # We will do our own polish
                updating='deferred',
                workers=1
            )
            if res_de.success or res_de.fun < best_f:
                if res_de.fun < best_f:
                    best_f = res_de.fun
                    best_x = res_de.x.copy()
                else:
                    # If DE didn't improve best_f (maybe it didn't call enough or converged to same), 
                    # we still use res_de.x as a good starting point for polish if it's finite
                    if best_x is None:
                        best_x = res_de.x.copy()
                        best_f = res_de.fun
                    # If best_x exists, we might want to polish the DE result specifically if it's close
                    # But safe_call already updates best_x if res_de.x is better.
                    # So best_x is the global best found so far.
        except StopIteration:
            pass
        except Exception:
            pass

        # Phase 2: L-BFGS-B Polish
        # Use the best point found by DE (or initial if DE failed)
        if best_x is None:
            # Fallback if DE failed completely
            best_x = np.zeros(dim)
            try:
                best_f = safe_call(best_x)
            except StopIteration:
                return best_x, best_f

        if check_time() and calls_made < budget - 10:
            def lbfun(x_in):
                nonlocal calls_made
                if calls_made >= budget:
                    raise StopIteration
                if time.time() - start_time > time_limit:
                    raise StopIteration
                calls_made += 1
                return float(func(x_in))

            try:
                res_lbfgs = minimize(
                    lbfun, 
                    best_x, 
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'maxfun': budget - calls_made, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res_lbfgs.fun < best_f:
                    best_f = res_lbfgs.fun
                    best_x = res_lbfgs.x.copy()
            except StopIteration:
                pass
            except Exception:
                pass

        # Phase 3: Second Polish if still not close to 0
        # Rosenbrock min is 0. If we are far, try a different start for L-BFGS
        # Sometimes L-BFGS gets stuck in local minima or flat regions.
        if check_time() and calls_made < budget - 10 and best_f > 1e-3:
            # Try perturbing the best solution
            # Rosenbrock valley is narrow, so small perturbations along the valley direction might help
            # But we don't know the direction. Just random jitter.
            x2 = np.clip(best_x + 0.5 * rng.standard_normal(dim), -5, 5)
            try:
                f2 = safe_call(x2)
                if f2 < best_f:
                    best_f = f2
                    best_x = x2.copy()

                def lbfun2(x_in):
                    nonlocal calls_made
                    if calls_made >= budget:
                        raise StopIteration
                    if time.time() - start_time > time_limit:
                        raise StopIteration
                    calls_made += 1
                    return float(func(x_in))

                res2 = minimize(
                    lbfun2, 
                    best_x, 
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'maxfun': budget - calls_made, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res2.fun < best_f:
                    best_f = res2.fun
                    best_x = res2.x.copy()
            except StopIteration:
                pass
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
