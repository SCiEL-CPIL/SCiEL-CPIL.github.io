"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import dual_annealing, basinhopping


class Optimizer:
    """Derivative-free global optimizer using Dual Annealing and Basin Hopping."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Track best solution and value
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)

        def safe_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return best_f
            calls_made += 1
            try:
                val = func(np.asarray(x, dtype=np.float64))
                val = float(val)
                if np.isfinite(val):
                    if val < best_f:
                        best_f = val
                        best_x = np.clip(np.asarray(x, dtype=np.float64), -5, 5).copy()
                    return val
                else:
                    return 1e10
            except Exception:
                return 1e10

        bounds = [(-5.0, 5.0) for _ in range(dim)]

        # Phase 1: Global exploration with Dual Annealing
        # Dual Annealing is a robust derivative-free global optimizer.
        # We set maxiter and maxfun to consume a portion of the budget.
        # The objective function is called multiple times per iteration, so we cap maxfun.
        
        # We want to use roughly 60% of the budget for the main global search.
        # Since dual_annealing uses a lot of calls, we set maxfun to a fraction.
        # Note: maxfun in dual_annealing is the total number of objective function calls.
        
        try:
            # maxfun is the total number of function evaluations allowed.
            # We allocate 300 calls to the main global search.
            maxfun_global = min(300, budget - 50)
            
            res = dual_annealing(
                safe_func,
                bounds,
                maxfun=maxfun_global,
                maxiter=100, # Cap iterations to avoid infinite loops if maxfun is high
                seed=42,
                no_local_search=False,
                initial_temp=5230.0,
                restart_temp_ratio=2e-5,
                visit=2.62,
                accept=0.66,
                policy='fast'
            )
            if res.fun < best_f:
                best_f = float(res.fun)
                best_x = np.clip(res.x, -5, 5).copy()
        except Exception:
            pass

        # Phase 2: Local refinement with Basin Hopping
        # Basin hopping is excellent for finding the global minimum in rugged landscapes
        # and uses a local minimizer (L-BFGS-B by default) internally.
        # However, L-BFGS-B uses numerical gradients, which costs 2N+1 function calls per step.
        # For N=5, that's 11 calls per step. 
        # We can use a custom local minimizer or just rely on the built-in one with tight bounds.
        
        # Let's try a few short basin hopping runs from the current best and a few random points.
        n_bh_runs = 3
        for i in range(n_bh_runs):
            if time.time() > deadline or calls_made >= budget - 20:
                break
            
            if i == 0:
                x0 = best_x.copy()
            else:
                x0 = rng.uniform(-5, 5, dim)
            
            try:
                # Basin hopping uses a local minimizer. 
                # We can pass a custom local minimizer to avoid the high cost of numerical gradients
                # or just use the default with strict maxiter.
                # Let's use the default 'L-BFGS-B' but with very few iterations to save budget,
                # or switch to 'Nelder-Mead' which is cheaper per step but slower convergence.
                # Given the small dimension (5), Nelder-Mead might be more efficient in terms of raw calls.
                
                # Let's define a wrapper for basin hopping that tracks calls strictly.
                # Actually, basin_hopping's local minimizer is called repeatedly.
                # To be safe, we'll limit the total calls remaining.
                
                res_bh = basinhopping(
                    safe_func,
                    x0,
                    stepsize=0.5,
                    niter=min(50, max(1, (budget - calls_made) // 20)), # Limit iterations
                    minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4}},
                    seed=42 + i
                )
                if res_bh.fun < best_f:
                    best_f = float(res_bh.fun)
                    best_x = np.clip(res_bh.x, -5, 5).copy()
            except Exception:
                pass

        # Phase 3: Final Polish with Nelder-Mead (Derivative-free, cheap per step)
        # Use remaining budget for a high-precision local search around the best found.
        if calls_made < budget - 5 and time.time() < deadline - 5:
            try:
                from scipy.optimize import minimize
                # Nelder-Mead doesn't use gradients, so it's efficient in terms of call count for small N.
                # We use a simple minimizer call.
                res_nm = minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max(10, budget - calls_made),
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                if res_nm.fun < best_f:
                    best_f = float(res_nm.fun)
                    best_x = np.clip(res_nm.x, -5, 5).copy()
            except Exception:
                pass

        # Ensure best_x is valid
        if best_x is None or not np.all(np.isfinite(best_x)):
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        return best_x, best_f


# EVOLVE-BLOCK-END
