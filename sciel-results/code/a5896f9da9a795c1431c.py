"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-restart Nelder-Mead with adaptive restarts for BBOB Sphere."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        calls = [0]
        best_x = None
        best_f = np.inf

        def wrapped(x):
            nonlocal best_f, best_x
            calls[0] += 1
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Phase 1: Random exploration to find multiple starting points
        n_explore = 20
        explore_results = []
        for i in range(n_explore):
            if calls[0] >= budget:
                break
            if time.time() > deadline:
                break
            if i == 0:
                p = np.zeros(d)
            else:
                p = rng.uniform(-5, 5, d)
            f = wrapped(p)
            explore_results.append((f, p.copy()))

        explore_results.sort(key=lambda t: t[0])

        # Phase 2: Multi-restart Nelder-Mead
        # Use the top 3 exploration points as starting points for NM
        # Each restart gets a portion of the remaining budget
        
        remaining_budget = budget - calls[0]
        if remaining_budget <= 0 or time.time() > deadline:
            return best_x, best_f

        # Split remaining budget among up to 3 restarts
        n_restarts = min(3, len(explore_results))
        budget_per_restart = remaining_budget // n_restarts
        
        # If budget_per_restart is too small, use just 1 restart with all budget
        if budget_per_restart < 30:
            n_restarts = 1
            budget_per_restart = remaining_budget

        for r in range(n_restarts):
            if calls[0] >= budget:
                break
            if time.time() > deadline:
                break
            
            # Starting point for this restart
            if r < len(explore_results):
                x0 = explore_results[r][1].copy()
            else:
                x0 = best_x.copy() if best_x is not None else np.zeros(d)
            
            # Nelder-Mead parameters
            # Initial simplex: x0 plus d points offset by small step
            initial_simplex = [x0.copy()]
            for j in range(d):
                x_j = x0.copy()
                x_j[j] += 0.1 * (5 - x0[j]) if x0[j] >= 0 else 0.1 * (5 + x0[j])
                x_j = np.clip(x_j, -5, 5)
                initial_simplex.append(x_j)
            
            # Run Nelder-Mead with budget limit
            nm_calls = [0]
            nm_best_f = np.inf
            nm_best_x = None
            
            def nm_wrapper(x):
                nonlocal nm_best_f, nm_best_x
                nm_calls[0] += 1
                x = np.clip(np.asarray(x, dtype=float), -5, 5)
                f = float(func(x))
                if f < nm_best_f:
                    nm_best_f = f
                    nm_best_x = x.copy()
                return f
            
            # Use scipy's Nelder-Mead with maxiter limit
            try:
                # maxiter in scipy corresponds to number of iterations, not function calls
                # Each iteration uses at most d+1 function calls, so maxiter ~ budget/(d+1)
                maxiter = max(1, budget_per_restart // (d + 1))
                res = minimize(nm_wrapper, x0, method='Nelder-Mead',
                              options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8,
                                       'adaptive': True})
            except Exception:
                pass
            
            # Update global best
            if nm_best_x is not None and nm_best_f < best_f:
                best_f = nm_best_f
                best_x = nm_best_x.copy()
            
            # If we still have budget and this restart made progress, consider another restart
            # from the best point found so far with a different initial simplex scale
            if calls[0] < budget and time.time() < deadline and r < n_restarts - 1:
                # Check if we should continue with next restart
                continue
        
        # Phase 3: Final polish with L-BFGS-B if we have budget left
        if calls[0] < budget and time.time() < deadline and best_x is not None:
            try:
                x0 = best_x.copy()
                remaining = budget - calls[0]
                maxfun = min(remaining, 100)
                
                lbg_calls = [0]
                def lbg_wrapper(x):
                    lbg_calls[0] += 1
                    x = np.clip(np.asarray(x, dtype=float), -5, 5)
                    f = float(func(x))
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    return f

                res = minimize(lbg_wrapper, x0, method='L-BFGS-B',
                              bounds=[(-5, 5)] * d,
                              options={'maxfun': maxfun, 'maxiter': maxfun,
                                       'ftol': 1e-14, 'gtol': 1e-14})
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
