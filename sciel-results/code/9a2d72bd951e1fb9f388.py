"""Black-box continuous optimizer on the BBOB-style function f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES optimizer with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0  # safety margin before 300s limit
        rng = np.random.default_rng(12345)
        d = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0
        time_exceeded = False

        def wrapped_func(x):
            nonlocal call_count, time_exceeded
            if call_count >= budget or time.time() > deadline:
                time_exceeded = True
                return np.inf
            call_count += 1
            try:
                fx = func(x)
                return fx
            except Exception:
                return np.inf

        # Multi-start strategy with adaptive budget allocation
        # Use fewer restarts with more evaluations each, then a final polish
        n_restarts = 8
        # Reserve ~15% of budget for final polish
        polish_budget = max(30, budget // 6)
        search_budget = budget - polish_budget
        evals_per_restart = search_budget // n_restarts
        if evals_per_restart < 15:
            n_restarts = max(1, search_budget // 15)
            evals_per_restart = search_budget // n_restarts

        for i in range(n_restarts):
            if call_count >= budget or time.time() > deadline:
                break
            
            # Random start point, slightly biased toward center for first few
            if i < 3:
                x0 = rng.uniform(-3, 3, d)
            else:
                x0 = rng.uniform(-5, 5, d)
            
            # Initial evaluation
            fx0 = wrapped_func(x0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()
            
            if time_exceeded or call_count >= budget:
                break

            # Run Nelder-Mead with relaxed tolerances for exploration
            try:
                res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                               options={'maxiter': evals_per_restart, 
                                        'xatol': 1e-3, 
                                        'fatol': 1e-3,
                                        'adaptive': True,
                                        'initial_simplex': None})
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                        
            except Exception:
                pass

            if time_exceeded or call_count >= budget:
                break

        # Final high-precision polish pass on the best point found
        if best_x is not None and not time_exceeded and call_count < budget:
            try:
                res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': polish_budget, 
                                        'xatol': 1e-6, 
                                        'fatol': 1e-6,
                                        'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Fallback
        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
