"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from scipy.optimize import minimize
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start Powell optimizer for BBOB f23 (Katsuura)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self._call_count = 0

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        self._call_count = 0
        best_x = None
        best_f = np.inf

        class BudgetExceeded(Exception):
            pass

        def counting_func(x):
            nonlocal best_x, best_f
            self._call_count += 1
            if self._call_count > budget:
                raise BudgetExceeded()
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Define starts
        # Katsuura minimum is at x_i = 1.
        # We use a mix of informed and random starts.
        rng = np.random.default_rng(0)
        
        starts = []
        # Informed starts
        starts.append(np.ones(dim))
        starts.append(np.zeros(dim))
        starts.append(np.full(dim, 0.5))
        
        # Random starts
        # We have budget 500. Powell is fast.
        # Let's try 6 random starts.
        for _ in range(6):
            starts.append(rng.uniform(-5, 5, dim))
            
        # Shuffle starts to ensure we don't waste budget on a bad start if they are ordered poorly?
        # Actually, informed starts first is better.
        
        for start in starts:
            if self._call_count >= budget - 10: # Leave some buffer
                break
                
            try:
                res = minimize(
                    counting_func,
                    start,
                    method='Powell',
                    options={'maxiter': 50, 'xtol': 1e-10, 'ftol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except BudgetExceeded:
                pass
            except Exception:
                pass
                
        # If we still have budget, do a final polish from the best point with Powell?
        # Powell from the best point might refine further.
        
        if best_x is not None and self._call_count < budget - 10:
            try:
                res = minimize(
                    counting_func,
                    best_x,
                    method='Powell',
                    options={'maxiter': 50, 'xtol': 1e-12, 'ftol': 1e-12}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except BudgetExceeded:
                pass
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
