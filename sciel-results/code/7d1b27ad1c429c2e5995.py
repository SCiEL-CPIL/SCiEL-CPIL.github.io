"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize


class Optimizer:
    """Differential Evolution with Local Polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self._calls = 0
        self._max_calls = budget

    def __call__(self, func):
        self._calls = 0
        self._max_calls = self.budget
        rng = np.random.default_rng(42)

        # Wrapper to track calls and enforce budget
        def wrapped_func(x):
            if self._calls >= self._max_calls:
                return float('inf')
            self._calls += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            if not np.isfinite(val):
                return float('inf')
            return float(val)

        best_x = None
        best_f = float('inf')
        
        bounds = [(-5.0, 5.0)] * self.dim
        
        # Phase 1: Global Search with Differential Evolution
        # Budget allocation: Use ~80% for DE, ~20% for Local Polish
        # DE calls approx popsize * (maxiter + 1)
        # Let's use popsize=10, maxiter=20 -> 10*21=210 calls.
        # This leaves ~290 calls for local search.
        
        popsize = 10
        maxiter = 20
        
        try:
            res_de = differential_evolution(
                wrapped_func,
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                seed=42,
                tol=1e-6,
                atol=1e-6,
                init='latinhypercube'
            )
            if res_de.x is not None and np.isfinite(res_de.fun):
                best_x = res_de.x.copy()
                best_f = res_de.fun
        except Exception:
            pass

        # Phase 2: Local Polish
        # Use remaining budget for Nelder-Mead or L-BFGS-B
        # Nelder-Mead is robust and derivative-free.
        # L-BFGS-B is faster if gradient is available, but we are black-box.
        # Nelder-Mead is a good choice for polishing.
        
        if best_x is not None and self._calls < self._max_calls - 5:
            # We can run multiple local searches from different points if budget allows
            # But usually one good polish is enough.
            # Let's try Nelder-Mead with a generous maxiter, but bounded by remaining budget.
            
            # Estimate remaining calls
            remaining = self._max_calls - self._calls
            # Nelder-Mead uses dim+1 calls per iteration.
            # Let's allow max_iter such that (dim+1)*max_iter <= remaining * 0.8
            # dim=5, so 6 calls/iter.
            max_iter_local = int((remaining * 0.8) / (self.dim + 1))
            if max_iter_local < 10:
                max_iter_local = 10 # Minimum sensible iterations
            
            try:
                res_nm = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter_local,
                        'xatol': 1e-8,
                        'fatol': 1e-8
                    }
                )
                if res_nm.x is not None and np.isfinite(res_nm.fun):
                    if res_nm.fun < best_f:
                        best_f = res_nm.fun
                        best_x = res_nm.x.copy()
            except Exception:
                pass

        # If we still have budget, try another local search from a slightly perturbed best_x
        if best_x is not None and self._calls < self._max_calls - 5:
            perturbation = 0.1 * rng.standard_normal(self.dim)
            x0 = np.clip(best_x + perturbation, -5.0, 5.0)
            
            remaining = self._max_calls - self._calls
            max_iter_local = int((remaining * 0.8) / (self.dim + 1))
            if max_iter_local >= 10:
                try:
                    res_nm2 = minimize(
                        wrapped_func,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': max_iter_local,
                            'xatol': 1e-8,
                            'fatol': 1e-8
                        }
                    )
                    if res_nm2.x is not None and np.isfinite(res_nm2.fun):
                        if res_nm2.fun < best_f:
                            best_f = res_nm2.fun
                            best_x = res_nm2.x.copy()
                except Exception:
                    pass

        # Fallback if no valid point found
        if best_x is None:
            best_x = rng.uniform(-5, 5, self.dim)
            if self._calls < self._max_calls:
                self._calls += 1
                best_f = wrapped_func(best_x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
