"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer for BBOB f12 (Bent Cigar).
    
    Strategy:
    1. Phase 1: Random sampling to locate the narrow valley (global exploration).
    2. Phase 2: L-BFGS-B with finite-difference gradients to exploit curvature 
       and rapidly converge along the valley (fast local convergence).
    3. Phase 3: Nelder-Mead polish for final precision if budget allows.
    
    This combines the structural idea of random restarts/sampling with the 
    refinement machinery of second-order quasi-Newton methods.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        # Track calls to ensure we don't exceed budget
        call_count = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Random sampling to find a good starting point
        # Bent Cigar has a narrow valley; random sampling helps locate it
        rng = np.random.default_rng(42)
        n_samples = 20
        for i in range(n_samples):
            if time.time() > deadline:
                break
            if call_count >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = wrapped_func(x)

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = wrapped_func(best_x)

        # Phase 2: Local optimization with L-BFGS-B
        # L-BFGS-B uses quasi-Newton approximation to adaptively scale steps,
        # which is crucial for the anisotropic valley of Bent Cigar
        bounds = [(-5.0, 5.0) for _ in range(self.dim)]
        remaining_budget = self.budget - call_count

        try:
            if remaining_budget > 10:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 500,
                        'maxfev': remaining_budget,
                        'ftol': 1e-15,
                        'gtol': 1e-15,
                        'eps': 1e-8  # Finite difference step
                    }
                )
                # Update best if L-BFGS-B found something better
                if hasattr(result, 'fun') and result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
        except Exception:
            pass

        # Phase 3: Final high-precision polish pass
        # Use Nelder-Mead with tight tolerances for a final squeeze
        remaining_budget = self.budget - call_count
        if remaining_budget > 5 and time.time() < deadline:
            try:
                result2 = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(remaining_budget, 100),
                        'xatol': 1e-12,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if hasattr(result2, 'fun') and result2.fun < best_f:
                    best_f = result2.fun
                    best_x = result2.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
