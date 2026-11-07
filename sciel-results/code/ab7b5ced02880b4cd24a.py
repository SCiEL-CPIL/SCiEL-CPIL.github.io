"""Global search with Differential Evolution followed by L-BFGS-B polish for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

class Optimizer:
    """DE global search + L-BFGS-B local refinement for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0

        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        rng = np.random.default_rng(12345)

        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
                val = float(func(x_clipped))
                if val < best_f:
                    best_f = val
                    best_x = x_clipped.copy()
                return val
            except Exception:
                return best_f

        def time_ok():
            return time.time() < deadline and calls < budget

        def budget_ok():
            return calls < budget

        # Phase 1: Differential Evolution for global exploration
        # DE is a population-based global optimizer.
        # We limit it to roughly 300 calls to leave room for polish.
        # popsize=10, maxiter=30 -> ~300 calls.
        
        de_obj_calls = 0
        def de_obj(x):
            nonlocal de_obj_calls
            if not time_ok() or not budget_ok():
                return best_f
            de_obj_calls += 1
            return eval_and_track(x)

        try:
            # Use Sobol initialization for better spread
            de_result = differential_evolution(
                de_obj,
                bounds=[(-5.0, 5.0)] * dim,
                popsize=10,
                maxiter=30,
                tol=1e-10,
                seed=42,
                init='sobol',
                polish=False  # We will do our own polish with L-BFGS-B
            )
            if de_result.success or de_result.x is not None:
                # Update best if DE found something better
                # de_obj already tracked best_f and best_x
                pass
        except Exception:
            pass

        if not time_ok():
            return best_x, best_f

        # Phase 2: Local Refinement with L-BFGS-B
        # Use the best point found so far (from DE)
        # L-BFGS-B is very efficient in smooth basins.
        
        def lbfgsb_obj(x):
            if not time_ok() or not budget_ok():
                return best_f
            return eval_and_track(x)

        # Run L-BFGS-B from the best point
        x0 = best_x.copy()
        try:
            minimize(
                lbfgsb_obj,
                x0,
                method='L-BFGS-B',
                bounds=[(-5.0, 5.0)] * dim,
                options={'maxiter': 100, 'maxls': 40, 'gtol': 1e-5}
            )
        except Exception:
            pass

        if not time_ok():
            return best_x, best_f

        # Phase 3: Final Polish / Check for further improvement
        # Sometimes L-BFGS-B gets stuck in a saddle or shallow local min.
        # A few more L-BFGS-B runs from slightly perturbed best_x can help.
        for _ in range(3):
            if not time_ok():
                break
            x0 = best_x + rng.normal(0.0, 0.1, size=dim)
            x0 = np.clip(x0, -5.0, 5.0)
            try:
                minimize(
                    lbfgsb_obj,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 50, 'maxls': 20, 'gtol': 1e-5}
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
