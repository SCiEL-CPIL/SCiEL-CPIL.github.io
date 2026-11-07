"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with stratified starts and coordinate descent polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        call_count = 0
        best_x = None
        best_f = float('inf')

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return best_f if best_f < float('inf') else 0.0
            call_count += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        bounds = [(-5.0, 5.0)] * dim
        rng = np.random.default_rng(42)

        # Generate diverse start points with stratified coverage
        starts = []
        # Key structural points for Rosenbrock
        starts.append(np.ones(dim))       # Unshifted minimum
        starts.append(np.zeros(dim))
        starts.append(-np.ones(dim))
        starts.append(np.full(dim, 0.5))
        starts.append(np.full(dim, -0.5))

        # Stratified random starts: divide [-4, 4] into sub-cubes
        for _ in range(20):
            x = rng.uniform(-4.0, 4.0, dim)
            starts.append(x)

        # Additional targeted starts near likely regions
        for _ in range(10):
            x = rng.normal(0.0, 1.5, dim)
            x = np.clip(x, -4.0, 4.0)
            starts.append(x)

        # Run L-BFGS-B from each start with time awareness
        for i, x0 in enumerate(starts):
            elapsed = time.time() - start_time
            if elapsed > deadline:
                break
            if call_count >= budget - 10:
                break

            remaining_time = deadline - elapsed
            # Reduce maxiter if time is tight
            if remaining_time < 30:
                maxiter = 20
            elif remaining_time < 60:
                maxiter = 35
            else:
                maxiter = 50

            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': maxiter,
                        'ftol': 1e-12,
                        'gtol': 1e-12,
                        'maxls': 30
                    }
                )
            except Exception:
                continue

        # Coordinate descent polish with remaining budget
        if best_x is not None and call_count < budget:
            x = best_x.copy()
            f = best_f
            step_sizes = [0.5, 0.1, 0.01, 0.001, 0.0001]
            
            for ss in step_sizes:
                if call_count >= budget or time.time() > deadline:
                    break
                for _ in range(3):  # 3 passes per step size
                    if call_count >= budget or time.time() > deadline:
                        break
                    for d in range(dim):
                        if call_count >= budget or time.time() > deadline:
                            break
                        for delta in [ss, -ss]:
                            if call_count >= budget:
                                break
                            x_try = x.copy()
                            x_try[d] = np.clip(x[d] + delta, -5.0, 5.0)
                            f_try = wrapped_func(x_try)
                            if f_try < f:
                                f = f_try
                                x = x_try.copy()
                                best_x = x.copy()
                                best_f = f

        return best_x, best_f
# EVOLVE-BLOCK-END
