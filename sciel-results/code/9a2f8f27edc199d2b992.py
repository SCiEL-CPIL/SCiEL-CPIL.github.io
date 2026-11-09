"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead with coordinate-descent polish and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit

        dim = self.dim
        budget = self.budget

        # Wrap func to count calls and enforce budget
        call_count = [0]
        best_x = [None]
        best_f = [np.inf]

        def wrapped(x):
            if call_count[0] >= budget:
                return np.inf
            call_count[0] += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            if f < best_f[0]:
                best_f[0] = f
                best_x[0] = x.copy()
            return f

        # Generate diverse initial points
        rng = np.random.default_rng(42)
        n_starts = 12
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Axis-aligned points
        for i in range(dim):
            for sign in [1, -1]:
                p = np.zeros(dim)
                p[i] = sign * 3.0
                starts.append(p)
        # Random points
        while len(starts) < n_starts:
            starts.append(rng.uniform(-4.5, 4.5, dim))

        # Phase 1: Nelder-Mead from each start
        for i, x0 in enumerate(starts):
            if time.time() > deadline - 10:
                break
            if call_count[0] >= budget:
                break
            try:
                res = minimize(
                    wrapped,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'maxfev': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
            except Exception:
                pass

        # Phase 2: Coordinate descent polish on the best point
        if best_x[0] is not None and time.time() < deadline - 5 and call_count[0] < budget:
            x = best_x[0].copy()
            f = best_f[0]
            step = 0.5
            for _ in range(200):
                if time.time() > deadline - 3:
                    break
                if call_count[0] >= budget:
                    break
                improved = False
                for d in range(dim):
                    if call_count[0] >= budget:
                        break
                    for sign in [1, -1]:
                        trial = x.copy()
                        trial[d] = np.clip(trial[d] + sign * step, -5.0, 5.0)
                        tf = wrapped(trial)
                        if tf < f:
                            x = trial
                            f = tf
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-10:
                        break
                else:
                    step *= 0.9

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END
