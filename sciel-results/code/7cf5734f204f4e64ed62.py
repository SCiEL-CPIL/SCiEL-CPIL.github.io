"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic space-filling + adaptive coordinate refinement for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def try_point(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline

        # Phase 1: Deterministic space-filling sampling (Halton sequence)
        n_initial = min(200, budget // 3)
        halton_points = self._halton_sequence(n_initial, dim)
        for i in range(n_initial):
            if not time_ok() or calls_made >= budget:
                break
            try_point(halton_points[i])

        # Phase 2: Adaptive coordinate-wise refinement with shrinking step sizes
        if best_x is not None and time_ok() and calls_made < budget:
            # Multiple refinement passes with decreasing step sizes
            step_sizes = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
            for step in step_sizes:
                if not time_ok() or calls_made >= budget:
                    break
                # Iterate over coordinates multiple times within this step size
                n_passes = max(1, int((budget - calls_made) / (dim * 4)))
                for _ in range(n_passes):
                    if not time_ok() or calls_made >= budget:
                        break
                    for d in range(dim):
                        if not time_ok() or calls_made >= budget:
                            break
                        local_best_x = best_x.copy()
                        local_best_f = best_f
                        # Try ±step in this coordinate
                        for sign in [-1, 1]:
                            if not time_ok() or calls_made >= budget:
                                break
                            x_try = local_best_x.copy()
                            x_try[d] = local_best_x[d] + sign * step
                            x_try = np.clip(x_try, -5.0, 5.0)
                            fx = func(x_try)
                            calls_made += 1
                            if fx < local_best_f:
                                local_best_f = fx
                                local_best_x = x_try.copy()
                        if local_best_f < best_f:
                            best_f = local_best_f
                            best_x = local_best_x.copy()

        # Phase 3: Fine local search with very small steps
        if best_x is not None and time_ok() and calls_made < budget:
            fine_steps = [0.0078125, 0.00390625, 0.001953125, 0.0009765625]
            for step in fine_steps:
                if not time_ok() or calls_made >= budget:
                    break
                n_passes = max(1, int((budget - calls_made) / (dim * 4)))
                for _ in range(n_passes):
                    if not time_ok() or calls_made >= budget:
                        break
                    for d in range(dim):
                        if not time_ok() or calls_made >= budget:
                            break
                        local_best_x = best_x.copy()
                        local_best_f = best_f
                        for sign in [-1, 1]:
                            if not time_ok() or calls_made >= budget:
                                break
                            x_try = local_best_x.copy()
                            x_try[d] = local_best_x[d] + sign * step
                            x_try = np.clip(x_try, -5.0, 5.0)
                            fx = func(x_try)
                            calls_made += 1
                            if fx < local_best_f:
                                local_best_f = fx
                                local_best_x = x_try.copy()
                        if local_best_f < best_f:
                            best_f = local_best_f
                            best_x = local_best_x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made = 1

        return best_x, best_f

    def _halton_sequence(self, n, dim):
        """Generate n points from the Halton sequence in [0,1]^dim, mapped to [-5,5]."""
        points = np.zeros((n, dim))
        for d in range(dim):
            base = d + 2  # primes: 2, 3, 5, 7, 11, ...
            for i in range(n):
                f = 1.0
                r = 0.0
                i_val = i + 1
                while i_val > 0:
                    f /= base
                    r += f * (i_val % base)
                    i_val //= base
                points[i, d] = r
        # Map from [0,1] to [-5,5]
        points = -5.0 + 10.0 * points
        return points
# EVOLVE-BLOCK-END
