"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic grid search + fixed-step local hill climb."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill
        n = self.dim
        budget = self.budget
        calls = 0

        def safe_call(x):
            nonlocal calls
            if calls >= budget:
                return None
            calls += 1
            return func(x)

        # Phase 1: Deterministic grid search over 2^n vertices + center + axis points
        # Vertices: all combinations of -5 and 5
        vertices = []
        for mask in range(2 ** n):
            v = np.array([5.0 if (mask >> i) & 1 else -5.0 for i in range(n)])
            vertices.append(v)
        # Add center
        vertices.append(np.zeros(n))
        # Add axis points: +/-3 along each axis
        for i in range(n):
            for val in [-3.0, 3.0]:
                p = np.zeros(n)
                p[i] = val
                vertices.append(p)

        # Evaluate all grid points
        best_x = None
        best_f = float('inf')
        for v in vertices:
            if calls >= budget or time.time() > deadline:
                break
            f = safe_call(v)
            if f is not None and f < best_f:
                best_f = f
                best_x = v.copy()

        if best_x is None:
            # Fallback: evaluate at origin
            best_x = np.zeros(n)
            best_f = safe_call(best_x)

        # Phase 2: Fixed-step local hill climb from best point
        # Use adaptive step sizes: start large, decrease
        step_sizes = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
        for step in step_sizes:
            improved = True
            while improved and calls < budget and time.time() < deadline:
                improved = False
                for i in range(n):
                    for direction in [-1, 1]:
                        y = best_x.copy()
                        y[i] += direction * step
                        y = np.clip(y, -5, 5)
                        fy = safe_call(y)
                        if fy is not None and fy < best_f:
                            best_x = y
                            best_f = fy
                            improved = True
                            if calls >= budget or time.time() > deadline:
                                break
                    if calls >= budget or time.time() > deadline:
                        break
                if calls >= budget or time.time() > deadline:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
