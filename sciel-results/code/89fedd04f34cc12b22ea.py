import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        x_best = np.zeros(self.dim)
        f_best = func(x_best)
        calls = 1

        if self.budget < 10:
            for _ in range(self.budget - calls):
                if time.time() > deadline:
                    break
                x = np.random.uniform(-5, 5, self.dim)
                fx = func(x)
                if fx < f_best:
                    f_best = fx
                    x_best = x
                calls += 1
            return x_best, f_best

        # Determine gradient signs with 1 probe per dimension
        probe_delta = 1.0  # Larger delta for more robust sign detection
        signs = np.zeros(self.dim, dtype=int)  # 1 for +5, -1 for -5
        vertex = np.zeros(self.dim)

        for i in range(self.dim):
            if time.time() > deadline or calls >= self.budget:
                break

            # Single probe: evaluate at x_best + delta * e_i
            x_probe = x_best.copy()
            x_probe[i] += probe_delta
            x_probe = np.clip(x_probe, -5, 5)
            f_probe = func(x_probe)
            calls += 1

            if f_probe < f_best:
                # Gradient is negative in this direction → min is at +5
                signs[i] = 1
                vertex[i] = 5.0
            else:
                # Gradient is positive in this direction → min is at -5
                signs[i] = -1
                vertex[i] = -5.0

            # Evaluate partial vertex with all known signs so far
            if i >= 0:  # Always true, but for clarity
                x_partial = np.zeros(self.dim)
                for j in range(i + 1):
                    x_partial[j] = vertex[j]
                # For dimensions not yet probed, keep at 0
                x_partial = np.clip(x_partial, -5, 5)
                f_partial = func(x_partial)
                calls += 1

                if f_partial < f_best:
                    f_best = f_partial
                    x_best = x_partial.copy()

        # Evaluate the full vertex
        if calls < self.budget and time.time() <= deadline:
            f_vertex = func(vertex)
            calls += 1
            if f_vertex < f_best:
                f_best = f_vertex
                x_best = vertex.copy()

        # Local refinement with remaining budget
        if calls < self.budget:
            sigma = 0.5
            while calls < self.budget and time.time() <= deadline:
                y = x_best + sigma * np.random.randn(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1

                if fy < f_best:
                    f_best = fy
                    x_best = y
                    sigma *= 1.2
                else:
                    sigma *= 0.7
                    if sigma < 1e-6:
                        break

        return x_best, f_best
