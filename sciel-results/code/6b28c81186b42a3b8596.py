"""Black-box continuous optimizer on the BBOB-style function f12 (Bent cigar)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Powell optimizer with adaptive step sizes for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        # Track all function evaluations
        self._eval_count = 0
        self._best_x = None
        self._best_f = float('inf')

        def tracked_func(x):
            self._eval_count += 1
            x = np.asarray(x, dtype=float)
            # Clip to box
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Generate multiple starting points using quasi-random (Sobol-like via Halton)
        # Use a mix of random and structured starts
        num_starts = min(10, self.budget // 50)  # at most 10 starts, each with ~50 budget
        start_points = []

        # First start: center
        start_points.append(np.zeros(self.dim))

        # Add some structured starts
        for i in range(1, num_starts):
            # Halton sequence for low-discrepancy
            x = np.array([self._halton(i + 1, base) for base in range(2, self.dim + 2)])
            # Map [0,1) to [-5, 5]
            x = x * 10.0 - 5.0
            start_points.append(x)

        # Add a few random starts
        for i in range(num_starts, 15):
            x = rng.uniform(-4, 4, self.dim)
            start_points.append(x)

        # Sort by distance from center (prefer closer starts first for early progress)
        start_points.sort(key=lambda x: np.linalg.norm(x))

        best_x = None
        best_f = float('inf')

        for start in start_points:
            if time.time() > deadline:
                break
            if self._eval_count >= self.budget - 10:
                break

            x0 = np.clip(start, -5.0, 5.0)
            f0 = tracked_func(x0)

            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            # Powell's method
            try:
                result = minimize(
                    tracked_func,
                    x0,
                    method='Powell',
                    options={
                        'maxiter': 50,
                        'maxfev': max(10, self.budget - self._eval_count - 5),
                        'xtol': 1e-12,
                        'ftol': 1e-12,
                    }
                )
            except Exception:
                pass

            if result.success or hasattr(result, 'fun'):
                fx = float(result.fun)
                if fx < best_f:
                    best_f = fx
                    best_x = np.clip(result.x, -5.0, 5.0).copy()

            # If we haven't used much budget, do a local (1+1)-ES polish
            remaining = self.budget - self._eval_count
            if remaining > 20 and time.time() < deadline:
                x = best_x.copy()
                fx = best_f
                sigma = 0.5
                for _ in range(min(remaining - 5, 100)):
                    if time.time() > deadline:
                        break
                    if self._eval_count >= self.budget - 5:
                        break
                    y = x + sigma * rng.standard_normal(self.dim)
                    y = np.clip(y, -5.0, 5.0)
                    fy = tracked_func(y)
                    if fy < fx:
                        x = y
                        fx = fy
                        sigma *= 1.2
                    else:
                        sigma *= 0.9

                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

        # Final check
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = tracked_func(best_x)

        return best_x, best_f

    @staticmethod
    def _halton(index, base):
        """Generate the index-th Halton sequence number in the given base."""
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
# EVOLVE-BLOCK-END
