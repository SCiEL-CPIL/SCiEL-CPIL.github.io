"""Rewritten optimizer using Powell's method for the bent-cigar function."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Powell's method with golden-section line search for f12 (Bent cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        calls = [0]

        def safe_call(x):
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            if calls[0] >= self.budget:
                return np.inf
            calls[0] += 1
            try:
                val = float(func(x))
            except Exception:
                return np.inf
            if not np.isfinite(val):
                return np.inf
            return val

        def line_search(x, direction, f_current):
            """Minimize f(x + t * direction) using golden-section search."""
            if calls[0] >= self.budget or time.time() > deadline:
                return x, f_current

            # Determine initial step
            step = 1.0
            # Try to find a descent direction
            x1 = np.clip(x + step * direction, -5, 5)
            f1 = safe_call(x1)

            if f1 >= f_current:
                # Try negative
                x1 = np.clip(x - step * direction, -5, 5)
                f1 = safe_call(x1)
                if f1 >= f_current:
                    return x, f_current
                direction = -direction

            # Expand to bracket
            x2 = x
            f2 = f_current
            x3 = x1
            f3 = f1
            step *= 2

            while f3 < f2 and calls[0] < self.budget and time.time() < deadline and step < 50.0:
                x2 = x3
                f2 = f3
                x3 = np.clip(x2 + step * direction, -5, 5)
                f3 = safe_call(x3)
                step *= 2

            # Now bracket: f2 is best among tested
            if f2 < f3:
                a = x2 - 0.5 * step * direction
                b = x3
            else:
                a = x2
                b = x3

            a = np.clip(a, -5, 5)
            b = np.clip(b, -5, 5)

            # Golden section
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            for _ in range(15):
                if calls[0] >= self.budget or time.time() > deadline:
                    break
                c = a + gr * (b - a)
                d = b - gr * (b - a)
                fc = safe_call(c)
                fd = safe_call(d)
                if fc < fd:
                    b = d
                else:
                    a = c
                if np.linalg.norm(b - a) < 1e-12:
                    break

            x_mid = 0.5 * (a + b)
            f_mid = safe_call(x_mid)
            return x_mid, f_mid

        def powell(x_init, f_init):
            """Powell's conjugate direction method."""
            x = x_init.copy()
            f = f_init
            directions = np.eye(self.dim)

            while calls[0] < self.budget and time.time() < deadline:
                x_start = x.copy()
                f_start = f
                min_f = np.inf
                min_i = 0

                for i in range(self.dim):
                    if calls[0] >= self.budget or time.time() > deadline:
                        break
                    x, f = line_search(x, directions[i], f)
                    if f < min_f:
                        min_f = f
                        min_i = i

                # Check convergence
                if abs(f_start - f) < 1e-14:
                    break

                # Update direction
                d_new = x - x_start
                if np.linalg.norm(d_new) > 1e-14:
                    directions[min_i] = d_new

                    # Extra line search along new direction
                    if calls[0] < self.budget and time.time() < deadline:
                        x, f = line_search(x, directions[min_i], f)

            return x, f

        # Multi-start strategy
        rng = np.random.default_rng(12345)

        # Start from center and a few random points
        starts = [
            np.zeros(self.dim),
            rng.uniform(-4, 4, self.dim),
            rng.uniform(-4, 4, self.dim),
            rng.uniform(-4, 4, self.dim),
        ]

        best_x = None
        best_f = np.inf

        # Allocate budget: use ~200 calls per start initially
        budget_per_start = max(50, self.budget // len(starts))

        for i, x0 in enumerate(starts):
            if calls[0] >= self.budget or time.time() > deadline:
                break
            remaining = self.budget - calls[0]
            alloc = min(budget_per_start, remaining)
            if alloc < 10:
                break

            x_res, f_res = powell(x0, safe_call(x0))
            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()

            # If we improved significantly, do a second Powell pass from there
            if calls[0] < self.budget and time.time() < deadline:
                x_res2, f_res2 = powell(x_res, f_res)
                if f_res2 < best_f:
                    best_f = f_res2
                    best_x = x_res2.copy()

        return best_x if best_x is not None else np.zeros(self.dim), best_f

# EVOLVE-BLOCK-END
