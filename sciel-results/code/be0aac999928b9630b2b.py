# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """BFGS with global warm-up for smooth convex functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        rng = np.random.default_rng(42)

        call_count = 0

        def count_func(x):
            nonlocal call_count
            call_count += 1
            return func(np.clip(x, -5, 5))

        # --- Phase 1: Global search with 40 random points ---
        n_global = 40
        x = np.zeros(self.dim)
        best_fx = np.inf
        for _ in range(n_global):
            if call_count >= self.budget or time.time() > deadline:
                break
            x_cand = rng.uniform(-5, 5, self.dim)
            fx_cand = count_func(x_cand)
            if fx_cand < best_fx:
                best_fx = fx_cand
                x = x_cand.copy()

        # --- Phase 2: BFGS local optimization ---
        def fd_grad(x, h=1e-5):
            g = np.zeros(self.dim)
            for i in range(self.dim):
                xp = x.copy(); xp[i] += h
                xm = x.copy(); xm[i] -= h
                g[i] = (count_func(xp) - count_func(xm)) / (2 * h)
            return g

        def line_search(x, d, fx):
            alpha = 1.0
            g = fd_grad(x, 1e-5)
            slope = np.dot(g, d)
            if slope >= 0:
                return x, fx, 0.0
            c1 = 1e-4
            for _ in range(20):
                if call_count >= self.budget or time.time() > deadline:
                    return x, fx, alpha
                x_new = x + alpha * d
                fx_new = count_func(x_new)
                if fx_new <= fx + c1 * alpha * slope:
                    return x_new, fx_new, alpha
                alpha *= 0.5
            return x, fx, alpha

        # Initialize
        g = fd_grad(x, 1e-5)
        B = np.eye(self.dim)

        for it in range(40):
            if call_count >= self.budget or time.time() > deadline:
                break

            # Search direction
            try:
                d = np.linalg.solve(B, -g)
            except np.linalg.LinAlgError:
                d = -g
            if np.dot(d, g) > 0:
                d = -g

            # Line search
            x_new, fx_new, alpha = line_search(x, d, best_fx)
            if alpha < 1e-10 or np.allclose(x_new, x):
                break

            # BFGS update
            g_new = fd_grad(x_new, 1e-5)
            s = x_new - x
            y = g_new - g
            sy = np.dot(s, y)

            if sy > 1e-10:
                rho = 1.0 / sy
                B = (np.eye(self.dim) - rho * np.outer(s, y)) @ B @ (np.eye(self.dim) - rho * np.outer(y, s)) + rho * np.outer(s, s)

            x = x_new
            best_fx = fx_new
            g = g_new

        return x, best_fx
# EVOLVE-BLOCK-END
