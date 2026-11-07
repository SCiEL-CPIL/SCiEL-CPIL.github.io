"""Model-and-Optimize with Powell coordinate-wise search for Sphere."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Model-based optimizer with Powell coordinate search for BBOB-style Sphere functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0

        self.calls = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(self.dim)

        N = self.dim
        bounds_low = -5.0
        bounds_high = 5.0

        def wrapped_func(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")

            x = np.clip(x, bounds_low, bounds_high)

            val = func(x)
            self.calls += 1

            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()

            return val

        # PHASE 1: MODEL AND OPTIMIZE
        # Estimate the center c of the Sphere f(x) = ||x-c||^2 + k
        # Using N+1 points: x0=0, xi = D*ei

        D = 4.0
        points = [np.zeros(N)]
        for i in range(N):
            p = np.zeros(N)
            p[i] = D
            points.append(p)

        values = []
        for p in points:
            try:
                v = wrapped_func(p)
                values.append(v)
            except (ValueError, TimeoutError, Exception):
                break

        # If model estimation failed, fallback to random search + Powell
        if len(values) < N + 1:
            rng = np.random.default_rng(42)
            for _ in range(10):
                if self.calls >= self.budget:
                    break
                x = rng.uniform(bounds_low, bounds_high, N)
                try:
                    wrapped_func(x)
                except:
                    break
            if self.calls < self.budget and time.time() < deadline:
                self._run_powell(wrapped_func, self.best_x, N)
            return (self.best_x, self.best_f)

        # Solve for c
        # f(xi) - f(x0) = ||xi||^2 - ||x0||^2 - 2 ci^T (xi - x0)
        # Since x0=0 and xi = D*ei:
        # 2 ci * D = D^2 - (f(xi) - f(0))
        # ci = (D^2 - (f(xi) - f(0))) / (2D)

        c_est = np.zeros(N)
        f0 = values[0]
        for i in range(N):
            fi = values[i + 1]
            c_est[i] = (D**2 - (fi - f0)) / (2.0 * D)

        x_start = np.clip(c_est, bounds_low, bounds_high)

        # Evaluate the estimated center
        if not np.allclose(x_start, self.best_x):
            try:
                wrapped_func(x_start)
            except (ValueError, TimeoutError, Exception):
                pass

        # PHASE 2: POLISH WITH POWELL'S METHOD
        # Powell's method is very efficient for smooth convex functions like Sphere
        # It maintains independent search directions per dimension
        if self.calls < self.budget and time.time() < deadline:
            self._run_powell(wrapped_func, self.best_x, N)

        return (self.best_x, self.best_f)

    def _run_powell(self, func, x0, dim):
        """Powell's method with adaptive step size for local polish."""
        x_best = x0.copy()
        f_best = self.best_f

        # Initial directions: standard basis vectors
        directions = [np.zeros(dim) for _ in range(dim)]
        for i in range(dim):
            directions[i][i] = 1.0

        # Initial step sizes
        steps = [0.5] * dim

        # Parameters
        max_iter = 50
        tol = 1e-10

        for iteration in range(max_iter):
            if self.calls >= self.budget or time.time() > self.deadline - 2:
                break

            x_current = x_best.copy()
            f_current = f_best
            min_delta = 0.0
            min_dir_idx = -1

            for i in range(dim):
                if self.calls >= self.budget or time.time() > self.deadline - 2:
                    break

                # Line search along direction i
                best_x_i = x_current
                best_f_i = f_current
                best_step = steps[i]

                # Try positive and negative directions
                for sign in [1, -1]:
                    # Exponential search to find bracket
                    s = best_step
                    for _ in range(20):
                        if self.calls >= self.budget or time.time() > self.deadline - 2:
                            break

                        x_try = x_current + sign * s * directions[i]
                        x_try = np.clip(x_try, -5.0, 5.0)

                        try:
                            f_try = func(x_try)
                        except (ValueError, TimeoutError, Exception):
                            break

                        if f_try < best_f_i:
                            best_x_i = x_try
                            best_f_i = f_try
                            best_step = s
                            s *= 1.5  # Expand step
                        else:
                            # Try smaller step
                            s *= 0.5
                            if s < 1e-8:
                                break

                # Update
                delta = f_current - best_f_i
                if delta > min_delta:
                    min_delta = delta
                    min_dir_idx = i

                x_current = best_x_i
                f_current = best_f_i
                steps[i] = best_step

            # Check for convergence
            if min_delta < tol:
                break

            # Update best
            if f_current < f_best:
                x_best = x_current.copy()
                f_best = f_current
                self.best_f = f_best
                self.best_x = x_best.copy()

            # Update directions (Powell's direction set update)
            if min_dir_idx >= 0:
                # New direction: x_best - previous x_best
                new_dir = x_best - x_current
                norm = np.linalg.norm(new_dir)
                if norm > 1e-12:
                    new_dir /= norm
                    # Replace the direction with the smallest improvement
                    # Actually in Powell's method, we replace the direction that
                    # gave the smallest improvement with the new direction
                    # But we need to be careful about the ordering
                    # For simplicity, we'll just keep the current directions
                    # and rely on the coordinate-wise search

        return (self.best_x, self.best_f)

# EVOLVE-BLOCK-END
