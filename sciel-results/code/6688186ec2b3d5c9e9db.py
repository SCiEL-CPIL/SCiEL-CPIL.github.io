"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Powell-based optimizer with Sobol initialization and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _sobol_points(self, n, dim, rng):
        """Generate quasi-random points in [-5, 5]^dim using a simple Sobol-like sequence."""
        # Use a simple Halton sequence as a lightweight quasi-random alternative
        points = np.zeros((n, dim))
        for i in range(n):
            for d in range(dim):
                # Halton base: use primes
                base = self._primes[d]
                f = 1.0 / base
                idx = i + 1
                while idx > 0:
                    f /= base
                    idx //= base
                points[i, d] = -5 + 10 * f
        return points

    @staticmethod
    def _primes():
        return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    def _line_search(self, func, x, direction, budget_tracker):
        """Golden section line search along direction from x, staying in [-5,5]^dim."""
        # Determine range for alpha
        # Find max alpha such that x + alpha*direction stays in bounds
        max_alpha = 1.0
        for d in range(self.dim):
            if direction[d] > 0:
                max_alpha = min(max_alpha, (5 - x[d]) / direction[d])
            elif direction[d] < 0:
                max_alpha = min(max_alpha, (-5 - x[d]) / (-direction[d]))
        max_alpha = max(max_alpha, 1e-8)

        # Golden section search
        gr = (np.sqrt(5) - 1) / 2
        a, b = 0.0, max_alpha
        c = b - gr * (b - a)
        d = a + gr * (b - a)

        if budget_tracker['calls'] >= budget_tracker['budget']:
            return x, func(x)

        fc = func(np.clip(x + c * direction, -5, 5))
        budget_tracker['calls'] += 1
        fd = func(np.clip(x + d * direction, -5, 5))
        budget_tracker['calls'] += 1

        for _ in range(12):
            if budget_tracker['calls'] >= budget_tracker['budget']:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = func(np.clip(x + c * direction, -5, 5))
                budget_tracker['calls'] += 1
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = func(np.clip(x + d * direction, -5, 5))
                budget_tracker['calls'] += 1

        # Best point
        best_alpha = (a + b) / 2
        best_x = np.clip(x + best_alpha * direction, -5, 5)
        return best_x, min(fc, fd)

    def _powell(self, func, x0, budget_tracker):
        """Powell's method with direction set update."""
        x = x0.copy()
        fx = func(x)
        budget_tracker['calls'] += 1

        # Initial directions: coordinate axes
        directions = [np.eye(self.dim)[i] for i in range(self.dim)]

        while budget_tracker['calls'] < budget_tracker['budget']:
            xold = x.copy()
            fold = fx
            max_delta = 0.0
            max_dir_idx = 0

            for i in range(self.dim):
                if budget_tracker['calls'] >= budget_tracker['budget']:
                    break
                # Line search along direction i
                x_new, f_new = self._line_search(func, x, directions[i], budget_tracker)
                delta = fx - f_new
                if delta > max_delta:
                    max_delta = delta
                    max_dir_idx = i
                x = x_new
                fx = f_new

            if max_delta < 1e-10:
                break

            # Update direction set: replace direction max_dir_idx with (x - xold)
            new_dir = x - xold
            norm = np.linalg.norm(new_dir)
            if norm > 1e-12:
                directions[max_dir_idx] = new_dir / norm

            # Extra step along the new direction
            if budget_tracker['calls'] < budget_tracker['budget']:
                x_new, f_new = self._line_search(func, x, new_dir, budget_tracker)
                if f_new < fx:
                    x = x_new
                    fx = f_new

        return x, fx

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin within 300s limit

        budget_tracker = {'calls': 0, 'budget': self.budget}

        # Initial points: mix of Sobol/Halton and random
        n_initial = 20
        x_candidates = self._sobol_points(n_initial, self.dim, None)

        # Evaluate all initial points
        best_x = None
        best_f = np.inf
        for x in x_candidates:
            if budget_tracker['calls'] >= budget_tracker['budget']:
                break
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            budget_tracker['calls'] += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped

        # Run Powell from the best initial point
        if best_x is not None and budget_tracker['calls'] < budget_tracker['budget']:
            x_final, f_final = self._powell(func, best_x, budget_tracker)
            if f_final < best_f:
                best_x = x_final
                best_f = f_final

        # If we still have budget, try a few more restarts from random points
        while budget_tracker['calls'] < budget_tracker['budget']:
            if time.time() > deadline:
                break
            x_rand = np.random.uniform(-5, 5, self.dim)
            x_clipped = np.clip(x_rand, -5, 5)
            f = func(x_clipped)
            budget_tracker['calls'] += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped
                # Run a short Powell from this new best
                if budget_tracker['calls'] < budget_tracker['budget'] and time.time() < deadline:
                    x_final, f_final = self._powell(func, best_x, budget_tracker)
                    if f_final < best_f:
                        best_x = x_final
                        best_f = f_final

        return best_x, best_f
# EVOLVE-BLOCK-END
