"""Rewritten optimizer for BBOB f22 (Gallagher 21 peaks) with multi-start local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead with diverse basins for 21-peak landscape."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit

        # Track best value found
        best_x = None
        best_f = np.inf
        eval_count = 0

        def counted_func(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return best_f  # Return best known to avoid extra calls
            x = np.asarray(x, dtype=np.float64)
            fx = func(x)
            eval_count += 1
            return fx

        # Generate diverse starting points for multi-start
        rng = np.random.default_rng(42)

        # Strategy 1: Center point
        starts = [np.zeros(self.dim)]

        # Strategy 2: Random points in the box
        n_random = 15
        starts.extend(rng.uniform(-4, 4, size=(n_random, self.dim)))

        # Strategy 3: Points along axes (helps with rotated landscapes)
        for i in range(self.dim):
            for sign in [1, -1]:
                p = np.zeros(self.dim)
                p[i] = sign * 3.5
                starts.append(p)

        # Strategy 4: Grid-like points at corners
        corners = []
        for bit in range(2 ** self.dim):
            p = np.zeros(self.dim)
            for d in range(self.dim):
                p[d] = 4.0 if (bit >> d) & 1 else -4.0
            corners.append(p)
        starts.extend(corners[:8])  # Limit to 8 corners to save budget

        # Shuffle starts for diversity
        start_arr = np.array(starts)
        perm = rng.permutation(len(start_arr))
        start_arr = start_arr[perm]

        # Run Nelder-Mead from each start, using remaining budget
        # Allocate budget: use small initial budget per start, more if improving
        per_start_budget = max(5, self.budget // (len(start_arr) + 5))

        for i, x0 in enumerate(start_arr):
            if eval_count >= self.budget:
                break
            if time.time() > deadline - 5:
                break

            # Nelder-Mead with adaptive maxiter
            remaining = self.budget - eval_count
            max_iter = min(per_start_budget, remaining)
            if max_iter < 3:
                break

            try:
                result = minimize(
                    counted_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'maxfev': max_iter,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x
            except Exception:
                pass

            # Check time
            if time.time() > deadline - 10:
                break

        # If we have budget left, do a fine-tuning run from the best point
        if eval_count < self.budget and best_x is not None and time.time() < deadline - 15:
            remaining = self.budget - eval_count
            if remaining > 10:
                try:
                    result = minimize(
                        counted_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': remaining,
                            'maxfev': remaining,
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x
                except Exception:
                    pass

        # Final safety: ensure at least one evaluation was made
        if eval_count == 0:
            x = rng.uniform(-5, 5, self.dim)
            best_f = func(x)
            best_x = x
            eval_count = 1

        return best_x, best_f
# EVOLVE-BLOCK-END
