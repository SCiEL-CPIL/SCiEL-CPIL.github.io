"""Pattern search optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Direct search pattern search with adaptive step sizes and sparse random restarts.
    Maintains orthogonal and diagonal search directions with binary search on step size.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _pattern_search(self, func, x0, budget_remaining, deadline, rng):
        """
        Pattern search with adaptive step sizes.
        Uses orthogonal and diagonal directions with binary search on step size.
        """
        dim = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        calls = 1
        best_x = x.copy()
        best_f = fx

        # Initialize step sizes
        step_sizes = np.full(dim, 1.0, dtype=float)

        # Directions: orthogonal axes and diagonal directions
        # For dim=5, we have 5 orthogonal + several diagonal directions
        directions = []
        for d in range(dim):
            e = np.zeros(dim)
            e[d] = 1.0
            directions.append(e)
        # Diagonal directions (all ones, and some random diagonals)
        directions.append(np.ones(dim) / np.sqrt(dim))
        directions.append(-np.ones(dim) / np.sqrt(dim))
        # Add a few random diagonal directions
        for _ in range(3):
            v = rng.standard_normal(dim)
            v = v / np.linalg.norm(v)
            directions.append(v)
            directions.append(-v)

        # Binary search parameters
        max_expansion = 2.0
        min_step = 1e-8
        max_step = 5.0

        while calls < budget_remaining:
            if time.time() > deadline - 1.0:
                break

            progress_made = False

            for d_idx, direction in enumerate(directions):
                if calls >= budget_remaining:
                    break
                if time.time() > deadline - 1.0:
                    break

                # Binary search on step size for this direction
                lo = min_step
                hi = max_step
                best_step = 0.0
                best_fx = fx

                for _ in range(10):  # 10 iterations of binary search
                    if calls >= budget_remaining:
                        break
                    if time.time() > deadline - 1.0:
                        break

                    mid = (lo + hi) / 2.0
                    # Try positive direction
                    x_try = x + mid * direction
                    x_try = np.clip(x_try, -5, 5)
                    f_try = func(x_try)
                    calls += 1

                    if f_try < best_fx:
                        best_fx = f_try
                        best_step = mid
                        lo = mid
                    else:
                        hi = mid

                if best_step > 0 and best_fx < fx:
                    x = x + best_step * direction
                    x = np.clip(x, -5, 5)
                    fx = best_fx
                    progress_made = True

                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

                    # Increase step size for this direction
                    if d_idx < dim:
                        step_sizes[d_idx] = min(step_sizes[d_idx] * 1.5, max_step)
                else:
                    # Decrease step size for this direction
                    if d_idx < dim:
                        step_sizes[d_idx] = max(step_sizes[d_idx] * 0.7, min_step)

            if not progress_made:
                # No progress in this full cycle, break
                break

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        # Create diverse starting points
        starts = [np.zeros(dim)]
        for d in range(dim):
            x0 = np.zeros(dim)
            x0[d] = 2.0
            starts.append(x0)
            x0[d] = -2.0
            starts.append(x0)
        for _ in range(4):
            starts.append(rng.uniform(-3, 3, size=dim))

        num_starts = len(starts)
        base_alloc = max(budget // num_starts, 10)

        for i, x0 in enumerate(starts):
            if total_calls >= budget:
                break
            remaining = budget - total_calls
            if remaining < 5:
                break
            if time.time() > deadline - 10.0:
                break

            if i == 0:
                alloc = min(base_alloc * 3, remaining)
            else:
                alloc = min(base_alloc, remaining)

            if alloc < 2:
                break

            bx, bf, used = self._pattern_search(func, x0, alloc, deadline, rng)
            total_calls += used

            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        # Final polish
        if total_calls < budget and time.time() < deadline - 5.0:
            remaining = budget - total_calls
            if remaining >= 10:
                bx, bf, used = self._pattern_search(func, best_x, remaining, deadline, rng)
                total_calls += used
                if bf < best_f:
                    best_f = bf
                    best_x = bx.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
