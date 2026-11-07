"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Powell's method for the Bent Cigar (f12)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s
        dim = self.dim
        budget = self.budget

        # Track global best
        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def timed_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return best_f
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Powell's method on a single trajectory
        def powell(x0, max_calls):
            nonlocal call_count, best_x, best_f
            n = len(x0)
            x = x0.copy()
            fx = timed_func(x)
            used = call_count
            if used >= max_calls:
                return x, fx

            # Initial directions: coordinate axes
            directions = [np.zeros(n) for _ in range(n)]
            for i in range(n):
                directions[i][i] = 1.0

            # Track old f for each direction
            old_f = [None] * n

            # Main Powell loop
            while call_count < max_calls and (time.time() < deadline):
                improved = False
                for i in range(n):
                    if call_count >= max_calls or time.time() >= deadline:
                        break
                    # Line search along direction i
                    d = directions[i]
                    # Find step size via golden section or simple bisection
                    # Start with a small step and expand
                    step = 0.5
                    x_try = x + step * d
                    x_try = np.clip(x_try, -5, 5)
                    f_try = timed_func(x_try)

                    if f_try < fx:
                        # Move in this direction
                        # Try to extend
                        for _ in range(10):
                            if call_count >= max_calls or time.time() >= deadline:
                                break
                            x_try2 = x + 2.0 * step * d
                            x_try2 = np.clip(x_try2, -5, 5)
                            f_try2 = timed_func(x_try2)
                            if f_try2 < f_try:
                                x = x_try2
                                fx = f_try2
                                step *= 2.0
                            else:
                                break
                        x = x + step * d
                        x = np.clip(x, -5, 5)
                        fx = timed_func(x)
                        improved = True
                    else:
                        # Try negative direction
                        x_try = x - step * d
                        x_try = np.clip(x_try, -5, 5)
                        f_try = timed_func(x_try)
                        if f_try < fx:
                            for _ in range(10):
                                if call_count >= max_calls or time.time() >= deadline:
                                    break
                                x_try2 = x - 2.0 * step * d
                                x_try2 = np.clip(x_try2, -5, 5)
                                f_try2 = timed_func(x_try2)
                                if f_try2 < f_try:
                                    x = x_try2
                                    fx = f_try2
                                    step *= 2.0
                                else:
                                    break
                            x = x - step * d
                            x = np.clip(x, -5, 5)
                            fx = timed_func(x)
                            improved = True

                if not improved:
                    break

                # Update directions: replace the worst direction with the new direction
                # Simple approach: rotate directions
                # Replace the last direction with the direction from old_x to new_x
                # For simplicity, just keep cycling through coordinate axes with adaptive steps
                # Actually, let's do a proper Powell direction update
                # The direction of most improvement replaces the direction with least improvement

            return x, fx

        # Better Powell implementation with proper direction management
        def powell_v2(x0, max_calls):
            nonlocal call_count, best_x, best_f
            n = len(x0)
            x = x0.copy()
            fx = timed_func(x)
            if call_count >= max_calls:
                return x, fx

            # Directions
            dirs = np.eye(n)
            # Step sizes per direction
            steps = np.ones(n) * 0.5

            for _ in range(50):  # Max Powell iterations
                if call_count >= max_calls or time.time() >= deadline:
                    break

                old_fx = fx
                x_old = x.copy()
                min_idx = 0
                min_delta = np.inf

                for i in range(n):
                    if call_count >= max_calls or time.time() >= deadline:
                        break
                    d = dirs[i]
                    # Adaptive step size
                    step = steps[i]
                    # Try positive
                    x_try = np.clip(x + step * d, -5, 5)
                    f_try = timed_func(x_try)
                    if f_try < fx:
                        # Extend in positive direction
                        for _ in range(8):
                            if call_count >= max_calls or time.time() >= deadline:
                                break
                            x_try2 = np.clip(x + 2.0 * step * d, -5, 5)
                            f_try2 = timed_func(x_try2)
                            if f_try2 < f_try:
                                x = x_try2
                                fx = f_try2
                                step *= 2.0
                            else:
                                break
                        x = np.clip(x + step * d, -5, 5)
                        fx = timed_func(x)
                        delta = old_fx - fx
                        if delta > 0 and delta < min_delta:
                            min_delta = delta
                            min_idx = i
                    else:
                        # Try negative
                        x_try = np.clip(x - step * d, -5, 5)
                        f_try = timed_func(x_try)
                        if f_try < fx:
                            for _ in range(8):
                                if call_count >= max_calls or time.time() >= deadline:
                                    break
                                x_try2 = np.clip(x - 2.0 * step * d, -5, 5)
                                f_try2 = timed_func(x_try2)
                                if f_try2 < f_try:
                                    x = x_try2
                                    fx = f_try2
                                    step *= 2.0
                                else:
                                    break
                            x = np.clip(x - step * d, -5, 5)
                            fx = timed_func(x)
                            delta = old_fx - fx
                            if delta > 0 and delta < min_delta:
                                min_delta = delta
                                min_idx = i

                if fx >= old_fx:
                    break

                # Update the direction with the least improvement
                new_dir = x - x_old
                norm = np.linalg.norm(new_dir)
                if norm > 1e-10:
                    new_dir = new_dir / norm
                    dirs[min_idx] = new_dir
                    steps[min_idx] = min(5.0, steps[min_idx] * 1.2)
                else:
                    break

            return x, fx

        # Multi-start: distribute budget across several trajectories
        # Use a few random starting points
        rng = np.random.default_rng(42)

        # Number of restarts based on budget
        n_restarts = 3
        calls_per_restart = budget // n_restarts

        for r in range(n_restarts):
            if call_count >= budget or time.time() >= deadline:
                break

            # Random starting point
            x0 = rng.uniform(-5, 5, dim)
            if r == 0:
                # First start: center
                x0 = np.zeros(dim)

            max_calls_this = min(calls_per_restart, budget - call_count)
            if max_calls_this < 10:
                break

            _, _ = powell_v2(x0, call_count + max_calls_this)

        # If we have remaining budget, do a final refinement from best
        if call_count < budget and time.time() < deadline:
            remaining = budget - call_count
            if remaining >= 5:
                powell_v2(best_x.copy(), budget)

        return best_x, best_f
# EVOLVE-BLOCK-END
