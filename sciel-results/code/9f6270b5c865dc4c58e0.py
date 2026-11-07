import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def time_ok():
            return time.time() < deadline

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # 1. Initial coarse scan - use stratified sampling for better coverage
        # We use a mix of the origin and random points to quickly identify promising regions.
        init_points = [np.zeros(dim)]
        for _ in range(8):
            init_points.append(np.random.uniform(-5, 5, dim))

        # Evaluate initial points
        for x in init_points:
            if not time_ok() or calls >= budget: break
            f = func(x)
            calls += 1
            record(x, f)

        if best_x is None:
            return None, np.inf

        # 2. Main Search: (1+1)-ES with Adaptive Step Size
        # We adopt the (1+1)-ES structure from the current program but refine the adaptation logic
        # to be more robust. The current program uses a simple EMA of success.
        # We will use a similar approach but ensure the step size doesn't collapse too fast
        # and includes a mechanism to "kick" the search if it gets stuck (similar to the restart logic).

        x = best_x.copy()
        f_current = best_f
        sigma = 0.5  # Start with a moderate step size

        # Parameters for step-size adaptation
        c_sigma = 0.5  # Learning rate for sigma (higher = faster adaptation)
        target_success = 0.25  # Target success rate (25% of moves should be improvements)
        success_ema = 0.0  # Exponential moving average of success
        stall_count = 0
        max_stall = 20  # Restart if no progress for this many steps

        # Phase 1: Main ES search
        while calls < budget and time_ok():
            # Generate a random direction
            z = np.random.randn(dim)
            norm_z = np.linalg.norm(z)
            if norm_z > 1e-10:
                z = z / norm_z
            else:
                z = np.zeros(dim)
                z[0] = 1.0

            # Propose new point
            x_new = x + sigma * z
            x_new = np.clip(x_new, -5, 5)

            # Evaluate
            if not time_ok() or calls >= budget: break
            f_new = func(x_new)
            calls += 1
            record(x_new, f_new)

            # Update best
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()

            # Adapt sigma based on success
            success = 1.0 if (f_new < f_current) else 0.0
            success_ema = (1 - c_sigma) * success_ema + c_sigma * success

            # Adjust sigma smoothly
            if success_ema > target_success:
                sigma *= 1.1
            elif success_ema < target_success:
                sigma *= 0.9

            # Clamp sigma
            sigma = max(1e-6, min(10.0, sigma))

            # Accept the new point if it's better (1+1 rule)
            if f_new < f_current:
                x = x_new
                f_current = f_new
                stall_count = 0
            else:
                stall_count += 1

            # Restart if stalled
            if stall_count >= max_stall:
                # Try a larger step from best
                sigma = max(sigma * 2.0, 1.0)
                stall_count = 0
                # Occasionally try a random restart near best
                if np.random.rand() < 0.3:
                    x = best_x + np.random.randn(dim) * sigma
                    x = np.clip(x, -5, 5)
                    f_current = func(x)
                    calls += 1
                    record(x, f_current)
                    if f_current < best_f:
                        best_f = f_current
                        best_x = x.copy()

            # If sigma is very small, we're likely converged - move to polish
            if sigma < 1e-4:
                break

        # 3. Final polish phase: use coordinate descent / local refinement
        # This is the refinement machinery from the current program, which is effective for
        # squeezing out the last bit of precision in low-dimensional spaces.
        if calls < budget and time_ok():
            # Try a simple local refinement from best_x
            x_polish = best_x.copy()
            f_polish = best_f
            step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]

            for step in step_sizes:
                if not time_ok() or calls >= budget:
                    break
                improved = True
                while improved and time_ok() and calls < budget:
                    improved = False
                    for d in range(dim):
                        # Try positive step
                        x_test = x_polish.copy()
                        x_test[d] += step
                        x_test = np.clip(x_test, -5, 5)
                        if calls >= budget or not time_ok():
                            break
                        f_test = func(x_test)
                        calls += 1
                        record(x_test, f_test)
                        if f_test < f_polish:
                            x_polish = x_test
                            f_polish = f_test
                            improved = True
                        # Try negative step
                        x_test = x_polish.copy()
                        x_test[d] -= step
                        x_test = np.clip(x_test, -5, 5)
                        if calls >= budget or not time_ok():
                            break
                        f_test = func(x_test)
                        calls += 1
                        record(x_test, f_test)
                        if f_test < f_polish:
                            x_polish = x_test
                            f_polish = f_test
                            improved = True
                # Update best from polish
                if f_polish < best_f:
                    best_f = f_polish
                    best_x = x_polish.copy()

        return best_x, best_f
