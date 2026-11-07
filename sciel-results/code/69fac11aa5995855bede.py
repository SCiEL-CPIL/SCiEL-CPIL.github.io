"""Optimized search for BBOB f13 Sharp Ridge."""
import numpy as np
import time


class Optimizer:
    """Coordinate Descent with adaptive step sizes for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0

        rng = np.random.default_rng(42)
        calls = [0]

        def safe_call(x):
            if calls[0] >= budget:
                return float('inf')
            x = np.clip(x, -5.0, 5.0)
            try:
                val = float(func(x))
                calls[0] += 1
                return val
            except:
                calls[0] += 1
                return float('inf')

        best_x = np.zeros(dim)
        best_f = float('inf')

        # Initial point
        x = rng.uniform(-3, 3, dim)
        f = safe_call(x)
        best_x = x.copy()
        best_f = f

        # Adaptive step sizes per dimension
        steps = np.ones(dim) * 1.0
        min_step = 1e-6
        max_step = 8.0

        # Track recent success per dimension
        success_count = np.zeros(dim, dtype=int)
        fail_count = np.zeros(dim, dtype=int)

        # Main loop
        while calls[0] < budget and time.time() < deadline:
            if time.time() > deadline - 3:
                break

            # Coordinate descent: sweep through all dimensions
            for d in range(dim):
                if calls[0] >= budget or time.time() > deadline - 3:
                    break

                # Try positive and negative steps
                improved = False
                for direction in [1, -1]:
                    x_trial = x.copy()
                    x_trial[d] += direction * steps[d]
                    x_trial = np.clip(x_trial, -5.0, 5.0)

                    f_trial = safe_call(x_trial)
                    if calls[0] >= budget:
                        break

                    if f_trial < f:
                        # Accept the move
                        x[d] = x_trial[d]
                        f = f_trial
                        improved = True
                        success_count[d] += 1
                        # Increase step size slightly
                        steps[d] = min(max_step, steps[d] * 1.2)

                        # Update best
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()

                        # Try again in the same direction (line search continuation)
                        for _ in range(3):
                            if calls[0] >= budget or time.time() > deadline - 3:
                                break
                            x_trial2 = x.copy()
                            x_trial2[d] += direction * steps[d]
                            x_trial2 = np.clip(x_trial2, -5.0, 5.0)
                            f_trial2 = safe_call(x_trial2)
                            if f_trial2 < f:
                                x[d] = x_trial2[d]
                                f = f_trial2
                                success_count[d] += 1
                                if f < best_f:
                                    best_f = f
                                    best_x = x.copy()
                            else:
                                break
                    else:
                        # Reject, decrease step
                        fail_count[d] += 1
                        steps[d] = max(min_step, steps[d] * 0.7)

                # If no improvement in this dimension, reduce step further
                if not improved:
                    steps[d] = max(min_step, steps[d] * 0.5)

            # Periodic random perturbation to escape local minima
            if calls[0] < budget and time.time() < deadline - 3:
                # Every ~50 calls, add a small random perturbation
                if calls[0] % 50 < 10:
                    perturbation = rng.normal(0, 0.5, dim)
                    x_pert = x + perturbation
                    x_pert = np.clip(x_pert, -5.0, 5.0)
                    f_pert = safe_call(x_pert)
                    if f_pert < f:
                        x = x_pert.copy()
                        f = f_pert
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                        # Reset steps to moderate values
                        steps = np.ones(dim) * 1.0

            # If all steps are very small, try a larger random jump
            if np.all(steps < 1e-4) and calls[0] < budget and time.time() < deadline - 3:
                x_jump = x + rng.normal(0, 2.0, dim)
                x_jump = np.clip(x_jump, -5.0, 5.0)
                f_jump = safe_call(x_jump)
                if f_jump < f:
                    x = x_jump.copy()
                    f = f_jump
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                # Reset steps
                steps = np.ones(dim) * 1.0

        return best_x, best_f
