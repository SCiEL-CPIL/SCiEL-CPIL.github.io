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

        # Initial point - wider range to ensure basin capture
        x = rng.uniform(-4.0, 4.0, dim)
        f = safe_call(x)
        best_x = x.copy()
        best_f = f

        # Adaptive step sizes per dimension - start larger to find basin
        steps = np.ones(dim) * 2.0
        min_step = 1e-8
        max_step = 10.0

        # Track recent best for stagnation detection
        last_improvement_call = calls[0]
        last_improvement_val = f

        # Main loop
        while calls[0] < budget and time.time() < deadline:
            if time.time() > deadline - 5:
                break

            # Check for stagnation
            if calls[0] - last_improvement_call > 20:
                # Stagnant: try a random jump
                jump_sigma = max(0.5, np.mean(steps))
                x_jump = x + rng.normal(0, jump_sigma, dim)
                x_jump = np.clip(x_jump, -5.0, 5.0)
                f_jump = safe_call(x_jump)
                if f_jump < best_f:
                    x = x_jump.copy()
                    f = f_jump
                    best_f = f_jump
                    best_x = x.copy()
                    last_improvement_call = calls[0]
                    last_improvement_val = f_jump
                    # Reset steps to moderate
                    steps = np.ones(dim) * 1.0
                else:
                    # If jump didn't help, try smaller perturbations
                    for _ in range(3):
                        if calls[0] >= budget: break
                        x_pert = x + rng.normal(0, 0.2, dim)
                        x_pert = np.clip(x_pert, -5.0, 5.0)
                        f_pert = safe_call(x_pert)
                        if f_pert < f:
                            x = x_pert.copy()
                            f = f_pert
                            if f < best_f:
                                best_f = f
                                best_x = x.copy()
                                last_improvement_call = calls[0]
                                last_improvement_val = f
                            break
                    # Reset steps if stuck
                    steps = np.ones(dim) * 0.5

            # Coordinate descent: sweep through all dimensions
            for d in range(dim):
                if calls[0] >= budget or time.time() > deadline - 5:
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
                        # Increase step size slightly (conservative for sharp ridges)
                        steps[d] = min(max_step, steps[d] * 1.1)

                        # Update best
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                            last_improvement_call = calls[0]
                            last_improvement_val = f

                        # Try again in the same direction (line search continuation)
                        for _ in range(3):
                            if calls[0] >= budget or time.time() > deadline - 5:
                                break
                            x_trial2 = x.copy()
                            x_trial2[d] += direction * steps[d]
                            x_trial2 = np.clip(x_trial2, -5.0, 5.0)
                            f_trial2 = safe_call(x_trial2)
                            if f_trial2 < f:
                                x[d] = x_trial2[d]
                                f = f_trial2
                                if f < best_f:
                                    best_f = f
                                    best_x = x.copy()
                                    last_improvement_call = calls[0]
                                    last_improvement_val = f
                            else:
                                break
                    else:
                        # Reject, decrease step
                        steps[d] = max(min_step, steps[d] * 0.7)

                # If no improvement in this dimension, reduce step further
                if not improved:
                    steps[d] = max(min_step, steps[d] * 0.5)

        return best_x, best_f
