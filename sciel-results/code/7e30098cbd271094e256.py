"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Coordinate Descent optimizer with adaptive step-size and periodic random restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize at a random point in the box
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        calls = 1

        # Initial step size - start larger for exploration
        step = 2.0
        best_x = x.copy()
        best_fx = fx

        # Coordinate descent with cyclic axis order
        # Periodic random perturbations to escape local minima
        perturb_every = 20  # every 20 function calls, add a random perturbation
        call_count = 0
        no_improve_count = 0
        stagnation_counter = 0

        while call_count < budget - 1:
            # Try each coordinate in turn
            for i in range(dim):
                if call_count >= budget - 1:
                    break

                # Try positive and negative steps along axis i
                improved = False
                for sign in [1.0, -1.0]:
                    if call_count >= budget - 1:
                        break

                    trial = x.copy()
                    trial[i] = np.clip(trial[i] + sign * step, -5.0, 5.0)

                    # If the step didn't change the point (already at boundary), skip
                    if np.array_equal(trial, x):
                        continue

                    f_trial = func(trial)
                    call_count += 1

                    if f_trial < fx:
                        x = trial
                        fx = f_trial
                        if fx < best_fx:
                            best_x = x.copy()
                            best_fx = fx
                        no_improve_count = 0
                        stagnation_counter = 0
                        # Increase step on success
                        step *= 1.05
                        improved = True
                        break  # stop trying the other sign for this axis

                # If no improvement on this axis, shrink step
                if not improved:
                    step *= 0.9
                    no_improve_count += 1

            # Track stagnation (no improvement in best_fx)
            if no_improve_count >= 5:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            # Periodic random perturbation to escape local minima
            # Also trigger if we've had many consecutive non-improvements
            if call_count > 0 and (call_count % perturb_every == 0 or no_improve_count >= 8 or stagnation_counter >= 3):
                # Add a random perturbation
                perturbation = rng.normal(0, step * 2.0, dim)
                trial = np.clip(x + perturbation, -5.0, 5.0)
                if call_count < budget - 1:
                    f_trial = func(trial)
                    call_count += 1
                    if f_trial < fx:
                        x = trial
                        fx = f_trial
                        if fx < best_fx:
                            best_x = x.copy()
                            best_fx = fx
                    no_improve_count = 0
                    stagnation_counter = 0

            # Global step size decay per full cycle
            step *= 0.98
            # Ensure step doesn't get too small
            step = max(step, 0.01)

        return x, fx
# EVOLVE-BLOCK-END
