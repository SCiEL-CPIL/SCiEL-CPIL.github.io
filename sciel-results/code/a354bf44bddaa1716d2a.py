"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Coordinate-descent hybrid with adaptive step sizes per dimension."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Start at box center
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Adaptive step sizes per dimension
        steps = np.ones(dim) * 1.0

        # Track progress for step adaptation
        calls_used = 1
        no_improve_count = 0

        while calls_used < budget:
            # Alternate between deterministic line searches and random perturbations
            # Determine which dimension to search along (cycling through dimensions)
            # Use a mix: mostly coordinate descent, occasionally random perturbation

            # Randomly decide: 80% coordinate descent, 20% random perturbation
            if rng.random() < 0.8:
                # Coordinate descent: pick a dimension (cycle or random)
                dim_idx = int(rng.integers(0, dim))
                direction = np.zeros(dim)
                direction[dim_idx] = 1.0

                # Line search along this direction
                step_size = steps[dim_idx]
                improved = False
                for _ in range(5):  # up to 5 line search steps
                    trial = x + step_size * direction
                    trial = np.clip(trial, -5, 5)
                    f_trial = func(trial)
                    calls_used += 1
                    if calls_used >= budget:
                        break
                    if f_trial < fx:
                        x = trial
                        fx = f_trial
                        improved = True
                        # Expand step in this direction
                        steps[dim_idx] = min(steps[dim_idx] * 1.5, 10.0)
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                    else:
                        # Try negative direction
                        trial = x - step_size * direction
                        trial = np.clip(trial, -5, 5)
                        f_trial = func(trial)
                        calls_used += 1
                        if calls_used >= budget:
                            break
                        if f_trial < fx:
                            x = trial
                            fx = f_trial
                            improved = True
                            # Shrink step (we overshot)
                            steps[dim_idx] = max(steps[dim_idx] * 0.7, 0.01)
                            if fx < best_fx:
                                best_fx = fx
                                best_x = x.copy()
                        else:
                            # No improvement in either direction, shrink step
                            steps[dim_idx] = max(steps[dim_idx] * 0.7, 0.01)
                            break

                if improved:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Random perturbation to escape local minima
                # Scale perturbation by current step sizes
                perturbation = rng.standard_normal(dim) * steps * 0.5
                trial = x + perturbation
                trial = np.clip(trial, -5, 5)
                f_trial = func(trial)
                calls_used += 1
                if f_trial < fx:
                    x = trial
                    fx = f_trial
                    no_improve_count = 0
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    no_improve_count += 1
                    # Occasionally reset to best point if stuck
                    if no_improve_count > 20:
                        x = best_x.copy()
                        fx = best_fx
                        no_improve_count = 0
                        # Randomize steps to escape
                        steps = np.ones(dim) * rng.uniform(0.5, 2.0)

            # Periodically adapt global step sizes based on progress
            if calls_used % 50 == 0 and calls_used < budget:
                # If we're making progress, keep steps; otherwise adjust
                if no_improve_count > 10:
                    # Increase exploration
                    steps = np.clip(steps * 1.2, 0.01, 10.0)
                elif no_improve_count == 0:
                    # Making good progress, fine-tune
                    steps = np.clip(steps * 0.9, 0.01, 10.0)

            # Ensure we don't exceed budget
            if calls_used >= budget:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
