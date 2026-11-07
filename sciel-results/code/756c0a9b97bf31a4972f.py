"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Coordinate-descent hybrid with adaptive step sizes per dimension."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize at box center
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Adaptive step sizes per dimension
        steps = np.ones(dim) * 1.0

        # Track best value after each call for potential early stopping logic
        # But we must make exactly budget calls (or up to budget)

        calls_made = 1
        call_history = []
        call_history.append(best_fx)

        def record(val):
            nonlocal best_x, best_fx
            if val < best_fx:
                best_fx = val
                best_x = x.copy()
            call_history.append(best_fx)

        # Phase 1: Coordinate descent with adaptive steps
        # Alternate between deterministic line searches and random perturbations
        dim_order = list(range(dim))
        rng.shuffle(dim_order)

        # We'll do a mix: mostly coordinate descent, occasionally random jump
        # Each "iteration" can consist of trying to improve along one axis
        # or doing a small random perturbation

        while calls_made < budget:
            # Decide: coordinate descent or random perturbation?
            # 80% coordinate descent, 20% random perturbation
            if rng.random() < 0.8:
                # Pick a random axis to optimize
                d = dim_order[rng.integers(dim)]
                step = steps[d]

                # Try positive and negative directions
                improved = False

                # Positive step
                x_trial = x.copy()
                x_trial[d] = np.clip(x_trial[d] + step, -5, 5)
                fx_trial = func(x_trial)
                calls_made += 1

                if fx_trial < fx:
                    x = x_trial
                    fx = fx_trial
                    steps[d] *= 1.2  # Increase step if improving
                    improved = True
                else:
                    # Try negative step
                    x_trial = x.copy()
                    x_trial[d] = np.clip(x_trial[d] - step, -5, 5)
                    fx_trial = func(x_trial)
                    calls_made += 1

                    if fx_trial < fx:
                        x = x_trial
                        fx = fx_trial
                        steps[d] *= 1.2
                        improved = True
                    else:
                        # No improvement, reduce step
                        steps[d] *= 0.7

                record(fx)

                # If we improved, maybe do a second try in the same direction
                if improved and calls_made < budget and rng.random() < 0.5:
                    x_trial = x.copy()
                    if fx_trial < fx:
                        # We moved in negative direction
                        x_trial[d] = np.clip(x_trial[d] - step, -5, 5)
                    else:
                        x_trial[d] = np.clip(x_trial[d] + step, -5, 5)
                    fx_trial = func(x_trial)
                    calls_made += 1

                    if fx_trial < fx:
                        x = x_trial
                        fx = fx_trial
                        steps[d] *= 1.2
                    else:
                        steps[d] *= 0.8

                    record(fx)

            else:
                # Random perturbation to escape local minima
                # Small random perturbation in all dimensions
                perturbation = rng.standard_normal(dim) * 0.5
                x_trial = np.clip(x + perturbation, -5, 5)
                fx_trial = func(x_trial)
                calls_made += 1

                if fx_trial < fx:
                    x = x_trial
                    fx = fx_trial
                    # Reset steps to moderate values
                    steps = np.ones(dim) * 1.0
                else:
                    # Keep current x, maybe slightly reduce steps
                    steps *= 0.95

                record(fx)

        # Ensure we have exactly budget calls or fewer
        # The loop condition handles this

        return best_x, best_fx
# EVOLVE-BLOCK-END
