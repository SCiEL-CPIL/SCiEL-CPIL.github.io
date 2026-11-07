"""Hierarchical grid search exploiting Katsuura fractal structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hierarchical grid-based search for Katsuura function on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls = 0

        def evaluate(x):
            nonlocal calls
            if calls >= budget:
                return None
            val = func(x)
            calls += 1
            return val

        # Phase 1: Coarse grid search
        # Use a 3^5 = 243 point grid over [-5, 5]^5
        grid_sizes = [3, 5, 7]
        current_box = np.array([[-5.0] * dim, [5.0] * dim])

        best_x = None
        best_f = np.inf

        for grid_n in grid_sizes:
            # Create grid points
            points = np.array(np.meshgrid(
                *[np.linspace(current_box[0][i], current_box[1][i], grid_n) for i in range(dim)]
            )).T.reshape(-1, dim)

            # Evaluate all grid points
            for x in points:
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                fx = evaluate(x)
                if fx is None:
                    break
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            if calls >= budget or time.time() > deadline:
                break

            # Find the best point and refine around it
            # The best point is at some grid index; find the sub-box
            # We need to figure out which sub-box contains the best point
            # For a grid of size grid_n, the spacing is (hi-lo)/(grid_n-1)
            # The best point's position in the grid tells us which cell it's in
            # Actually, let's just use the best point and create a smaller box around it

            # Calculate the spacing
            spacing = (current_box[1][0] - current_box[0][0]) / (grid_n - 1) if grid_n > 1 else 10.0

            # Create a new box centered on the best point with half the previous spacing
            new_half_span = spacing / 2.0
            new_box = np.array([
                [max(-5.0, best_x[i] - new_half_span) for i in range(dim)],
                [min(5.0, best_x[i] + new_half_span) for i in range(dim)]
            ])

            # Ensure the box is valid
            for i in range(dim):
                if new_box[1][i] <= new_box[0][i]:
                    new_box[0][i] = best_x[i] - 1e-6
                    new_box[1][i] = best_x[i] + 1e-6

            current_box = new_box

        # Phase 2: Fine local search using Nelder-Mead style or simple coordinate descent
        if best_x is not None and calls < budget and time.time() < deadline:
            # Use a simple adaptive local search
            x = best_x.copy()
            fx = best_f
            step_size = (current_box[1][0] - current_box[0][0]) / 10.0
            if step_size < 1e-4:
                step_size = 0.01

            for _ in range(200):
                if calls >= budget or time.time() > deadline:
                    break

                improved = False
                for i in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break

                    # Try positive direction
                    trial = x.copy()
                    trial[i] = np.clip(trial[i] + step_size, -5.0, 5.0)
                    f_trial = evaluate(trial)
                    if f_trial is not None and f_trial < fx:
                        x = trial
                        fx = f_trial
                        improved = True
                        continue

                    # Try negative direction
                    trial = x.copy()
                    trial[i] = np.clip(trial[i] - step_size, -5.0, 5.0)
                    f_trial = evaluate(trial)
                    if f_trial is not None and f_trial < fx:
                        x = trial
                        fx = f_trial
                        improved = True

                if not improved:
                    step_size *= 0.5
                    if step_size < 1e-6:
                        break
                else:
                    step_size *= 1.2  # grow slightly if we're making progress

            best_x = x
            best_f = fx

        # Phase 3: Final polish with very small steps
        if best_x is not None and calls < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            step_size = 1e-4

            for _ in range(100):
                if calls >= budget or time.time() > deadline:
                    break

                improved = False
                for i in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break

                    trial = x.copy()
                    trial[i] = np.clip(trial[i] + step_size, -5.0, 5.0)
                    f_trial = evaluate(trial)
                    if f_trial is not None and f_trial < fx:
                        x = trial
                        fx = f_trial
                        improved = True
                        continue

                    trial = x.copy()
                    trial[i] = np.clip(trial[i] - step_size, -5.0, 5.0)
                    f_trial = evaluate(trial)
                    if f_trial is not None and f_trial < fx:
                        x = trial
                        fx = f_trial
                        improved = True

                if not improved:
                    step_size *= 0.5
                    if step_size < 1e-8:
                        break

            best_x = x
            best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
