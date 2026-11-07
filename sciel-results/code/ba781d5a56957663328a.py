"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hooke-Jeeves pattern search for bent-cigar (f12) optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)
        call_count = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            # The task states func clips, but we ensure it's a valid vector
            x = np.clip(x, -5.0, 5.0)
            f_val = func(x)
            call_count += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # 1. Initialization: Sample a few points to find a good basin
        n_init = 10
        for _ in range(n_init):
            if call_count >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        if best_x is None:
            x0 = np.zeros(dim)
            evaluate(x0)
            best_x = x0.copy()

        # 2. Hooke-Jeeves pattern search with coordinate-wise line searches
        # This is well-suited for bent cigar's anisotropic valley structure
        current_x = best_x.copy()
        current_f = best_f
        steps = np.ones(dim)
        min_step = 1e-6
        max_step = 5.0
        step_multiplier = 2.0
        step_shrink = 0.5

        # Track which coordinates have been successfully stepped
        coord_success = np.zeros(dim, dtype=bool)

        while call_count < budget and time.time() < deadline:
            improved = False

            # Try each coordinate in sequence
            for i in range(dim):
                if call_count >= budget or time.time() >= deadline:
                    break

                # Try positive direction
                trial_x = current_x.copy()
                trial_x[i] += steps[i]
                trial_x = np.clip(trial_x, -5, 5)
                f_trial = evaluate(trial_x)

                if f_trial < current_f:
                    current_x = trial_x.copy()
                    current_f = f_trial
                    improved = True
                    coord_success[i] = True
                    # Continue in same direction (line search)
                    while call_count < budget and time.time() < deadline:
                        next_x = current_x.copy()
                        next_x[i] += steps[i]
                        next_x = np.clip(next_x, -5, 5)
                        f_next = evaluate(next_x)
                        if f_next < current_f:
                            current_x = next_x.copy()
                            current_f = f_next
                        else:
                            break
                    # Increase step size for this coordinate
                    steps[i] = min(steps[i] * step_multiplier, max_step)
                else:
                    # Try negative direction
                    trial_x = current_x.copy()
                    trial_x[i] -= steps[i]
                    trial_x = np.clip(trial_x, -5, 5)
                    f_trial = evaluate(trial_x)

                    if f_trial < current_f:
                        current_x = trial_x.copy()
                        current_f = f_trial
                        improved = True
                        coord_success[i] = True
                        # Continue in same direction (line search)
                        while call_count < budget and time.time() < deadline:
                            next_x = current_x.copy()
                            next_x[i] -= steps[i]
                            next_x = np.clip(next_x, -5, 5)
                            f_next = evaluate(next_x)
                            if f_next < current_f:
                                current_x = next_x.copy()
                                current_f = f_next
                            else:
                                break
                        # Increase step size for this coordinate
                        steps[i] = min(steps[i] * step_multiplier, max_step)
                    else:
                        coord_success[i] = False

            # If no coordinate improved, shrink all step sizes
            if not improved:
                steps *= step_shrink
                # If all steps are too small, do a random restart
                if np.all(steps < min_step):
                    # Random restart from best point with some noise
                    noise = rng.standard_normal(dim) * 0.5
                    new_x = best_x + noise
                    new_x = np.clip(new_x, -5, 5)
                    f_new = evaluate(new_x)
                    if f_new < current_f:
                        current_x = new_x.copy()
                        current_f = f_new
                    # Reset steps
                    steps = np.ones(dim) * 0.5

            # Periodic global exploration
            if call_count % 100 == 0 and call_count > 0:
                if rng.random() < 0.2:
                    jump = rng.uniform(-2, 2, dim)
                    new_x = best_x + jump
                    new_x = np.clip(new_x, -5, 5)
                    f_new = evaluate(new_x)
                    if f_new < current_f:
                        current_x = new_x.copy()
                        current_f = f_new
                    # Reset steps after jump
                    steps = np.ones(dim) * 1.0

        return best_x, best_f
# EVOLVE-BLOCK-END
