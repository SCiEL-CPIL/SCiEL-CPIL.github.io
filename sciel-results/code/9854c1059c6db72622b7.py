"""Coordinate Descent with adaptive per-axis steps for BBOB f13 (Sharp Ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive coordinate descent with per-axis step sizes and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin under 300s

        rng = np.random.default_rng(12345)

        # --- Phase 0: initial broad sampling to find a decent starting point ---
        n_init = min(20, budget // 10)
        candidates = rng.uniform(-4.5, 4.5, size=(n_init, dim))
        best_x = None
        best_f = np.inf
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = candidates[i]
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = rng.uniform(-4, 4, dim)
            best_f = float(func(best_x))

        # --- Phase 1: coordinate descent with adaptive per-axis steps ---
        x = best_x.copy()
        fx = best_f
        # Per-axis step sizes: start moderately wide
        steps = np.full(dim, 1.5)
        axis_fail = np.zeros(dim, dtype=int)

        calls_made = n_init
        stagnation_count = 0
        no_improve_window = 0

        while calls_made < budget:
            if time.time() > deadline:
                break

            improved_this_round = False
            # Cycle through dimensions
            for axis in range(dim):
                if calls_made >= budget or time.time() > deadline:
                    break

                step = steps[axis]

                # Try positive step
                x_plus = x.copy()
                x_plus[axis] = np.clip(x[axis] + step, -5.0, 5.0)
                f_plus = float(func(x_plus))
                calls_made += 1

                # Try negative step
                x_minus = x.copy()
                x_minus[axis] = np.clip(x[axis] - step, -5.0, 5.0)
                f_minus = float(func(x_minus))
                calls_made += 1

                if f_plus < fx and f_plus <= f_minus:
                    # Accept positive step
                    x = x_plus
                    fx = f_plus
                    improved_this_round = True
                    dir_sign = 1.0
                elif f_minus < fx:
                    # Accept negative step
                    x = x_minus
                    fx = f_minus
                    improved_this_round = True
                    dir_sign = -1.0
                else:
                    # No improvement: shrink step for this axis
                    steps[axis] = max(steps[axis] * 0.7, 0.01)
                    axis_fail[axis] += 1
                    continue

                # Adapt step: grow a bit on success
                steps[axis] = min(steps[axis] * 1.3, 4.0)
                axis_fail[axis] = 0

                # Quick line-search refinement along this axis in the improving direction
                current_step = steps[axis]
                for refine_iter in range(3):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    x_ref = x.copy()
                    x_ref[axis] = np.clip(x[axis] + dir_sign * current_step * 0.5, -5.0, 5.0)
                    f_ref = float(func(x_ref))
                    calls_made += 1
                    if f_ref < fx:
                        x = x_ref
                        fx = f_ref
                        improved_this_round = True
                        current_step *= 0.5
                    else:
                        break

            # Track stagnation
            if improved_this_round:
                stagnation_count = 0
                no_improve_window = 0
            else:
                stagnation_count += 1
                no_improve_window += 1

            # If stagnating with small steps, restart with perturbation
            if (stagnation_count >= 5 or
                    (no_improve_window >= 30 and np.all(steps < 0.1))):
                if calls_made + 1 <= budget:
                    perturbation = rng.standard_normal(dim) * 2.0
                    x_new = np.clip(x + perturbation, -5.0, 5.0)
                    f_new = float(func(x_new))
                    calls_made += 1
                    if f_new < fx:
                        x = x_new
                        fx = f_new
                    # Reset steps to medium
                    steps = np.full(dim, 1.0)
                    axis_fail = np.zeros(dim, dtype=int)
                    stagnation_count = 0
                    no_improve_window = 0

            # Occasional exploratory restart while still making progress
            if improved_this_round and calls_made < budget * 0.7:
                if rng.random() < 0.02:
                    x_exp = rng.uniform(-4.5, 4.5, dim)
                    f_exp = float(func(x_exp))
                    calls_made += 1
                    if f_exp < fx:
                        x = x_exp
                        fx = f_exp
                        steps = np.full(dim, 2.0)
                        stagnation_count = 0
                        no_improve_window = 0

            # If all steps are tiny, do a final random nudge
            if np.all(steps < 0.05) and calls_made < budget * 0.9:
                x_final = np.clip(x + rng.standard_normal(dim) * 0.5, -5.0, 5.0)
                f_final = float(func(x_final))
                calls_made += 1
                if f_final < fx:
                    x = x_final
                    fx = f_final
                steps = np.full(dim, 0.5)
                stagnation_count = 0
                no_improve_window = 0

        return x, fx
# EVOLVE-BLOCK-END
