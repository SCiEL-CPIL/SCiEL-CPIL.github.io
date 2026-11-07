import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s kill

        best_x = np.zeros(dim)
        best_f = np.inf
        evals = 0

        def eval_point(x):
            nonlocal best_x, best_f, evals
            f = float(func(x))
            evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_left():
            return (deadline - time.time()) > 2.0

        def check_time():
            return evals < budget and time_left()

        # Phase 1: Deterministic lattice search to identify basin
        # Use a grid with spacing ~1.0 over [-4, 4]^5
        # That's 9^5 = 59049 points - too many for budget 500
        # Instead, use a coarser grid and then refine
        # Grid: 5 points per dimension from -4 to 4 -> 5^5 = 3125, still too many
        # Use 3 points per dim: -4, 0, 4 -> 3^5 = 241 points
        # Then refine around best with 5 points per dim in a smaller region

        # First pass: coarse grid
        coarse_vals = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
        grid_points = np.array(np.meshgrid(*[coarse_vals] * dim)).T.reshape(-1, dim)

        for i in range(len(grid_points)):
            if not check_time():
                break
            eval_point(grid_points[i])

        # Phase 2: Local refinement around best point using coordinate-wise line search
        # Schaffers F7 is ill-conditioned but axis-aligned in its local structure
        # Use a multi-scale coordinate descent approach

        def coordinate_line_search(x_center, f_center, scales, max_iter_per_coord=10):
            """
            For each coordinate, try small perturbations at multiple scales
            and keep the best improvement. Repeat until no improvement.
            """
            nonlocal best_x, best_f, evals
            x = x_center.copy()
            f = f_center
            improved = True
            while improved and check_time():
                improved = False
                for d in range(dim):
                    for scale in scales:
                        for direction in [-1, 1]:
                            x_trial = x.copy()
                            x_trial[d] += direction * scale
                            # Clip to box
                            x_trial = np.clip(x_trial, -5.0, 5.0)
                            if not check_time():
                                break
                            f_trial = eval_point(x_trial)
                            if f_trial < f:
                                x = x_trial
                                f = f_trial
                                improved = True
                        if not check_time():
                            break
                    if not check_time():
                        break
            return x, f

        # Multi-scale coordinate descent
        scales = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125,
                  0.00390625, 0.001953125, 0.0009765625]

        # Refine from the best point found in coarse grid
        x_ref = best_x.copy()
        f_ref = best_f

        # Do several rounds of coordinate descent with decreasing scales
        for scale_idx in range(len(scales)):
            if not check_time():
                break
            current_scale = scales[scale_idx]
            # Try perturbations in each coordinate
            for d in range(dim):
                if not check_time():
                    break
                for direction in [-1, 1]:
                    x_trial = x_ref.copy()
                    x_trial[d] += direction * current_scale
                    x_trial = np.clip(x_trial, -5.0, 5.0)
                    if not check_time():
                        break
                    f_trial = eval_point(x_trial)
                    if f_trial < f_ref:
                        x_ref = x_trial
                        f_ref = f_trial

        # Phase 3: Further refinement with even smaller steps
        if check_time():
            fine_scales = [0.0005, 0.0002, 0.0001, 0.00005, 0.00002]
            for scale in fine_scales:
                if not check_time():
                    break
                for d in range(dim):
                    if not check_time():
                        break
                    for direction in [-1, 1]:
                        x_trial = x_ref.copy()
                        x_trial[d] += direction * scale
                        x_trial = np.clip(x_trial, -5.0, 5.0)
                        if not check_time():
                            break
                        f_trial = eval_point(x_trial)
                        if f_trial < f_ref:
                            x_ref = x_trial
                            f_ref = f_trial

        # Phase 4: If budget remains, do a second pass of coordinate descent
        # starting from slightly perturbed versions of the best point
        if check_time():
            # Try a few random restarts near the best point
            for restart in range(5):
                if not check_time():
                    break
                x_start = x_ref + np.random.randn(dim) * 0.1
                x_start = np.clip(x_start, -5.0, 5.0)
                f_start = eval_point(x_start)
                if f_start < f_ref:
                    x_ref = x_start.copy()
                    f_ref = f_start

                # Quick coordinate descent from this point
                for scale in [0.05, 0.02, 0.01, 0.005, 0.002]:
                    if not check_time():
                        break
                    for d in range(dim):
                        if not check_time():
                            break
                        for direction in [-1, 1]:
                            x_trial = x_ref.copy()
                            x_trial[d] += direction * scale
                            x_trial = np.clip(x_trial, -5.0, 5.0)
                            if not check_time():
                                break
                            f_trial = eval_point(x_trial)
                            if f_trial < f_ref:
                                x_ref = x_trial
                                f_ref = f_trial

        return best_x, best_f
