"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start adaptive optimizer for Rastrigin-rotated f15."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best
        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            calls_made += 1
            return func(x)

        # Phase 1: Coarse multi-start with Nelder-Mead-like local search
        # Use several diverse starting points
        n_starts = min(15, max(3, budget // 40))
        starts = []
        # Include origin and random points
        starts.append(np.zeros(dim))
        for i in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, dim))

        # Phase 2: For each start, do a local adaptive search
        for s_idx, x0 in enumerate(starts):
            if calls_made >= budget:
                break
            if time.time() > deadline - 5:
                break

            x = np.clip(x0.copy(), -5, 5)
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Adaptive step size per dimension
            step = np.ones(dim) * 1.5
            success_count = 0
            max_iters = max(5, (budget - calls_made) // max(1, n_starts - s_idx))

            for it in range(max_iters):
                if calls_made >= budget:
                    break
                if time.time() > deadline - 5:
                    break

                # Try to move in each dimension
                improved = False
                for d in range(dim):
                    if calls_made >= budget:
                        break
                    if time.time() > deadline - 5:
                        break

                    # Try positive step
                    x_trial = x.copy()
                    x_trial[d] = np.clip(x_trial[d] + step[d], -5, 5)
                    if not np.allclose(x_trial, x):
                        f_trial = safe_func(x_trial)
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            step[d] *= 1.3
                            success_count += 1
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            continue

                    # Try negative step
                    x_trial = x.copy()
                    x_trial[d] = np.clip(x_trial[d] - step[d], -5, 5)
                    if not np.allclose(x_trial, x):
                        f_trial = safe_func(x_trial)
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            step[d] *= 1.3
                            success_count += 1
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()

                if not improved:
                    # Shrink all steps
                    step *= 0.6
                    if np.max(step) < 0.01:
                        break

            # Local polish with small random perturbations
            if calls_made < budget and time.time() < deadline - 5:
                for _ in range(10):
                    if calls_made >= budget:
                        break
                    if time.time() > deadline - 5:
                        break
                    x_trial = np.clip(x + rng.standard_normal(dim) * 0.1, -5, 5)
                    f_trial = safe_func(x_trial)
                    if f_trial < fx:
                        x = x_trial
                        fx = f_trial
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

        # Phase 3: Final global exploration if budget remains
        if calls_made < budget and time.time() < deadline - 5:
            remaining = budget - calls_made
            n_extra = min(remaining, 50)
            for _ in range(n_extra):
                if calls_made >= budget:
                    break
                if time.time() > deadline - 5:
                    break
                x_trial = rng.uniform(-5, 5, dim)
                f_trial = safe_func(x_trial)
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial.copy()
                    # Local refinement around this new best
                    x = x_trial.copy()
                    fx = f_trial
                    step = np.ones(dim) * 0.5
                    for _ in range(15):
                        if calls_made >= budget:
                            break
                        if time.time() > deadline - 5:
                            break
                        improved = False
                        for d in range(dim):
                            if calls_made >= budget:
                                break
                            x_trial2 = x.copy()
                            x_trial2[d] = np.clip(x_trial2[d] + step[d], -5, 5)
                            if not np.allclose(x_trial2, x):
                                f_trial2 = safe_func(x_trial2)
                                if f_trial2 < fx:
                                    x = x_trial2
                                    fx = f_trial2
                                    step[d] *= 1.2
                                    improved = True
                                    if fx < best_f:
                                        best_f = fx
                                        best_x = x.copy()
                                    continue
                            x_trial2 = x.copy()
                            x_trial2[d] = np.clip(x_trial2[d] - step[d], -5, 5)
                            if not np.allclose(x_trial2, x):
                                f_trial2 = safe_func(x_trial2)
                                if f_trial2 < fx:
                                    x = x_trial2
                                    fx = f_trial2
                                    step[d] *= 1.2
                                    improved = True
                                    if fx < best_f:
                                        best_f = fx
                                        best_x = x.copy()
                        if not improved:
                            step *= 0.5
                            if np.max(step) < 0.01:
                                break

        if best_x is None:
            best_x = np.zeros(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
