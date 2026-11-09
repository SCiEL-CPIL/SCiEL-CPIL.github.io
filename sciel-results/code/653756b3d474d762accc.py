"""Rewritten program for BBOB f7 (Step ellipsoidal) using iterated local search with coordinate pattern search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Iterated local search with coordinate-wise pattern search for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Helper to check if we can continue
        def can_continue():
            return calls < budget and (time.time() - start_time) < 280

        # Phase 1: Global exploration with diverse random starts
        x_best = np.clip(rng.uniform(-5, 5, dim), -5, 5)
        f_best = func(x_best)
        calls = 1

        # Evaluate a spread of candidate starting points
        n_starts = 15
        for i in range(n_starts):
            if not can_continue():
                break
            if i == 0:
                x_try = np.zeros(dim)
            elif i == 1:
                x_try = rng.uniform(-1, 1, dim)
            elif i == 2:
                x_try = rng.uniform(-3, 3, dim)
            else:
                x_try = rng.uniform(-5, 5, dim)
            x_try = np.clip(x_try, -5, 5)
            f_try = func(x_try)
            calls += 1
            if f_try < f_best:
                f_best = f_try
                x_best = x_try.copy()

        # Phase 2: Iterated local search with coordinate-wise pattern refinement
        # Each iteration: refine current best via coordinate search, then perturb for next iteration
        n_its = 30  # max iterations of the outer loop
        for it in range(n_its):
            if not can_continue():
                break

            # Coordinate-wise pattern search to refine x_best
            x_cur = x_best.copy()
            f_cur = f_best
            step_sizes = [5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

            for step in step_sizes:
                if not can_continue():
                    break
                improved = False
                for coord in range(dim):
                    if not can_continue():
                        break
                    # Try positive direction
                    x_pos = x_cur.copy()
                    x_pos[coord] += step
                    if x_pos[coord] > 5.0:
                        continue
                    f_pos = func(x_pos)
                    calls += 1
                    if f_pos < f_cur:
                        x_cur = x_pos
                        f_cur = f_pos
                        improved = True
                        continue

                    # Try negative direction
                    x_neg = x_cur.copy()
                    x_neg[coord] -= step
                    if x_neg[coord] < -5.0:
                        continue
                    f_neg = func(x_neg)
                    calls += 1
                    if f_neg < f_cur:
                        x_cur = x_neg
                        f_cur = f_neg
                        improved = True

                if not improved:
                    break  # No improvement at this step size, move to smaller step

            # Update global best if coordinate search found something better
            if f_cur < f_best:
                f_best = f_cur
                x_best = x_cur.copy()

            # Perturbation for next iteration (simulated annealing-like)
            if not can_continue():
                break

            # Adaptive perturbation: larger when stuck, smaller when improving
            # Use a simple scheme: perturb with decreasing magnitude
            perturbation_scale = 1.5 * (0.95 ** it)
            x_pert = x_best + perturbation_scale * rng.standard_normal(dim)
            x_pert = np.clip(x_pert, -5, 5)
            f_pert = func(x_pert)
            calls += 1

            # Accept if better or with probability based on "temperature"
            if f_pert < f_best:
                x_best = x_pert.copy()
                f_best = f_pert
            else:
                # Occasionally accept worse solutions to escape plateaus
                if rng.random() < 0.1 * (0.95 ** it):
                    x_best = x_pert.copy()

        return x_best, f_best
# EVOLVE-BLOCK-END
