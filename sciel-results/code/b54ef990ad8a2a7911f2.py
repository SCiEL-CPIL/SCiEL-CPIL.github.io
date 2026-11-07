"""Deterministic hierarchical coordinate-wise descent for Katsuura-like f23."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Deterministic hierarchical coordinate-wise descent exploiting the
    separable structure of the Katsuura function.
    
    Phase 1 (calls 1-250): Coarse grid search to locate the global basin.
    Phase 2 (calls 251-500): Fine-grained coordinate-wise descent with
    adaptive step sizes for rapid convergence.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls = 0

        def evaluate(x):
            nonlocal calls
            calls += 1
            return float(func(x))

        # Phase 1: Coarse grid search
        # For each dimension, try a grid of candidate values while keeping
        # other dimensions at a reasonable default (0 or previous best).
        # We use a hierarchical approach: first find good values for each
        # dimension independently, then refine jointly.

        # Start with center of the box
        x = np.zeros(dim)
        fx = evaluate(x)
        best_x = x.copy()
        best_f = fx

        # Coarse grid points for each dimension
        # Use 9 points per dimension in [-4, 4] to stay within safe range
        # where the minimum is guaranteed to be
        coarse_points = np.linspace(-4.0, 4.0, 9)

        # For each dimension, sweep through coarse points
        # Keeping other dimensions at their current best values
        for d in range(dim):
            for pt in coarse_points:
                if calls >= 250:
                    break
                candidate = best_x.copy()
                candidate[d] = pt
                f_val = evaluate(candidate)
                if f_val < best_f:
                    best_f = f_val
                    best_x = candidate

        # If we've used more than half the budget, skip to fine phase
        # Otherwise, do a second pass with refined grid
        if calls < 250:
            # Second pass with intermediate grid
            mid_points = np.linspace(-4.0, 4.0, 17)
            for d in range(dim):
                for pt in mid_points:
                    if calls >= 250:
                        break
                    candidate = best_x.copy()
                    candidate[d] = pt
                    f_val = evaluate(candidate)
                    if f_val < best_f:
                        best_f = f_val
                        best_x = candidate

        # Phase 2: Fine coordinate-wise descent with adaptive steps
        # Use geometrically decreasing step sizes
        # For Katsuura, the function has many local minima but the global
        # minimum is unique. After locating the basin, coordinate-wise
        # descent with small steps should converge quickly.

        # Define a sequence of step sizes to try
        base_steps = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001,
                       0.0005, 0.0002, 0.0001]

        current_x = best_x.copy()
        current_f = best_f

        for step in base_steps:
            # Multiple passes with this step size
            for _ in range(3):  # 3 passes per step size
                for d in range(dim):
                    if calls >= budget:
                        break
                    # Try moving in both directions
                    for direction in [-1, 1]:
                        if calls >= budget:
                            break
                        candidate = current_x.copy()
                        candidate[d] += direction * step
                        # Clip to box
                        candidate[d] = np.clip(candidate[d], -5.0, 5.0)
                        f_val = evaluate(candidate)
                        if f_val < current_f:
                            current_f = f_val
                            current_x = candidate

                if calls >= budget:
                    break

            if calls >= budget:
                break

        # Final local refinement with very small steps if budget remains
        if calls < budget:
            for step in [0.00005, 0.00002, 0.00001]:
                for _ in range(2):
                    for d in range(dim):
                        if calls >= budget:
                            break
                        for direction in [-1, 1]:
                            if calls >= budget:
                                break
                            candidate = current_x.copy()
                            candidate[d] += direction * step
                            candidate[d] = np.clip(candidate[d], -5.0, 5.0)
                            f_val = evaluate(candidate)
                            if f_val < current_f:
                                current_f = f_val
                                current_x = candidate

                    if calls >= budget:
                        break

                if calls >= budget:
                    break

        return current_x, current_f
# EVOLVE-BLOCK-END
