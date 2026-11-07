"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio of (1+1)-ES restarts with simple 1/5-rule adaptation.

    Strategy:
      1. Four (1+1)-ES restarts with diverse random initializations and step sizes.
      2. Simple 1/5-rule step-size adaptation (1.5x on success, 0.87x on failure).
      3. Equal budget allocation across restarts for consistent diversity.
      4. Final local refinement around the best point found using a smaller sigma.
      5. Explicit wall-clock deadline guard.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin

        call_count = [0]

        def make_call(x):
            if call_count[0] >= budget:
                return None
            call_count[0] += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            return float(func(x))

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Coarse grid-based exploration
        # Systematically sample the domain to identify the global basin of attraction.
        # A 3x3x3x3x3 grid covers [-5, 5]^5 with 243 points.
        # This ensures early AOCC gains by avoiding local minima traps inherent in random-start methods.
        grid_levels = 3
        grid_points = []
        for i in range(grid_levels):
            for j in range(grid_levels):
                for k in range(grid_levels):
                    for l in range(grid_levels):
                        for m in range(grid_levels):
                            grid_points.append([
                                -5.0 + 10.0 * i / (grid_levels - 1),
                                -5.0 + 10.0 * j / (grid_levels - 1),
                                -5.0 + 10.0 * k / (grid_levels - 1),
                                -5.0 + 10.0 * l / (grid_levels - 1),
                                -5.0 + 10.0 * m / (grid_levels - 1),
                            ])
        
        # Shuffle grid points to avoid bias from ordering
        rng.shuffle(grid_points)
        
        # Evaluate all grid points (or as many as budget allows)
        for pt in grid_points:
            if call_count[0] >= budget:
                break
            if time.time() > deadline - 20:
                break
            x = np.array(pt, dtype=np.float64)
            fx = make_call(x)
            if fx is None:
                break
            update_best(x, fx)
        
        # Phase 2: Targeted high-resolution local search from the best grid point
        if best_x is not None and call_count[0] < budget - 5 and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            # Start refinement with a moderate sigma to exploit the best region
            sigma = 0.5
            remaining_calls = budget - call_count[0] - 5
            
            for _ in range(remaining_calls):
                if call_count[0] >= budget - 5:
                    break
                if time.time() > deadline - 5:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = make_call(y)
                if fy is None:
                    break

                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.3
                else:
                    sigma *= 0.8

                sigma = float(np.clip(sigma, 0.01, 5.0))

        return best_x, best_f
# EVOLVE-BLOCK-END
