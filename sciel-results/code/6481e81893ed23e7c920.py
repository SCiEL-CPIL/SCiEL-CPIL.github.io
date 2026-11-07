"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid deterministic coordinate descent + CMA-ES for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 270  # safety margin before 300s kill
        budget = self.budget
        dim = self.dim
        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def eval_point(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Deterministic coordinate descent with large steps
        # Try multiple starting points and large step sizes to quickly
        # traverse the search space
        rng = np.random.default_rng(42)

        # Start from several deterministic points
        start_points = [
            np.zeros(dim),
            np.ones(dim) * 4.0,
            -np.ones(dim) * 4.0,
            np.array([4.0, -4.0, 4.0, -4.0, 4.0]),
            np.array([-4.0, 4.0, -4.0, 4.0, -4.0]),
        ]

        # Coordinate descent with adaptive step size
        for sp in start_points:
            if calls >= budget or time.time() > deadline:
                break

            x = sp.copy()
            fx = eval_point(x)

            # Try large steps first, then refine
            step_sizes = [4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
            for ss in step_sizes:
                if calls >= budget or time.time() > deadline:
                    break
                improved = True
                while improved and calls < budget and time.time() < deadline:
                    improved = False
                    for d in range(dim):
                        if calls >= budget or time.time() > deadline:
                            break
                        for direction in [1, -1]:
                            x_new = x.copy()
                            x_new[d] += direction * ss
                            if np.any(x_new < -5) or np.any(x_new > 5):
                                continue
                            fx_new = eval_point(x_new)
                            if fx_new < fx:
                                x = x_new
                                fx = fx_new
                                improved = True
                                break

        # Phase 2: CMA-ES refinement around the best point found
        if calls < budget and time.time() < deadline:
            x = best_x.copy()
            sigma = 1.0
            # Simple (1+1)-ES refinement
            for _ in range(min(budget - calls, 200)):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = eval_point(y)
                if fy < best_f:
                    x = y
                    sigma *= 1.3
                else:
                    sigma *= 0.85
                    if sigma < 0.001:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
