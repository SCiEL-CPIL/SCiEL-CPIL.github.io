"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-Start Local Search with Fixed-Step Two-Phase Refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        
        # Partition budget into independent restarts
        n_restarts = 30
        evals_per_restart = budget // n_restarts  # 10
        
        best_x = None
        best_f = np.inf

        for r in range(n_restarts):
            # 1. Random Initialization
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            remaining = evals_per_restart - 1  # 9 evals left
            
            if remaining <= 0:
                continue

            # Phase 1: Coarse Search (Large Steps)
            # Use 5 evals for coarse steps
            step_coarse = 2.0
            coarse_budget = min(5, remaining)
            for _ in range(coarse_budget):
                if remaining <= 0: break
                d = rng.integers(dim)
                # Try positive
                y = x.copy()
                y[d] = np.clip(y[d] + step_coarse, -5, 5)
                fy = func(y)
                remaining -= 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Try negative if evals remain
                    if remaining <= 0: break
                    y = x.copy()
                    y[d] = np.clip(y[d] - step_coarse, -5, 5)
                    fy = func(y)
                    remaining -= 1
                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

            # Phase 2: Fine Search (Small Steps)
            # Use remaining evals for fine steps
            step_fine = 0.5
            while remaining > 0:
                d = rng.integers(dim)
                # Try positive
                y = x.copy()
                y[d] = np.clip(y[d] + step_fine, -5, 5)
                fy = func(y)
                remaining -= 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Try negative if evals remain
                    if remaining <= 0: break
                    y = x.copy()
                    y[d] = np.clip(y[d] - step_fine, -5, 5)
                    fy = func(y)
                    remaining -= 1
                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
