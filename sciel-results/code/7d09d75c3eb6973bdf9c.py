"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-Start Local Search with Expanded Fixed-Step Schedule."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        n_restarts = 30
        evals_per_restart = budget // n_restarts  # 10
        if evals_per_restart < 1:
            evals_per_restart = 1

        best_x = None
        best_f = np.inf

        # Expanded step schedule to cover larger distances and finer details
        steps = [3.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]

        for r in range(n_restarts):
            # Random start point
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            remaining = evals_per_restart - 1
            if remaining <= 0:
                continue

            # Randomize coordinate order to break symmetry
            coord_order = rng.permutation(dim)

            # Iterate through step sizes from large to small
            for step in steps:
                if remaining <= 0:
                    break
                
                improved = True
                while improved and remaining > 0:
                    improved = False
                    for d in coord_order:
                        if remaining <= 0:
                            break
                        
                        # Try positive direction
                        y = x.copy()
                        y[d] = np.clip(y[d] + step, -5, 5)
                        fy = func(y)
                        remaining -= 1
                        if fy < fx:
                            x, fx = y, fy
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            break  # Move to next coordinate after improvement
                    
                        # Try negative direction if positive failed
                        if remaining <= 0:
                            break
                        y = x.copy()
                        y[d] = np.clip(y[d] - step, -5, 5)
                        fy = func(y)
                        remaining -= 1
                        if fy < fx:
                            x, fx = y, fy
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            break  # Move to next coordinate after improvement

        return best_x, best_f
# EVOLVE-BLOCK-END
