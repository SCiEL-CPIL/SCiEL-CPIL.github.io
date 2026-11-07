"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid: population exploration + (1+1)-ES exploitation + Rastrigin-aware refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s kill

        def time_ok():
            return time.time() < deadline

        # Phase 1: Population-based exploration to find promising basins
        # Use a small population with different initializations
        pop_size = 10
        population = []
        for i in range(pop_size):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            population.append((x, fx))

        calls = len(population)
        if not population:
            return None, None

        # Sort by fitness and keep best
        population.sort(key=lambda t: t[1])
        best_x, best_fx = population[0]

        # Phase 2: (1+1)-ES exploitation from the best point found
        x = best_x.copy()
        fx = best_fx
        sigma = 1.5

        # Run (1+1)-ES with adaptive sigma
        for _ in range(budget - calls - 50):  # Reserve some calls for refinement
            if not time_ok():
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
            else:
                sigma *= 0.9
                if sigma < 1e-8:
                    break

        # Phase 3: Rastrigin-aware refinement
        # Rastrigin has local minima at integer points; try snapping coordinates
        remaining = budget - calls
        if remaining > 10 and time_ok():
            # Try integer snapping for each coordinate
            for coord in range(dim):
                if not time_ok() or calls >= budget - 5:
                    break
                # Try rounding current coordinate to nearest integer
                x_int = x.copy()
                x_int[coord] = np.round(x[coord])
                x_int = np.clip(x_int, -5, 5)
                fx_int = func(x_int)
                calls += 1
                if fx_int < fx:
                    x, fx = x_int, fx_int

            # Coordinate-wise local search with small steps
            sigma_small = 0.1
            for _ in range(min(remaining - 5, 30)):
                if not time_ok() or calls >= budget - 2:
                    break
                y = np.clip(x + sigma_small * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma_small *= 1.1
                else:
                    sigma_small *= 0.85
                    if sigma_small < 1e-9:
                        break

        return x, fx
# EVOLVE-BLOCK-END
