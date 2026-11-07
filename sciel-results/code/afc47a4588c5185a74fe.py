"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with coordinate-descent polishing for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Phase 1: Multi-start coarse search with adaptive step sizes
        # Use a population-based approach with multiple restarts
        best_x = None
        best_f = np.inf
        calls_used = 0

        def eval_func(x):
            nonlocal calls_used
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            fx = float(func(x))
            calls_used += 1
            return x, fx

        # Initial diverse starts
        n_starts = min(10, budget // 50)
        starts = []
        for i in range(n_starts):
            x = rng.uniform(-4, 4, dim)
            x, fx = eval_func(x)
            starts.append((fx, x))
            if fx < best_f:
                best_f, best_x = fx, x.copy()

        # Phase 2: CMA-ES-like evolution from the best start
        # Use a simple population-based evolutionary strategy
        pop_size = min(15, max(5, (budget - calls_used) // 20))
        sigma = 1.5
        x_mean = best_x.copy()
        f_mean = best_f

        for gen in range(200):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break

            # Generate offspring
            offspring = []
            for _ in range(pop_size):
                if calls_used >= budget:
                    break
                # Mix of local and global exploration
                if rng.random() < 0.7:
                    y = x_mean + sigma * rng.standard_normal(dim)
                else:
                    y = x_mean + sigma * 2.0 * rng.standard_normal(dim)
                y, fy = eval_func(y)
                offspring.append((fy, y))
                if fy < best_f:
                    best_f, best_x = fy, y.copy()

            if not offspring:
                break

            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[0])
            # Update mean using top half
            top_k = max(1, pop_size // 2)
            top = [o[1] for o in offspring[:top_k]]
            x_mean = np.mean(top, axis=0)
            f_mean = offspring[0][0]

            # Adapt sigma based on improvement
            if offspring[0][0] < f_mean:
                sigma *= 1.2
            else:
                sigma *= 0.85

            f_mean = offspring[0][0]

            # If sigma becomes too small, try a restart
            if sigma < 0.01:
                # Coordinate descent polish
                break

        # Phase 3: Coordinate-descent polishing (exploits axis-aligned structure of f6)
        if best_x is not None and calls_used < budget:
            # Multiple passes of coordinate descent with decreasing step sizes
            step_sizes = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
            for step in step_sizes:
                if time.time() > deadline or calls_used >= budget:
                    break
                x_cur = best_x.copy()
                f_cur = best_f
                improved = True
                while improved and calls_used < budget and time.time() < deadline:
                    improved = False
                    for d in range(dim):
                        if calls_used >= budget or time.time() > deadline:
                            break
                        # Try positive and negative steps along each coordinate
                        for direction in [1, -1]:
                            y = x_cur.copy()
                            y[d] += direction * step
                            y, fy = eval_func(y)
                            if fy < f_cur:
                                x_cur = y.copy()
                                f_cur = fy
                                improved = True
                                if fy < best_f:
                                    best_f, best_x = fy, x_cur.copy()
                                break  # Don't try opposite direction if we improved

        # Phase 4: Final fine-grained random perturbation polish
        if best_x is not None and calls_used < budget:
            sigma_fine = 0.01
            for _ in range(min(50, budget - calls_used)):
                if time.time() > deadline:
                    break
                y = best_x + sigma_fine * rng.standard_normal(dim)
                y, fy = eval_func(y)
                if fy < best_f:
                    best_f, best_x = fy, y.copy()
                # Occasionally try larger perturbations to escape local minima
                if rng.random() < 0.1:
                    y = best_x + 0.1 * rng.standard_normal(dim)
                    y, fy = eval_func(y)
                    if fy < best_f:
                        best_f, best_x = fy, y.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
