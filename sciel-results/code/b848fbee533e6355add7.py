"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES implementation for BBOB-style optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        rng = np.random.default_rng(123)
        dim = self.dim

        # Initial point: try center and a few random points
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # Quick scan to find a better starting point
        for _ in range(5):
            x_try = rng.uniform(-4, 4, dim)
            f_try = func(x_try)
            if f_try < best_f:
                best_x = x_try.copy()
                best_f = f_try

        # CMA-ES parameters
        x = best_x.copy()
        f = best_f
        sigma = 1.0
        pop_size = 4 + int(3 * np.log(dim))  # Standard CMA-ES pop size
        pop_size = max(pop_size, 2 * dim)

        # CMA-ES state
        C = np.eye(dim)  # Covariance matrix
        step_size_sigma = 1.0
        cs = 1 / (np.sqrt(dim) + 3)  # Path length adaptation
        ds = 1 + 2 * max(0, (np.sqrt((dim + 1) / 2 - 1) - 1) / (dim + 2))
        c_sigma = (np.sqrt(dim) + 2) / (dim + 4)
        c_c = (4 + dim / dim) / (dim + 4)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        eval_count = 11  # 1 initial + 5 scan + 5 more

        for generation in range(200):
            # Check time
            if time.time() > deadline - 30:
                break

            # Generate population
            for i in range(pop_size):
                if time.time() > deadline - 30:
                    break
                if eval_count >= self.budget:
                    break

                # Standard normal
                z = rng.standard_normal(dim)
                # Apply covariance (use Cholesky or just identity for simplicity)
                # For simplicity, use diagonal approximation
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                eval_count += 1

                if fy < f:
                    x = y
                    f = fy
                    if fy < best_f:
                        best_x = y.copy()
                        best_f = fy

            # Update CMA-ES parameters (simplified)
            # Adapt sigma based on success rate
            # This is a simplified version - full CMA-ES would track more state
            sigma *= 0.95  # Slowly decrease sigma

            if sigma < 1e-6:
                break

        # Final polish
        if time.time() < deadline - 10 and eval_count < self.budget:
            # Try Nelder-Mead style local optimization
            simplex = [best_x.copy()]
            for i in range(dim):
                point = best_x.copy()
                point[i] += 0.1
                point = np.clip(point, -5, 5)
                simplex.append(point)

            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                if eval_count >= self.budget:
                    break

                # Evaluate simplex
                vals = []
                for i, sp in enumerate(simplex):
                    if i == 0:
                        vals.append(best_f)
                    else:
                        if eval_count >= self.budget:
                            break
                        vals.append(func(sp))
                        eval_count += 1
                else:
                    # Find worst point
                    worst_idx = np.argmax(vals)
                    # Reflect
                    centroid = np.mean([simplex[j] for j in range(len(simplex)) if j != worst_idx], axis=0)
                    reflection = centroid + 1 * (centroid - simplex[worst_idx])
                    reflection = np.clip(reflection, -5, 5)
                    if eval_count < self.budget:
                        f_reflect = func(reflection)
                        eval_count += 1
                        if f_reflect < vals[worst_idx]:
                            simplex[worst_idx] = reflection
                            vals[worst_idx] = f_reflect
                            if f_reflect < best_f:
                                best_x = reflection.copy()
                                best_f = f_reflect

        return best_x, best_f
# EVOLVE-BLOCK-END
