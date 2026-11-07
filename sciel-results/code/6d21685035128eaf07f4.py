"""Direct-search pattern search with time guard for BBOB f18."""
import numpy as np
import time


class Optimizer:
    """Pattern search with orthogonal/diagonal directions and adaptive step sizes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Build initial search directions: standard basis + diagonal directions
        directions = [np.eye(dim)]  # shape (dim, dim)
        # Add diagonal directions: ±1 in all coordinates
        diag = np.ones(dim)
        directions.append(np.outer(diag, diag))  # not quite, let me use proper vectors
        # Actually, let's use a set of 1D direction vectors
        dir_vectors = []
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            dir_vectors.append(e)
        # Diagonal directions
        d1 = np.ones(dim) / np.sqrt(dim)
        d2 = np.ones(dim) / np.sqrt(dim)
        d2[0] = -d2[0]
        dir_vectors.append(d1)
        dir_vectors.append(d2)

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        # Initial step sizes per direction
        step_sizes = np.array([1.0] * len(dir_vectors))

        # Phase 1: Explore with random restarts to find a good region
        # Use first ~100 evaluations for exploration
        explore_budget = min(80, budget // 5)
        for _ in range(explore_budget):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            x_try = rng.uniform(-5, 5, dim)
            fy = func(x_try)
            calls += 1
            if fy < best_f:
                best_f = fy
                best_x = x_try.copy()

        # Now do pattern search from best_x
        x = best_x.copy()
        fx = best_f

        # Reset step sizes
        step_sizes = np.array([0.5] * len(dir_vectors))

        # Pattern search loop
        no_improve_count = 0
        max_no_improve = 100

        while calls < budget:
            if time.time() > deadline:
                break

            improved = False
            for i in range(len(dir_vectors)):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

                d = dir_vectors[i]
                step = step_sizes[i]

                # Try positive direction
                x_plus = np.clip(x + step * d, -5, 5)
                if not np.allclose(x_plus, x):
                    fp = func(x_plus)
                    calls += 1
                    if fp < fx:
                        x = x_plus
                        fx = fp
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        improved = True
                        step_sizes[i] *= 1.5
                        no_improve_count = 0
                        continue

                # Try negative direction
                x_minus = np.clip(x - step * d, -5, 5)
                if not np.allclose(x_minus, x):
                    fm = func(x_minus)
                    calls += 1
                    if fm < fx:
                        x = x_minus
                        fx = fm
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        improved = True
                        step_sizes[i] *= 1.5
                        no_improve_count = 0
                        continue

                # No improvement in this direction, shrink step
                step_sizes[i] *= 0.5

            if not improved:
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    # Stagnation: restart from a perturbed point near best
                    if calls + 10 > budget:
                        break
                    x = best_x + 0.5 * rng.standard_normal(dim)
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Reset step sizes
                    step_sizes = np.array([0.5] * len(dir_vectors))
                    no_improve_count = 0

        return best_x, best_f
