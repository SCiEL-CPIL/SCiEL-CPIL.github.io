"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: LHS init + population-based search with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Population size: balance diversity and exploitation
        pop_size = min(20, max(10, budget // 10))

        # 1. LHS Initialization
        perm = np.argsort(np.argsort(np.random.rand(pop_size, dim)), axis=0)
        u = (perm + np.random.rand(pop_size, dim)) / pop_size
        X = lo + range_ * u

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track best hypervolume proxy: sum of minima per objective (lower is better)
        best_sum = np.sum(np.min(F, axis=0))
        no_improve_count = 0
        stagnation_threshold = 30  # evaluations without improvement before boosting mutation

        # 2. Iterative refinement
        while used < budget:
            if time.time() > deadline:
                break

            # Identify non-dominated points
            non_dom_mask = np.ones(pop_size, dtype=bool)
            for i in range(pop_size):
                if non_dom_mask[i]:
                    for j in range(pop_size):
                        if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            non_dom_mask[i] = False
                            break

            # Select parent: prefer non-dominated
            if np.any(non_dom_mask):
                parent_idx = np.random.choice(np.where(non_dom_mask)[0])
            else:
                parent_idx = np.random.randint(pop_size)

            x_parent = X[parent_idx]

            # Adaptive step size: base 0.1, boost when stagnant
            base_step = 0.1
            if no_improve_count > stagnation_threshold:
                step = base_step * (1.0 + 0.5 * min(1.0, (no_improve_count - stagnation_threshold) / 50.0))
            else:
                # Slightly decrease as we approach budget end for exploitation
                progress = used / budget
                step = base_step * (1.0 - 0.3 * progress)

            # Gaussian mutation
            x_child = x_parent + step * range_ * np.random.randn(dim)
            x_child = np.clip(x_child, lo, hi)

            # Periodic random injection: every 50 evaluations, inject a random point
            if used % 50 == 0:
                x_child = lo + range_ * np.random.rand(dim)

            f_child = problem(x_child)
            used += 1

            if np.any(np.isnan(f_child)) or np.any(np.isinf(f_child)):
                continue

            # Replace the worst individual (highest sum of objectives)
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = x_child
            F[worst_idx] = f_child

            # Check for improvement
            current_best_sum = np.sum(np.min(F, axis=0))
            if current_best_sum < best_sum:
                best_sum = current_best_sum
                no_improve_count = 0
            else:
                no_improve_count += 1

        return X
# EVOLVE-BLOCK-END
