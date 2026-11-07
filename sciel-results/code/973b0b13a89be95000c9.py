"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for RE37 with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # 300s limit, 30s safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use the evaluator's seeded RNG (np.random.seed was called before us)
        # This ensures reproducibility across seeds 0 and 1.
        # We'll use np.random directly since the evaluator seeds it.

        # Initial population: Latin Hypercube Sampling for good coverage
        pop_size = min(50, budget // 4)
        pop_size = max(pop_size, 10)

        # LHS initialization
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            X[:, d] = (perm + 0.5) / pop_size
        X = lo + X * span

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Store all evaluated points for final hypervolume
        all_X = X.copy()
        all_F = F.copy()

        # NSGA-II style loop
        # Population size for the evolutionary loop
        n_pop = min(40, budget - used)
        if n_pop < 4:
            n_pop = max(4, pop_size // 2)

        # We'll maintain a population of size n_pop
        # Start with the first n_pop points
        pop = X[:n_pop].copy()
        pop_F = F[:n_pop].copy()

        # Track remaining budget
        remaining = budget - used

        # Adaptive mutation: start large, shrink over time
        n_generations = max(1, remaining // n_pop)
        gen = 0

        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = self._non_dominated_sort(pop_F)

            # Build mating pool from top fronts
            # Select parents: prefer non-dominated (front 0)
            if len(fronts) > 0 and len(fronts[0]) > 0:
                # Sample from front 0 primarily, mix in front 1 if exists
                if len(fronts) > 1 and len(fronts[1]) > 0:
                    # 70% from front 0, 30% from front 1
                    n_from_0 = max(1, int(0.7 * n_pop))
                    n_from_1 = n_pop - n_from_0
                    idx0 = np.random.choice(len(fronts[0]), size=n_from_0, replace=True)
                    idx1 = np.random.choice(len(fronts[1]), size=n_from_1, replace=True)
                    parent_indices = list(fronts[0][idx0]) + list(fronts[1][idx1])
                else:
                    parent_indices = list(fronts[0])
                    while len(parent_indices) < n_pop:
                        parent_indices.append(np.random.choice(len(fronts[0])))
            else:
                parent_indices = list(range(n_pop))

            # Generate children
            n_children = min(n_pop, budget - used)
            if n_children <= 0:
                break

            # Progress: 0 at start, 1 at end
            progress = min(1.0, gen / max(1, n_generations))
            # Mutation sigma: start at 0.25, end at 0.03
            sigma = 0.25 - 0.22 * progress
            # Crossover rate
            cx_rate = 0.8

            children = np.zeros((n_children, dim))
            children_F = np.zeros((n_children, n_obj))

            for i in range(n_children):
                # Crossover: pick two parents
                if np.random.random() < cx_rate and len(parent_indices) >= 2:
                    p1_idx = parent_indices[np.random.randint(len(parent_indices))]
                    p2_idx = parent_indices[np.random.randint(len(parent_indices))]
                    while p2_idx == p1_idx and len(parent_indices) > 1:
                        p2_idx = parent_indices[np.random.randint(len(parent_indices))]
                    # Blend crossover
                    alpha = np.random.random()
                    child = alpha * pop[p1_idx] + (1 - alpha) * pop[p2_idx]
                else:
                    p1_idx = parent_indices[np.random.randint(len(parent_indices))]
                    child = pop[p1_idx].copy()

                # Mutation
                child = child + sigma * span * np.random.standard_normal(dim)
                child = np.clip(child, lo, hi)

                children[i] = child
                children_F[i] = problem(child)
                used += 1

                # Check deadline mid-batch
                if used >= budget or time.time() > deadline:
                    break

            if i + 1 < n_children:
                children = children[:i + 1]
                children_F = children_F[:i + 1]
                n_children = i + 1

            if n_children == 0:
                break

            # Combine parent and child populations
            combined_X = np.vstack([pop, children])
            combined_F = np.vstack([pop_F, children_F])

            # Non-dominated sort the combined population
            combined_fronts = self._non_dominated_sort(combined_F)

            # Select next population of size n_pop
            new_pop = np.zeros((n_pop, dim))
            new_pop_F = np.zeros((n_pop, n_obj))
            count = 0
            for front in combined_fronts:
                if count + len(front) <= n_pop:
                    for idx in front:
                        new_pop[count] = combined_X[idx]
                        new_pop_F[count] = combined_F[idx]
                        count += 1
                else:
                    # Need to pick from this front to fill
                    needed = n_pop - count
                    chosen = np.random.choice(len(front), size=needed, replace=False)
                    for j, idx in enumerate(chosen):
                        new_pop[count + j] = combined_X[idx]
                        new_pop_F[count + j] = combined_F[idx]
                    count = n_pop
                    break

            pop = new_pop
            pop_F = new_pop_F

            # Record all children in archive
            all_X = np.vstack([all_X, children])
            all_F = np.vstack([all_F, children_F])

            gen += 1

        return all_X

    def _non_dominated_sort(self, F):
        """
        Non-dominated sorting. Returns list of fronts, each front is a list of indices.
        """
        n = len(F)
        if n == 0:
            return []

        # domination[i] = list of indices that dominate i
        domination = [[] for _ in range(n)]
        # dominated[i] = count of points that dominate i
        dominated_count = np.zeros(n, dtype=int)

        for i in range(n):
            for j in range(i + 1, n):
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    domination[i].append(j)
                    dominated_count[j] += 1
                # Check if j dominates i
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    domination[j].append(i)
                    dominated_count[i] += 1

        fronts = []
        current_front = [i for i in range(n) if dominated_count[i] == 0]
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in domination[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        next_front.append(j)
            current_front = next_front

        return fronts
# EVOLVE-BLOCK-END
