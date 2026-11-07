"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE/rand/1 with adaptive mutation and worst-replacement for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Population size
        pop_size = min(50, max(20, self.budget // 5))
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            """Check if a dominates b (a is better than b in all objectives, strictly better in at least one)"""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_front(F):
            """Return indices of the first non-dominated front"""
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(i + 1, n):
                    if dominated[j]:
                        continue
                    if dominates(F[i], F[j]):
                        dominated[j] = True
                    elif dominates(F[j], F[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of non-dominated solutions"""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            cd = np.zeros(n)
            for m in range(self.n_obj):
                obj = F[indices, m]
                sorted_idx = np.argsort(obj)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                obj_range = obj[sorted_idx[-1]] - obj[sorted_idx[0]]
                if obj_range > 0:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (obj[sorted_idx[i+1]] - obj[sorted_idx[i-1]]) / obj_range
            return cd

        # Main loop
        while used < self.budget:
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Find non-dominated front
            front_indices = non_dominated_front(F)
            front_size = len(front_indices)
            
            # Compute crowding distance for the front
            cd = crowding_distance(F, front_indices)
            
            # Select parents from the front based on crowding distance
            if front_size >= 3:
                # Pick 3 parents with highest crowding distance
                sorted_front = np.argsort(cd)[::-1]
                p1_idx = front_indices[sorted_front[rng.integers(0, min(3, front_size))]]
                p2_idx = front_indices[sorted_front[rng.integers(0, min(3, front_size))]]
                p3_idx = front_indices[sorted_front[rng.integers(0, min(3, front_size))]]
                while p2_idx == p1_idx:
                    p2_idx = front_indices[sorted_front[rng.integers(0, min(3, front_size))]]
                while p3_idx in (p1_idx, p2_idx):
                    p3_idx = front_indices[sorted_front[rng.integers(0, min(3, front_size))]]
            elif front_size == 2:
                p1_idx, p2_idx = front_indices
                p3_idx = front_indices[rng.integers(0, 2)]
            else:
                p1_idx = front_indices[0]
                p2_idx = front_indices[rng.integers(0, pop_size)]
                p3_idx = front_indices[rng.integers(0, pop_size)]

            # DE/rand/1 mutation
            F1 = 0.5 + 0.5 * rng.random()  # random F factor
            child = X[p1_idx] + F1 * (X[p2_idx] - X[p3_idx])
            child = np.clip(child, lo, hi)

            # Adaptive Gaussian perturbation
            sigma = 0.1 * (1.0 - 0.5 * used / self.budget)
            child = child + sigma * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace the worst solution (highest sum of objectives)
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst] = child
            F[worst] = fc

        return X
# EVOLVE-BLOCK-END
