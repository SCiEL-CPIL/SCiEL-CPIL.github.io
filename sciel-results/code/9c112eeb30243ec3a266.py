"""Multi-objective optimizer for RE34 with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based multi-objective optimizer with dominance and time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 325.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        # Initial population
        pop_size = min(50, self.budget)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        def check_time():
            return time.time() < deadline

        def compute_dominance(F):
            """Return boolean array: True if dominated by at least one other point."""
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                # Check if any other point dominates i
                # j dominates i if F[j] <= F[i] in all objectives and < in at least one
                diff = F - F[i]  # shape (n, n_obj)
                # j dominates i if diff[j] <= 0 for all objectives and diff[j] < 0 for at least one
                le = np.all(diff <= 0, axis=1)
                lt = np.any(diff < 0, axis=1)
                dom_mask = le & lt
                dom_mask[i] = False  # exclude self
                if np.any(dom_mask):
                    dominated[i] = True
            return dominated

        def non_dominated_sort(F):
            """Return list of non-dominated fronts (each is an array of indices)."""
            n = len(F)
            fronts = []
            remaining = set(range(n))
            while remaining:
                front = []
                for i in remaining:
                    # Check if i is dominated by any other in remaining
                    dominated = False
                    for j in remaining:
                        if i == j:
                            continue
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated = True
                            break
                    if not dominated:
                        front.append(i)
                if not front:
                    # Should not happen, but safety
                    front = list(remaining)
                fronts.append(np.array(front))
                for i in front:
                    remaining.discard(i)
            return fronts

        # Main loop
        while used < self.budget and check_time():
            # Compute non-dominated fronts
            fronts = non_dominated_sort(F)
            
            # Select parents from best fronts
            # Prefer points from first front, then second, etc.
            parent_pool = []
            for front in fronts:
                parent_pool.extend(front.tolist())
                if len(parent_pool) >= 10:
                    break
            
            if not parent_pool:
                parent_pool = list(range(len(F)))
            
            # Pick two parents from the pool
            if len(parent_pool) >= 2:
                p1 = parent_pool[rng.integers(len(parent_pool))]
                p2 = parent_pool[rng.integers(len(parent_pool))]
                # Simple crossover (blend)
                alpha = rng.random()
                child = alpha * X[p1] + (1 - alpha) * X[p2]
            else:
                p1 = parent_pool[0]
                child = X[p1].copy()
            
            # Mutation
            # Adaptive mutation rate
            progress = used / self.budget
            mut_rate = 0.3 * (1.0 - progress * 0.5)  # decrease slightly over time
            mask = rng.random(self.dim) < mut_rate
            child[mask] += 0.15 * (hi - lo)[mask] * rng.standard_normal(np.sum(mask))
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst point in population
            # Use a simple criterion: sum of normalized objectives
            # Or just replace the last point
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
