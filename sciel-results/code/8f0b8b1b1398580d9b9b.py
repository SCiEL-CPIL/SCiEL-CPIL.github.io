"""Multi-objective optimizer on WFG7: Simple mutation of non-dominated parents."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Simple evolutionary multi-objective search: random init, mutate non-dominated parents."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Small population to allow many generations within budget
        pop_size = min(20, self.budget // 4)
        if pop_size < 5:
            pop_size = 5

        # Initial population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Find non-dominated points in the current population
            # F is (pop_size, n_obj)
            # A point i is dominated if there exists j such that F[j] <= F[i] and F[j] != F[i]
            # Vectorized dominance check
            # diff[i, j] = F[j] - F[i]
            # We want to know for each i, if any j dominates it.
            
            # Efficient way:
            # F_j <= F_i for all objectives AND F_j < F_i for at least one objective
            # Let's just loop for simplicity and speed in Python if pop_size is small (20)
            # 20*20 = 400 checks per iteration. 200 iterations = 80,000 checks. Very fast.
            
            is_dominated = np.zeros(pop_size, dtype=bool)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dominated[i] = True
                        break

            # Select parents from non-dominated set
            non_dom_indices = np.where(~is_dominated)[0]
            if len(non_dom_indices) == 0:
                # Fallback if all are dominated (shouldn't happen with elitism-like replacement, but safe)
                parents_idx = np.arange(pop_size)
            else:
                parents_idx = non_dom_indices

            # Pick a random parent
            p_idx = parents_idx[rng.integers(len(parents_idx))]
            
            # Mutate
            # Use a step size that allows exploration but not too much
            # 0.1 * span is a reasonable default
            child = X[p_idx] + 0.1 * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            # Evaluate
            fc = problem(child)
            used += 1
            
            # Replace the worst point in the population
            # Heuristic: Sum of objectives (scalarization)
            # This is a simple proxy for "worst" that encourages diversity to some extent
            # by not letting any single objective get too large.
            scores = np.sum(F, axis=1)
            worst_idx = int(np.argmax(scores))
            
            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
