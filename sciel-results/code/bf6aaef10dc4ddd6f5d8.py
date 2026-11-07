"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Lightweight multi-objective evolutionary optimizer.
    Uses simple non-dominated sorting and greedy replacement.
    Optimized for speed to maximize evaluations within the time budget.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: The evaluator kills at 330s, but we should aim for ~290s 
        # to ensure we return cleanly.
        deadline = start_time + 290.0 

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size: Keep it small to minimize sorting/selection overhead
        # The inspiration that scored 1.0 used min(20, budget // 4)
        pop_size = int(min(20, self.budget // 4))
        if pop_size < 2:
            pop_size = 2

        # Use a fixed seed for reproducibility as per task description implications
        # The task says "For seeds 0 and 1 the evaluator calls np.random.seed(seed)"
        # But inside the class, using a local RNG is safer to avoid side effects 
        # if the evaluator relies on global state for other things, though 
        # typically we just use the global state or a local one. 
        # The inspirations used np.random.default_rng(0). 
        # Let's stick to local RNG for isolation.
        rng = np.random.default_rng(0)

        # Initialize population uniformly at random
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Pre-allocate arrays for speed if possible, but lists are fine for small pop
        while used < self.budget:
            # Time check
            if time.time() > deadline:
                break
            
            # Identify non-dominated individuals
            # A point p is dominated by q if q <= p and q != p
            # We want the set of points NOT dominated by any other point in F.
            
            # Efficient check: 
            # For each i, check if there exists j such that F[j] <= F[i] and F[j] != F[i]
            # Vectorized approach for small pop_size:
            # D[i, j] is True if j dominates i
            # F[j] <= F[i] for all objectives AND F[j] < F[i] for at least one objective
            
            # Let's use a simple loop for clarity and speed with small N, 
            # or vectorized numpy if N is larger.
            # Given N=20, a loop is fast enough.
            
            is_dominated = np.zeros(pop_size, dtype=bool)
            for i in range(pop_size):
                if is_dominated[i]:
                    continue
                for j in range(pop_size):
                    if i == j:
                        continue
                    # Check if j dominates i
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dominated[i] = True
                        break

            # Get indices of non-dominated individuals
            non_dom_indices = np.where(~is_dominated)[0]
            
            if len(non_dom_indices) == 0:
                # Fallback: pick random parents if all dominated (shouldn't happen often)
                parents_idx = np.arange(pop_size)
            else:
                parents_idx = non_dom_indices

            # Select a parent randomly from the non-dominated set
            parent_idx = parents_idx[rng.integers(len(parents_idx))]
            parent = X[parent_idx]

            # Gaussian Mutation
            # The 1.0 inspiration used 0.1 * (hi - lo) * rng.standard_normal(self.dim)
            # This is a good balance.
            mutation_scale = 0.1
            child = parent + mutation_scale * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            # Evaluate child
            fc = problem(child)
            used += 1

            # Greedy Replacement: Replace the worst individual
            # "Worst" is defined by the sum of objective values (simple scalarization)
            # This is fast and effective for maintaining diversity in small populations.
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            
            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
