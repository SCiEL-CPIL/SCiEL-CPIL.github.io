"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE21 with adaptive mutation and origin bias."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        budget = self.budget

        # Use a fixed seed for the internal RNG to ensure reproducibility
        # regardless of the global np.random.seed(seed) call, as the problem is deterministic.
        # If the problem had stochasticity, we might want to use global rng, but RE21 is fixed.
        rng = np.random.default_rng(0)

        # Initial population size
        pop_size = 20
        
        # Mixed initialization: 50% near origin (Rastrigin-like min), 50% random
        X = np.zeros((pop_size, dim))
        n_origin = pop_size // 2
        # Points near origin with small variance
        X[:n_origin] = 0.05 * span * rng.standard_normal((n_origin, dim))
        # Points uniformly random
        X[n_origin:] = lo + span * rng.random((pop_size - n_origin, dim))
        
        # Clip to bounds just in case
        X = np.clip(X, lo, hi)

        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Adaptive mutation parameters
        base_scale = 0.3
        min_scale = 0.02

        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Calculate progress for adaptive scaling
            progress = used / budget
            current_scale = base_scale * (1.0 - progress) + min_scale

            # Find non-dominated points in current population
            # F is (pop_size, n_obj)
            # A point i is dominated if exists j such that F[j] <= F[i] and F[j] < F[i]
            dominated = np.zeros(pop_size, dtype=bool)
            for i in range(pop_size):
                # Vectorized domination check:
                # F <= F[i] is (pop_size, n_obj)
                # Any F[j] < F[i] is (pop_size,)
                # We need to exclude i itself
                mask_le = np.all(F <= F[i], axis=1)
                mask_lt = np.any(F < F[i], axis=1)
                mask_le[i] = False # Exclude self
                if np.any(mask_le & mask_lt):
                    dominated[i] = True
                    break

            # Select parent from non-dominated set
            nd_indices = np.where(~dominated)[0]
            if len(nd_indices) == 0:
                # Fallback to random if all dominated (should not happen often)
                parent_idx = rng.integers(pop_size)
            else:
                parent_idx = nd_indices[rng.integers(len(nd_indices))]
            
            parent = X[parent_idx]

            # Mutate
            mutation = current_scale * span * rng.standard_normal(dim)
            child = parent + mutation
            child = np.clip(child, lo, hi)

            # Evaluate
            fc = problem(child)
            used += 1

            # Replace the worst point in the population
            # "Worst" defined by the sum of objectives (simple scalarization)
            # This helps maintain a spread along the Pareto front
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            
            # Only replace if the child is actually better in some sense?
            # Or just always replace to maintain diversity?
            # Inspiration 1 always replaced. Let's stick to that for simplicity and speed.
            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
