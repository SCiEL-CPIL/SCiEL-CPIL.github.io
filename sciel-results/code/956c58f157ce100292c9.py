"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using a combination of:
    - Random initialization with Latin Hypercube sampling
    - Adaptive mutation (Cauchy + Gaussian)
    - Elitism-based replacement
    - Time-bounded search with deadline guard
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Use the seeded RNG from the evaluator
        rng = np.random

        # Initial population: use Latin Hypercube for better coverage
        pop_size = min(40, budget // 5)
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = rng.permutation(pop_size)
            X[:, d] = (perm + 0.5) / pop_size
        X = lo + (hi - lo) * X

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track best hypervolume-contributing points
        all_X = [X.copy()]
        all_F = [F.copy()]

        def non_dominated_indices(F):
            """Return indices of non-dominated points."""
            n = len(F)
            if n == 0:
                return np.array([], dtype=int)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j or not nd[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] = False
                        break
            return np.where(nd)[0]

        def dominated_by_any(x, F):
            """Check if x is dominated by any point in F."""
            for f in F:
                if np.all(f <= x) and np.any(f < x):
                    return True
            return False

        # Adaptive mutation strategy
        mutation_scale = 0.15  # initial scale

        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Get non-dominated set
            nd_idx = non_dominated_indices(F)
            if len(nd_idx) == 0:
                nd_idx = np.array([np.argmin(np.sum(F, axis=1))])

            # Select parent: prefer non-dominated, with some randomness
            if len(nd_idx) > 0 and rng.random() < 0.8:
                parent_idx = nd_idx[rng.integers(len(nd_idx))]
            else:
                parent_idx = rng.integers(len(F))

            parent = X[parent_idx]

            # Adaptive mutation: mix of Gaussian and Cauchy-like perturbations
            # Scale decreases slowly as we use more budget
            progress = used / budget
            scale = mutation_scale * (1.0 - 0.5 * progress)

            # Use a mix: 70% Gaussian, 30% larger jumps
            if rng.random() < 0.7:
                noise = rng.standard_normal(dim)
            else:
                # Cauchy-like: use ratio of two normals for heavy tails
                noise = rng.standard_normal(dim) / (0.3 + 0.7 * np.abs(rng.standard_normal(dim)))
                noise = np.clip(noise, -3, 3)  # prevent extreme values

            child = parent + scale * (hi - lo) * noise
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Elitist replacement: replace the worst point if child is not dominated
            # Find the worst point (most dominated or worst sum)
            if not dominated_by_any(fc, F):
                # Child is non-dominated; replace the point with worst sum
                worst_idx = np.argmax(np.sum(F, axis=1))
                X[worst_idx] = child
                F[worst_idx] = fc
            else:
                # Child is dominated; still add it if it's better than the worst
                child_sum = np.sum(fc)
                worst_sum = np.sum(F[np.argmax(np.sum(F, axis=1))])
                if child_sum < worst_sum:
                    worst_idx = np.argmax(np.sum(F, axis=1))
                    X[worst_idx] = child
                    F[worst_idx] = fc

        # Collect all unique points evaluated (for hypervolume, all recorded points count)
        # The evaluator records every objective vector returned by problem()
        # So we need to make sure we've used all budget evaluations
        # Return doesn't matter, but we must not exceed budget

        return X
# EVOLVE-BLOCK-END
