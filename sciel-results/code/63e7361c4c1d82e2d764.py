"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer combining random initialization with
    evolutionary refinement via non-dominated parent selection and
    adaptive mutation.
    """
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use global np.random which is seeded by the evaluator
        # This ensures different behavior for seed 0 vs seed 1

        # Phase 1: Initial random population for broad coverage
        init_pop = min(40, budget)
        X_init = lo + (hi - lo) * np.random.random((init_pop, dim))
        F_init = np.array([problem(x) for x in X_init])
        evals_used = init_pop

        # Store all evaluated points
        all_X = [x.copy() for x in X_init]
        all_F = [f.copy() for f in F_init]

        # Helper: compute non-dominated front indices
        def get_non_dominated_indices(F_list):
            n = len(F_list)
            if n == 0:
                return []
            F_arr = np.array(F_list)
            nd_indices = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_indices.append(i)
            return nd_indices

        # Phase 2: Evolutionary refinement
        # Adaptive mutation: start large, shrink over remaining budget
        remaining = budget - evals_used
        mutation_scale_start = 0.3  # Fraction of range
        mutation_scale_end = 0.02

        while evals_used < budget:
            if time.time() > deadline:
                break

            # Get non-dominated front from all evaluated points
            nd_idx = get_non_dominated_indices(all_F)

            # Select parent: prefer non-dominated points, with some randomness
            if nd_idx and len(nd_idx) > 0:
                # Pick a random non-dominated point as parent
                parent_local = nd_idx[np.random.randint(len(nd_idx))]
            else:
                # Fallback: pick random point
                parent_local = np.random.randint(len(all_X))

            parent_X = all_X[parent_local]
            parent_F = all_F[parent_local]

            # Adaptive mutation step size
            progress = 1.0 - (budget - evals_used) / max(1, remaining)
            scale = mutation_scale_start + (mutation_scale_end - mutation_scale_start) * progress
            scale = max(scale, mutation_scale_end)

            # Gaussian mutation
            mutation = scale * (hi - lo) * np.random.standard_normal(dim)
            child_X = np.clip(parent_X + mutation, lo, hi)

            # Evaluate child
            child_F = problem(child_X)
            evals_used += 1

            # Add to archive
            all_X.append(child_X)
            all_F.append(child_F)

        return None
# EVOLVE-BLOCK-END
