"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid MOEA: Single-step evolution with Pareto-based population maintenance."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 300.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: small enough to allow many generations, large enough for diversity
        # With budget 200, pop_size 10 allows ~20 generations. Pop_size 20 allows ~10.
        # Inspiration 1 used pop_size ~5 (budget//4) and scored 1.0.
        # Let's use pop_size = 10 to balance exploration and exploitation.
        pop_size = 10

        rng = np.random.default_rng(0)

        # Initialize population
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper: Non-dominated sort
        def non_dominated_sort(F_mat):
            n = len(F_mat)
            if n == 0:
                return [[]]
            
            S = [[] for _ in range(n)]
            n_doms = [0] * n
            ranks = [0] * n
            current_front = []

            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    # p dominates q?
                    if np.all(F_mat[p] <= F_mat[q]) and np.any(F_mat[p] < F_mat[q]):
                        S[p].append(q)
                    elif np.all(F_mat[q] <= F_mat[p]) and np.any(F_mat[q] < F_mat[p]):
                        n_doms[p] += 1

                if n_doms[p] == 0:
                    ranks[p] = 0
                    current_front.append(p)

            i = 0
            while current_front:
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n_doms[q] -= 1
                        if n_doms[q] == 0:
                            ranks[q] = i + 1
                            next_front.append(q)
                i += 1
                current_front = next_front

            fronts = [[] for _ in range(i)]
            for idx, r in enumerate(ranks):
                if r < i:
                    fronts[r].append(idx)
            return fronts

        # Helper: Crowding distance
        def crowding_distance(F_mat, indices):
            if not indices:
                return {}
            n = len(indices)
            if n <= 2:
                return {idx: float('inf') for idx in indices}

            dists = {idx: 0.0 for idx in indices}
            m = n_obj
            for k in range(m):
                sorted_idx = sorted(indices, key=lambda i: F_mat[i][k])
                dists[sorted_idx[0]] = float('inf')
                dists[sorted_idx[-1]] = float('inf')
                f_min = F_mat[sorted_idx[0]][k]
                f_max = F_mat[sorted_idx[-1]][k]
                if f_max == f_min:
                    continue
                for j in range(1, n - 1):
                    idx = sorted_idx[j]
                    dists[idx] += (F_mat[sorted_idx[j + 1]][k] - F_mat[sorted_idx[j - 1]][k]) / (f_max - f_min)
            return dists

        # Main loop: Single-step evolution
        while used < budget:
            if time.time() > deadline:
                break

            # Identify non-dominated individuals in current population
            # For small pop_size, O(N^2) is fine
            n = len(X)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[j] = True
                        break
            
            non_dominated_indices = np.where(~dominated)[0]
            
            # Select parent from non-dominated set if available, else random
            if len(non_dominated_indices) > 0:
                parent_idx = non_dominated_indices[rng.integers(len(non_dominated_indices))]
            else:
                parent_idx = rng.integers(n)
            
            parent = X[parent_idx]
            
            # Mutation: Gaussian perturbation
            # Adaptive step size? Start with 0.1 * span, maybe reduce slightly over time?
            # Keep it simple: constant 0.1 * span worked in Inspiration 1.
            mutation_scale = 0.1 * span
            child = parent + mutation_scale * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)
            
            # Evaluate child
            f_child = problem(child)
            used += 1
            
            # Add child to population
            X_new = np.vstack([X, child])
            F_new = np.vstack([F, f_child])
            
            # Select best pop_size individuals from X_new, F_new
            fronts = non_dominated_sort(F_new)
            
            X_selected = []
            F_selected = []
            
            for front in fronts:
                if len(X_selected) + len(front) <= pop_size:
                    for idx in front:
                        X_selected.append(X_new[idx])
                        F_selected.append(F_new[idx])
                else:
                    # Need to prune this front
                    cd = crowding_distance(F_new, front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    needed = pop_size - len(X_selected)
                    for idx in sorted_front[:needed]:
                        X_selected.append(X_new[idx])
                        F_selected.append(F_new[idx])
                    break
            
            X = np.array(X_selected)
            F = np.array(F_selected)

        return F
# EVOLVE-BLOCK-END
