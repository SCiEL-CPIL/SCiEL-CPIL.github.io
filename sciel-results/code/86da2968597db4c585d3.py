import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0
        time_budget = 325.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        rng = np.random

        N = 30
        F = 0.7
        CR = 0.9
        used = 0

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def get_non_dominated_indices(F_set):
            K = len(F_set)
            if K == 0:
                return []
            nd_indices = []
            for i in range(K):
                is_dom = False
                for j in range(K):
                    if i != j and dominates(F_set[j], F_set[i]):
                        is_dom = True
                        break
                if not is_dom:
                    nd_indices.append(i)
            return nd_indices

        def hypervolume_2d(points, ref):
            if not points:
                return 0.0
            valid_points = [p for p in points if p[0] < ref[0] and p[1] < ref[1]]
            if not valid_points:
                return 0.0
            sorted_pts = sorted(valid_points, key=lambda p: p[0])
            hv = 0.0
            prev_f2 = ref[1]
            for p in sorted_pts:
                f1, f2 = p
                if f2 < prev_f2:
                    hv += (ref[0] - f1) * (prev_f2 - f2)
                    prev_f2 = f2
            return hv

        def get_non_dominated_points(points):
            non_dominated = []
            for i, p in enumerate(points):
                is_dominated = False
                for j, q in enumerate(points):
                    if i != j and dominates(q, p):
                        is_dominated = True
                        break
                if not is_dominated:
                    non_dominated.append(p)
            return non_dominated

        # Latin Hypercube initialization
        n_init = min(N, budget)
        X = np.zeros((n_init, dim))
        for d in range(dim):
            perm = rng.permutation(n_init)
            fractions = (perm + rng.random(n_init)) / n_init
            X[:, d] = lo[d] + fractions * (hi[d] - lo[d])

        F_vals = np.array([problem(x) for x in X])
        used = n_init

        # Main DE loop
        gen = 0
        while used < budget:
            if time.time() > deadline:
                break
            gen += 1
            num_offspring = min(N, budget - used)
            U = np.zeros((num_offspring, dim))
            U_vals = np.zeros((num_offspring, n_obj))

            for i in range(num_offspring):
                if time.time() > deadline:
                    break
                r1, r2, r3 = rng.choice(N, 3, replace=False)
                v = X[r1] + F * (X[r2] - X[r3])
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() <= CR or j == j_rand:
                        U[i, j] = v[j]
                    else:
                        U[i, j] = X[i, j]
                U[i] = np.clip(U[i], lo, hi)
                U_vals[i] = problem(U[i])
                used += 1

            if used >= budget or time.time() > deadline:
                break

            F_combined = np.vstack([F_vals, U_vals[:num_offspring]])
            X_combined = np.vstack([X, U[:num_offspring]])

            n_total = len(F_combined)
            non_dom_indices = get_non_dominated_indices(F_combined)

            if len(non_dom_indices) > 0:
                ref_point = np.max(F_combined[non_dom_indices], axis=0) + 1.0
            else:
                ref_point = np.max(F_combined, axis=0) + 1.0

            selected_indices = []
            remaining_indices = non_dom_indices[:]
            for _ in range(min(N, len(remaining_indices))):
                if not remaining_indices:
                    break
                if time.time() > deadline:
                    break
                best_idx = -1
                best_hv = -1.0
                current_selected_pts = [F_combined[i] for i in selected_indices]
                for idx in remaining_indices:
                    candidate_pts = current_selected_pts + [F_combined[idx]]
                    nd_pts = get_non_dominated_points(candidate_pts)
                    hv_val = hypervolume_2d(nd_pts, ref_point)
                    if hv_val > best_hv:
                        best_hv = hv_val
                        best_idx = idx
                if best_idx != -1:
                    selected_indices.append(best_idx)
                    remaining_indices.remove(best_idx)

            if len(selected_indices) < N:
                selected_set = set(selected_indices)
                for i in range(n_total):
                    if len(selected_indices) >= N:
                        break
                    if i not in selected_set:
                        selected_indices.append(i)
                        selected_set.add(i)
                while len(selected_indices) < N:
                    selected_indices.append(selected_indices[-1])

            X = X_combined[selected_indices]
            F_vals = F_combined[selected_indices]

        # Polish phase with Nelder-Mead local search
        if used < budget and time.time() < deadline - 5:
            nd_idx = get_non_dominated_indices(F_vals)
            if len(nd_idx) > 0:
                top_k = min(8, len(nd_idx))
                polish_indices = nd_idx[:top_k]
                remaining = budget - used
                
                if remaining > 20:
                    # Define a scalar objective for Nelder-Mead using weighted sum
                    # with weights based on ideal/nadir normalization to balance objectives
                    f_min = np.min(F_vals, axis=0)
                    f_max = np.max(F_vals, axis=0)
                    f_range = f_max - f_min
                    f_range[f_range == 0] = 1.0
                    weights = 1.0 / f_range
                    
                    def objective_weighted(x):
                        nonlocal used
                        if used >= budget:
                            return 1e10
                        x_clip = np.clip(x, lo, hi)
                        f = problem(x_clip)
                        used += 1
                        # Normalize and weight to balance objectives
                        f_norm = (f - f_min) / f_range
                        return np.dot(weights, f_norm)

                    # Estimate max iterations for Nelder-Mead
                    # Each iteration uses up to dim+1 function calls
                    # We want to use most of the remaining budget
                    # Split budget among top_k points
                    budget_per_nm = remaining // top_k
                    max_iter = max(15, budget_per_nm // (dim + 1))
                    
                    for idx in polish_indices:
                        if used >= budget or time.time() > deadline:
                            break
                        
                        x0 = X[idx].copy()
                        
                        try:
                            res = minimize(objective_weighted, x0, method='Nelder-Mead', 
                                           options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10,
                                                    'adaptive': True})
                            # The calls to problem inside minimize have already updated 'used'
                            # and been recorded by the evaluator.
                        except Exception:
                            pass

        return X
