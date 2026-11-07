"""Multi-objective optimizer on RE37 using Differential Evolution and Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """DE/rand/1/bin with NSGA-II selection and Nelder-Mead polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        max_evals = self.budget

        used = 0
        archive = []  # List of (x, f) tuples for non-dominated points

        def eval_point(x):
            nonlocal used
            if used >= max_evals:
                return None
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            return x, f

        def fast_non_dominated_sort(F):
            """
            F: NxM array of objective values.
            Returns: List of fronts, where each front is a list of indices.
            """
            N = F.shape[0]
            S = [[] for _ in range(N)]
            n = [0] * N
            rank = [0] * N
            fronts = []
            front0 = []
            
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n[j] += 1
                    # Check if j dominates i
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n[i] += 1
                if n[i] == 0:
                    rank[i] = 0
                    front0.append(i)
            
            k = 0
            while front0:
                fronts.append(front0)
                next_front = []
                for i in front0:
                    for j in S[i]:
                        n[j] -= 1
                        if n[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                front0 = next_front
                k += 1
            return fronts

        def crowding_distance(F, indices):
            """
            F: NxM array.
            indices: List of indices belonging to the same front.
            Returns: Dictionary mapping index to crowding distance.
            """
            M = F.shape[1]
            N = len(indices)
            if N <= 2:
                return {i: float('inf') for i in indices}
            
            cd = {i: 0.0 for i in indices}
            for m in range(M):
                sorted_idx = sorted(indices, key=lambda i: F[i, m])
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                
                f_min = F[sorted_idx[0], m]
                f_max = F[sorted_idx[-1], m]
                f_range = f_max - f_min
                if f_range == 0:
                    continue
                
                for k in range(1, N - 1):
                    i = sorted_idx[k]
                    cd[i] += (F[sorted_idx[k+1], m] - F[sorted_idx[k-1], m]) / f_range
            return cd

        # Phase 1: Initialization
        pop_size = 30
        if max_evals < pop_size:
            pop_size = max_evals
            
        X = []
        F = []
        for _ in range(pop_size):
            x = lo + span * np.random.rand(dim)
            res = eval_point(x)
            if res is None:
                break
            x, f = res
            X.append(x)
            F.append(f)
            
        X = np.array(X)
        F = np.array(F)
        
        # Phase 2: Differential Evolution
        F_diff = 0.7
        CR = 0.9
        n_generations = 4 # 4 * 30 = 120 evals
        
        for gen in range(n_generations):
            if used >= max_evals or time.time() > deadline:
                break
                
            X_new = []
            F_new = []
            
            for i in range(len(X)):
                # Select 3 distinct parents
                if len(X) < 3:
                    # Fallback to random mutation if population is too small
                    x_r = lo + span * np.random.rand(dim)
                else:
                    idxs = np.random.choice(len(X), 3, replace=False)
                    x1, x2, x3 = X[idxs[0]], X[idxs[1]], X[idxs[2]]
                    mutant = x1 + F_diff * (x2 - x3)
                    
                    # Crossover
                    x_new = mutant.copy()
                    for j in range(dim):
                        if np.random.rand() < CR or j == np.random.randint(dim):
                            x_new[j] = X[i, j]
                    x_r = x_new
                
                res = eval_point(x_r)
                if res is None:
                    break
                x, f = res
                X_new.append(x)
                F_new.append(f)
                
            if not X_new:
                break
                
            X_new = np.array(X_new)
            F_new = np.array(F_new)
            
            # Combine and Select
            X_comb = np.vstack([X, X_new])
            F_comb = np.vstack([F, F_new])
            
            fronts = fast_non_dominated_sort(F_comb)
            
            X_next = []
            F_next = []
            for front in fronts:
                if len(X_next) + len(front) <= pop_size:
                    for idx in front:
                        X_next.append(X_comb[idx])
                        F_next.append(F_comb[idx])
                else:
                    # Need to truncate this front
                    cd = crowding_distance(F_comb, front)
                    # Sort front by crowding distance descending
                    front_sorted = sorted(front, key=lambda idx: cd[idx], reverse=True)
                    needed = pop_size - len(X_next)
                    for idx in front_sorted[:needed]:
                        X_next.append(X_comb[idx])
                        F_next.append(F_comb[idx])
                    break
                
            if len(X_next) == 0:
                break
                
            X = np.array(X_next)
            F = np.array(F_next)
            
        # Phase 3: Polish
        # Use remaining budget to polish the best points
        # Select up to 3 points from the final non-dominated front
        fronts = fast_non_dominated_sort(F)
        if fronts:
            front0 = fronts[0]
            # Pick 3 diverse points from the first front
            # Use crowding distance to pick diverse ones? Or just random?
            # Let's pick the 3 with highest crowding distance from front0
            cd = crowding_distance(F, front0)
            front0_sorted = sorted(front0, key=lambda idx: cd[idx], reverse=True)
            polish_indices = front0_sorted[:3]
        else:
            polish_indices = []

        remaining_budget = max_evals - used
        if remaining_budget > 0 and polish_indices:
            n_polish = min(3, len(polish_indices))
            budget_per_polish = remaining_budget // n_polish
            
            for idx in polish_indices[:n_polish]:
                if used >= max_evals or time.time() > deadline:
                    break
                
                x0 = X[idx]
                
                def obj(x):
                    nonlocal used
                    if used >= max_evals:
                        return 1e9
                    f = np.asarray(problem(x), dtype=float)
                    used += 1
                    return float(np.sum(f))
                
                try:
                    res = minimize(obj, x0, method='Nelder-Mead',
                                   options={'maxiter': budget_per_polish, 'xatol': 1e-6, 'fatol': 1e-9,
                                            'adaptive': True})
                    x_best = np.clip(res.x, lo, hi)
                    f_best = np.asarray(problem(x_best), dtype=float)
                    used += 1
                    # Add to archive if non-dominated
                    # We don't maintain a strict archive in the main loop, 
                    # but we can add to a final list
                    # For simplicity, we just record it. The return value is ignored 
                    # but "Every returned objective vector is recorded".
                    # Wait, the contract says "Every returned objective vector is recorded".
                    # This usually means the evaluator records what `problem(x)` returns 
                    # during the call. So we don't need to return anything specific.
                    # But we should ensure we don't exceed budget.
                except Exception:
                    pass

        # The return value is ignored by the evaluator, but we return the archive X
        return X
# EVOLVE-BLOCK-END
