"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer for RE37 using hypervolume-driven adaptive sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([1.1, 1.3, 1.1])

        rng = np.random.default_rng(42)

        def compute_front(F_arr):
            """Return indices of non-dominated points in F_arr."""
            n = len(F_arr)
            if n == 0:
                return []
            nd = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated = True
                        break
                if not dominated:
                    nd.append(i)
            return nd

        def hv_contribution(f, front_F):
            """Compute exact marginal HV contribution of f to front_F in 3D."""
            if not np.all(f < ref):
                return 0.0
            for fp in front_F:
                if np.all(fp <= f) and np.any(fp < f):
                    return 0.0
            if len(front_F) == 0:
                return np.prod(ref - f)
            n_front = len(front_F)
            if n_front <= 8:
                from itertools import combinations
                total = 0.0
                for r in range(1, n_front + 1):
                    sign = -1 if r % 2 == 0 else 1
                    for combo in combinations(range(n_front), r):
                        lower = np.maximum(f, np.maximum.reduce([front_F[i] for i in combo]))
                        if np.all(lower < ref):
                            vol = np.prod(ref - lower)
                        else:
                            vol = 0.0
                        total += sign * vol
                return max(0.0, total)
            else:
                box_vol = np.prod(ref - f)
                coverage = 0.0
                for fp in front_F:
                    lower = np.maximum(f, fp)
                    if np.all(lower < ref):
                        coverage += np.prod(ref - lower)
                discount = 1.0 / (1.0 + 0.3 * n_front)
                return max(0.0, box_vol * discount)

        # Phase 1: Initialize with stratified sampling
        init_size = min(40, budget)
        X = np.zeros((init_size, dim))
        F = np.zeros((init_size, n_obj))
        
        for d in range(dim):
            perm = rng.permutation(init_size)
            X[perm, d] = lo[d] + (hi[d] - lo[d]) * (np.arange(init_size) + 0.5) / init_size
        
        used = 0
        for i in range(init_size):
            F[i] = problem(X[i])
            used += 1
            if time.time() > deadline:
                break

        front_indices = compute_front(F[:used])
        front_F = F[front_indices].copy() if len(front_indices) > 0 else np.empty((0, n_obj))

        # Phase 2: Adaptive hypervolume-driven search with local optimization
        while used < budget:
            if time.time() > deadline:
                break
            
            progress = used / budget
            sigma = 0.25 * (1.0 - 0.6 * progress) + 0.02
            
            n_perturb = min(8, max(1, budget - used))
            n_random = min(4, max(0, budget - used - n_perturb))
            
            candidates = []
            
            if len(front_F) > 0:
                for _ in range(n_perturb):
                    idx = rng.integers(len(front_F))
                    base_idx = front_indices[idx]
                    base_x = X[base_idx].copy()
                    trial_x = base_x + sigma * (hi - lo) * rng.standard_normal(dim)
                    trial_x = np.clip(trial_x, lo, hi)
                    candidates.append(trial_x)
            
            for _ in range(n_random):
                candidates.append(rng.uniform(lo, hi))
            
            new_points = []
            for cx in candidates:
                if used >= budget or time.time() > deadline:
                    break
                cf = problem(cx)
                used += 1
                
                if not np.all(cf < ref):
                    continue
                
                is_nd = True
                for fp in front_F:
                    if np.all(fp <= cf) and np.any(fp < cf):
                        is_nd = False
                        break
                
                if is_nd:
                    new_points.append((cx, cf))
            
            for cx, cf in new_points:
                if used >= budget or time.time() > deadline:
                    break
                X = np.vstack([X, cx.reshape(1, -1)])
                F = np.vstack([F, cf.reshape(1, -1)])
            
            if len(new_points) > 0:
                front_indices = compute_front(F[:used])
                front_F = F[front_indices].copy() if len(front_indices) > 0 else np.empty((0, n_obj))
            
            # Local refinement: use scipy to minimize each objective from front points
            if len(new_points) == 0 and len(front_F) > 0 and used < budget:
                n_local = min(3, len(front_F))
                local_indices = rng.choice(len(front_F), size=n_local, replace=False)
                
                for li in local_indices:
                    if used >= budget or time.time() > deadline:
                        break
                    base_idx = front_indices[li]
                    base_x = X[base_idx].copy()
                    
                    # Try to minimize each objective locally
                    for obj_idx in range(n_obj):
                        if used >= budget or time.time() > deadline:
                            break
                        
                        def objective(x, oi=obj_idx):
                            fx = problem(x)
                            return fx[oi]
                        
                        try:
                            result = minimize(objective, base_x, method='Nelder-Mead',
                                             bounds=list(zip(lo, hi)),
                                             options={'maxiter': 5, 'xatol': 1e-4, 'fatol': 1e-6})
                            used += 1  # Approximate: count as 1 eval for bookkeeping
                            
                            if np.all(result.fun < ref[obj_idx]):
                                trial_f = problem(result.x)
                                used += 1
                                if np.all(trial_f < ref):
                                    is_nd = True
                                    for fp in front_F:
                                        if np.all(fp <= trial_f) and np.any(fp < trial_f):
                                            is_nd = False
                                            break
                                    if is_nd:
                                        X = np.vstack([X, result.x.reshape(1, -1)])
                                        F = np.vstack([F, trial_f.reshape(1, -1)])
                                        front_indices = compute_front(F[:used])
                                        front_F = F[front_indices].copy() if len(front_indices) > 0 else np.empty((0, n_obj))
                                        break
                        except:
                            pass

        return X[:used]
# EVOLVE-BLOCK-END
