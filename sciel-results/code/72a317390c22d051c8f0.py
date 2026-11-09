"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Augmented Epsilon-Constraint Method for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295  # 5s safety margin
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        # Phase 1: Latin Hypercube Sampling to get initial spread
        pop_size = min(20, budget // 4)
        rng = np.random.default_rng(42)
        
        def lhs_sample(n, d):
            points = np.zeros((n, d))
            for j in range(d):
                perm = rng.permutation(n)
                points[:, j] = (perm + rng.random(n)) / n
            return points

        X_lhs = lo + span * lhs_sample(pop_size, dim)
        F_lhs = np.array([problem(x) for x in X_lhs])
        
        all_X = [X_lhs]
        all_F = [F_lhs]
        used = pop_size

        # Determine range of second objective from LHS
        f2_min = np.min(F_lhs[:, 1])
        f2_max = np.max(F_lhs[:, 1])
        
        # Phase 2: Epsilon-Constraint Method
        # We optimize f1 subject to f2 <= epsilon
        # We will sweep epsilon from f2_max down to f2_min
        # Number of epsilon levels based on remaining budget
        # Each minimize call uses ~10-20 evaluations. Let's assume ~15.
        # Remaining budget = budget - used
        # Let's try to do about 10-15 epsilon levels
        
        remaining_budget = budget - used
        # Reserve some for final polish if needed, but mainly use for EC
        # Let's aim for 10-15 levels
        n_levels = min(15, max(5, remaining_budget // 12))
        
        # Epsilon values: from f2_max down to f2_min
        # Use log-spacing or linear? Linear is safer for coverage.
        epsilons = np.linspace(f2_max, f2_min, n_levels)
        
        # For each epsilon, minimize f1 subject to f2 <= epsilon
        # We can use penalty method or constraint. L-BFGS-B doesn't support constraints directly.
        # So we use penalty: minimize f1 + lambda * max(0, f2 - epsilon)^2
        
        lambda_penalty = 100.0
        
        for eps in epsilons:
            if not time_ok() or used >= budget:
                break
            
            def penalized_obj(x):
                fx = problem(x)
                f1, f2 = fx[0], fx[1]
                penalty = 0.0
                if f2 > eps:
                    penalty = lambda_penalty * (f2 - eps) ** 2
                return f1 + penalty
            
            # Start from a good point: maybe the LHS point closest to this epsilon?
            # Or just random/LHS
            # Let's pick the LHS point with f2 closest to eps
            dists = np.abs(F_lhs[:, 1] - eps)
            best_start_idx = np.argmin(dists)
            x0 = X_lhs[best_start_idx].copy()
            
            try:
                res = minimize(
                    penalized_obj,
                    x0,
                    method='L-BFGS-B',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if used < budget:
                    x_new = np.clip(res.x, lo, hi)
                    f_new = problem(x_new)
                    used += 1
                    all_X.append(x_new.reshape(1, -1))
                    all_F.append(f_new.reshape(1, -1))
            except Exception:
                pass

        # Combine all results
        X_all = np.vstack(all_X)
        F_all = np.vstack(all_F)
        
        # Phase 3: Local polish on non-dominated points
        # Find non-dominated points
        def non_dominated_indices(F):
            n = len(F)
            is_nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not is_nd[i]:
                    continue
                for j in range(n):
                    if i == j or not is_nd[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_nd[i] = False
                        break
            return np.where(is_nd)[0]

        nd_idx = non_dominated_indices(F_all)
        
        # Polish a few of them
        n_polish = min(5, len(nd_idx), (budget - used) // 5)
        if n_polish > 0 and time_ok():
            # Select diverse points
            if len(nd_idx) > n_polish:
                # Simple diversity: pick evenly spaced in f1
                nd_idx_sorted = nd_idx[np.argsort(F_all[nd_idx, 0])]
                step = len(nd_idx_sorted) // n_polish
                selected = nd_idx_sorted[::step][:n_polish]
            else:
                selected = nd_idx
            
            for idx in selected:
                if not time_ok() or used >= budget:
                    break
                
                x0 = X_all[idx].copy()
                
                # Polish by minimizing weighted sum with small weights
                # Try a few weight vectors
                weights_list = [
                    [1.0, 0.1],
                    [0.1, 1.0],
                    [1.0, 1.0]
                ]
                
                for w in weights_list:
                    if used >= budget or not time_ok():
                        break
                    
                    def weighted_obj(x):
                        fx = problem(x)
                        return np.dot(w, fx)
                    
                    try:
                        res = minimize(
                            weighted_obj,
                            x0,
                            method='L-BFGS-B',
                            bounds=list(zip(lo, hi)),
                            options={'maxiter': 20, 'ftol': 1e-10}
                        )
                        if res.success and used < budget:
                            x_new = np.clip(res.x, lo, hi)
                            f_new = problem(x_new)
                            used += 1
                            all_X.append(x_new.reshape(1, -1))
                            all_F.append(f_new.reshape(1, -1))
                    except Exception:
                        pass

        # Final combine
        X_final = np.vstack(all_X)
        F_final = np.vstack(all_F)
        
        return X_final
# EVOLVE-BLOCK-END
