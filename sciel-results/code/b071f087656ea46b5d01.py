"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
        deadline = start_time + 300.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Reference point for hypervolume calculation
        ref = np.array([3000.0, 0.06])
        
        # State
        X = []
        F = []
        used = 0
        
        def is_valid(f):
            return np.all(f < ref)
            
        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            used += 1
            return x, f

        def hv_contribution(f, current_front):
            """Calculate the hypervolume added by point f to the current front."""
            if not is_valid(f):
                return 0.0
            # If front is empty, contribution is area of f vs ref
            if not current_front:
                return (ref[0] - f[0]) * (ref[1] - f[1])
            
            # For 2D, contribution is the area of the rectangle defined by f and ref,
            # minus the area already covered by points in the front that dominate the part of that rectangle.
            # Actually, simpler: Sort front by f1. 
            # The contribution of a new point is the area of the box (f, ref) 
            # minus the union of boxes of existing points that intersect it.
            # In 2D, we can just check which existing points are "below-right" of f? No.
            # Let's use the standard incremental HV update for 2D.
            # Sort current front by f[0] ascending.
            front_sorted = sorted(current_front, key=lambda x: x[0])
            
            # Find the region dominated by f but not by any point in front_sorted.
            # A point p dominates the part of f's box if p[0] <= f[0] and p[1] <= f[1].
            # Wait, dominance: p <= f means p is better.
            # So if p is better than f, p "covers" part of the space f would claim?
            # No, HV is the volume dominated by the set.
            # If we add f, the new HV is HV(front U {f}).
            # Delta HV = HV(front U {f}) - HV(front).
            
            # Let's compute HV of (front U {f}) and HV of (front) and subtract.
            # This is O(N^2) worst case but N is small.
            
            def calc_hv(points, ref):
                if not points:
                    return 0.0
                # Sort by first objective
                pts = sorted(points, key=lambda x: x[0])
                hv = 0.0
                max_f2 = -np.inf
                for i in range(len(pts)):
                    f = pts[i]
                    if f[1] > max_f2:
                        # This point is on the Pareto front of the subset
                        # It contributes width * height
                        # Width is distance to next point's f1 (or ref)
                        # Height is distance from f1 to ref
                        next_f1 = pts[i+1][0] if i+1 < len(pts) else ref[0]
                        if next_f1 > f[0]:
                            hv += (next_f1 - f[0]) * (ref[1] - f[1])
                        max_f2 = f[1]
                return hv

            hv_old = calc_hv(current_front, ref)
            hv_new = calc_hv(current_front + [f], ref)
            return hv_new - hv_old

        def get_pareto_front(X, F):
            """Return indices of non-dominated points."""
            n = len(F)
            if n == 0: return []
            is_dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if is_dominated[i]: continue
                for j in range(n):
                    if i == j: continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dominated[i] = True
                        break
            return [i for i in range(n) if not is_dominated[i]]

        # 1. Initial Random Sampling
        # Use a significant portion of budget for initialization to ensure diversity
        init_budget = min(40, self.budget // 5)
        for _ in range(init_budget):
            if used >= self.budget: break
            if time.time() > deadline - 10: break
            x = lo + (hi - lo) * np.random.rand(self.dim)
            x, f = evaluate(x)
            if is_valid(f):
                X.append(x)
                F.append(f)

        # 2. Memetic Evolutionary Loop
        # We will alternate between global exploration (mutation/crossover) and local exploitation (Nelder-Mead)
        # Strategy:
        # - Identify the Pareto front.
        # - Calculate HV contribution of each point on the front.
        # - Pick the point with the highest "potential" (high HV contribution or isolated in objective space).
        # - Perform local search around it.
        # - Also perform random mutations to maintain diversity.

        # Local Search Function
        def local_search(x0, f0, max_evals=5):
            nonlocal used
            # Nelder-Mead on a scalarized objective (e.g. Tchebycheff or simple sum)
            # Since we want to improve HV, we can try to minimize a weighted sum or distance to a specific point.
            # Here, let's try to minimize the sum of objectives, but constrained to stay in the "interesting" region.
            # Actually, standard NM minimizes a scalar. Let's minimize f0[0] + f0[1].
            
            if not is_valid(f0):
                return x0, f0
            
            def obj(x):
                nonlocal used
                if used >= self.budget:
                    return np.inf
                x_clipped = np.clip(x, lo, hi)
                f_val = np.array(problem(x_clipped))
                used += 1
                if not is_valid(f_val):
                    return np.inf
                return f_val[0] + f_val[1]
                
            # Initial simplex
            x0 = np.clip(x0, lo, hi)
            xk = x0.copy()
            # Perturb slightly to create simplex
            step = 0.1 * (hi - lo)
            simplex = [x0]
            for i in range(self.dim):
                xi = x0.copy()
                xi[i] += step[i]
                simplex.append(xi)
            
            res = minimize(obj, x0, method='Nelder-Mead', 
                           options={'maxiter': max_evals, 'xatol': 1e-6, 'fatol': 1e-6})
            
            x_best = res.x
            f_best = np.array(problem(np.clip(x_best, lo, hi)))
            used += 1
            return x_best, f_best

        # Main Loop
        while used < self.budget and time.time() < deadline - 5:
            
            # Get current Pareto front
            front_indices = get_pareto_front(X, F)
            
            if not front_indices:
                # If no valid points, just do random search
                x = lo + (hi - lo) * np.random.rand(self.dim)
                x, f = evaluate(x)
                if is_valid(f):
                    X.append(x)
                    F.append(f)
                continue
            
            # Calculate HV contribution for each point on the front
            # To speed up, we can approximate or just pick the one that seems most "central" or has large gap
            # For simplicity and robustness, let's pick a random point from the front 50% of the time,
            # and the one with max "gap" to neighbors 50% of the time.
            
            if np.random.rand() < 0.5:
                # Exploit: Pick point with largest gap to neighbors in objective space
                # Sort front by f[0]
                front_points = [(i, F[i]) for i in front_indices]
                front_points.sort(key=lambda x: x[1][0])
                
                # Calculate gaps
                best_idx = front_points[0][0]
                max_gap = -1
                for k in range(len(front_points)):
                    i, f = front_points[k]
                    # Gap to left neighbor
                    left_f1 = front_points[k-1][1][0] if k > 0 else -np.inf
                    right_f1 = front_points[k+1][1][0] if k < len(front_points)-1 else np.inf
                    # Approximate gap as distance to nearest neighbor in f1
                    gap_l = abs(f[0] - left_f1) if np.isfinite(left_f1) else 1000
                    gap_r = abs(f[0] - right_f1) if np.isfinite(right_f1) else 1000
                    gap = min(gap_l, gap_r)
                    if gap > max_gap:
                        max_gap = gap
                        best_idx = i
                target_idx = best_idx
            else:
                # Explore: Pick random point from front
                target_idx = front_indices[np.random.randint(len(front_indices))]
                
            # Perform Local Search around target
            x_target = X[target_idx]
            f_target = F[target_idx]
            
            # Determine how many evals to spend on local search
            # Adaptive: if we are close to deadline, spend less
            remaining = self.budget - used
            time_left = deadline - time.time()
            
            # Cap local search evals
            ls_evals = min(5, remaining)
            
            if ls_evals > 0:
                x_new, f_new = local_search(x_target, f_target, max_evals=ls_evals)
                
                # Add to population if valid
                if is_valid(f_new):
                    X.append(x_new)
                    F.append(f_new)
                    
                    # Optional: Remove dominated points to keep population small?
                    # Not strictly necessary for HV calculation which uses all points,
                    # but keeps memory manageable. Since budget is only 200, we can keep all.
            
            # Also add a random mutation to maintain diversity
            if used < self.budget and time.time() < deadline - 2:
                # Mutate a random point
                i_mut = np.random.randint(len(X))
                x_mut = X[i_mut] + 0.2 * (hi - lo) * np.random.randn(self.dim)
                x_mut, f_mut = evaluate(x_mut)
                if is_valid(f_mut):
                    X.append(x_mut)
                    F.append(f_mut)

        return X
# EVOLVE-BLOCK-END
