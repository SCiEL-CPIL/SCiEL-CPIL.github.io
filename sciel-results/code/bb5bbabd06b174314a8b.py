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
        timeout = 290  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        
        rng = np.random.default_rng(42)
        
        # Latin Hypercube Sampling for initialization
        def lhs_sample(n, d, rng):
            """Generate n points in d dimensions using LHS."""
            points = np.zeros((n, d))
            for j in range(d):
                perm = rng.permutation(n)
                points[:, j] = (perm + rng.random(n)) / n
            return points
        
        # Initialization: ~25% of budget for diverse initial population
        init_n = max(20, min(50, self.budget // 4))
        
        # Mix of LHS and corner points
        X_init = []
        # LHS points
        lhs_pts = lhs_sample(init_n * 2 // 3, dim, rng)
        for pt in lhs_pts:
            X_init.append(pt * span + lo)
        # Corner and edge points
        n_corners = init_n - len(X_init)
        for _ in range(n_corners):
            # Random corner: each dim at lo or hi
            x = np.where(rng.random(dim) < 0.5, lo, hi)
            X_init.append(x)
            # Or random edge: 2-3 dims at bounds
            if rng.random() < 0.5 and dim >= 2:
                x = rng.random(dim) * span + lo
                n_bound = rng.integers(2, min(3, dim) + 1)
                idx = rng.choice(dim, size=n_bound, replace=False)
                x[idx] = np.where(rng.random(n_bound) < 0.5, lo[idx], hi[idx])
                X_init.append(x)
        
        X_init = np.array(X_init[:init_n])
        
        # Evaluate initial population
        F_init = np.array([problem(x) for x in X_init])
        used_evals = len(X_init)
        
        # Build initial Pareto front
        def is_pareto(F):
            """Return boolean mask of non-dominated points in F."""
            n = len(F)
            pareto = np.ones(n, dtype=bool)
            for i in range(n):
                if not pareto[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        pareto[i] = False
                        break
            return pareto
        
        pareto_mask = is_pareto(F_init)
        pareto_front = list(zip(X_init[pareto_mask], F_init[pareto_mask]))
        
        # Track best hypervolume for stagnation detection (simplified)
        ref_point = np.array([3000.0, 0.06])
        
        def compute_hypervolume(points, ref):
            """Compute 2D hypervolume. points: list of (x, y) objective vectors."""
            if not points:
                return 0.0
            pts = [p for p in points if all(p[i] < ref[i] for i in range(len(ref)))]
            if not pts:
                return 0.0
            pts.sort(key=lambda p: p[0])
            hv = 0.0
            prev_y = ref[1]
            for p in pts:
                if p[1] < prev_y:
                    hv += (ref[0] - p[0]) * (prev_y - p[1])
                    prev_y = p[1]
            return hv
        
        best_hv = compute_hypervolume(pareto_front, ref_point) if pareto_front else 0.0
        evals_since_improvement = 0
        stagnation_threshold = 50
        
        # Adaptive mutation parameters
        sigma_base = 0.3  # Initial step size as fraction of span
        sigma_min = 0.005
        sigma = sigma_base
        
        # Local polish using Nelder-Mead with Tchebycheff aggregation
        def tchebycheff_objective(x, weights, ref):
            """Tchebycheff aggregation of objectives."""
            f = problem(x)
            return np.max(weights * np.abs(f - ref))
        
        def polish_point(x0, budget_remaining):
            """Polish a point using Nelder-Mead with Tchebycheff aggregation."""
            if budget_remaining < 20:
                return None, None, 0
            
            best_x = None
            best_f = None
            best_agg = np.inf
            evals_used = 0
            
            # Use a smaller set of weights to save budget
            weights_list = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5])]
            
            for w in weights_list:
                if time.time() - start_time > timeout:
                    break
                if budget_remaining - evals_used < 15:
                    break
                
                try:
                    # Nelder-Mead doesn't use bounds directly, so we rely on clipping in problem or penalization
                    # However, problem clips to bounds, so NM might wander outside.
                    # To be safe, we can use a penalty or just let it clip.
                    # Since problem clips, NM will see the same value for any point outside bounds.
                    # This can cause NM to get stuck on the boundary.
                    # Better to use L-BFGS-B with bounds if available, but NM is often faster for 4D.
                    # Let's try NM with tight initial simplex.
                    
                    # Initialize simplex around x0
                    delta = 0.05 * span
                    simplex = [x0]
                    for i in range(dim):
                        x_i = x0.copy()
                        x_i[i] += delta[i]
                        x_i = np.clip(x_i, lo, hi)
                        simplex.append(x_i)
                        
                    res = minimize(
                        tchebycheff_objective,
                        x0,
                        args=(w, ref_point),
                        method='Nelder-Mead',
                        options={'maxiter': 50, 'maxfev': 30, 'xatol': 1e-6, 'fatol': 1e-6}
                    )
                    evals_used += res.nfev
                    
                    if res.success or res.fun < best_agg:
                        best_x = res.x
                        best_f = problem(res.x)
                        best_agg = res.fun
                except Exception:
                    continue
            
            return best_x, best_f, evals_used
        
        # Main loop
        while used_evals < self.budget:
            if time.time() - start_time > timeout:
                break
            
            if not pareto_front:
                # Fallback: random restart
                x = rng.random(dim) * span + lo
                f = problem(x)
                pareto_front.append((x, f))
                used_evals += 1
                continue
            
            # Select parent using tournament selection based on hypervolume contribution
            # Simple approach: random selection with bias toward extreme points
            n_front = len(pareto_front)
            
            # Weighted selection: points with more "space" around them get higher weight
            if n_front <= 2:
                parent_idx = rng.integers(n_front)
            else:
                # Compute simple crowding-like distance
                weights = np.ones(n_front)
                for i in range(n_front):
                    # Distance to nearest neighbor in objective space
                    min_dist = np.inf
                    for j in range(n_front):
                        if i == j:
                            continue
                        d = np.linalg.norm(pareto_front[i][1] - pareto_front[j][1])
                        min_dist = min(min_dist, d)
                    weights[i] = min_dist if min_dist > 0 else 1e-10
                # Normalize
                weights = weights / weights.sum()
                parent_idx = rng.choice(n_front, p=weights)
            
            parent_x, parent_f = pareto_front[parent_idx]
            
            # Decide between mutation and local polish
            # Use local polish for points that have been on the front for a while
            progress = used_evals / self.budget
            polish_prob = 0.3 + 0.4 * progress  # Increase polish probability as we progress
            
            if rng.random() < polish_prob and used_evals + 30 < self.budget:
                # Local polish
                polished_x, polished_f, evals_used = polish_point(parent_x, self.budget - used_evals)
                if polished_x is not None:
                    used_evals += evals_used
                    
                    # Check if polished point is non-dominated
                    dominated = False
                    for _, f_old in pareto_front:
                        if np.all(f_old <= polished_f) and np.any(f_old < polished_f):
                            dominated = True
                            break
                    
                    if not dominated:
                        pareto_front.append((polished_x, polished_f))
                        pareto_front = [
                            (x_p, f_p) for x_p, f_p in pareto_front
                            if not (np.all(polished_f <= f_p) and np.any(polished_f < f_p))
                        ]
                        
                        hv = compute_hypervolume(pareto_front, ref_point)
                        if hv > best_hv:
                            best_hv = hv
                            evals_since_improvement = 0
                        else:
                            evals_since_improvement += 1
                else:
                    evals_since_improvement += 1
                    continue
            else:
                # Mutation: Gaussian with adaptive sigma
                sigma = max(sigma_min, sigma_base * (1 - progress) ** 1.5)
                
                noise = rng.standard_normal(dim) * sigma * span
                child_x = parent_x + noise
                child_x = np.clip(child_x, lo, hi)
                
                # Evaluate
                child_f = problem(child_x)
                used_evals += 1
                
                # Update Pareto front
                dominated = False
                for _, f_old in pareto_front:
                    if np.all(f_old <= child_f) and np.any(f_old < child_f):
                        dominated = True
                        break
                
                if not dominated:
                    # Add child
                    pareto_front.append((child_x, child_f))
                    # Remove dominated points
                    pareto_front = [
                        (x_p, f_p) for x_p, f_p in pareto_front
                        if not (np.all(child_f <= f_p) and np.any(child_f < f_p))
                    ]
                        
                    # Update hypervolume
                    hv = compute_hypervolume(pareto_front, ref_point)
                    if hv > best_hv:
                        best_hv = hv
                        evals_since_improvement = 0
                    else:
                        evals_since_improvement += 1
                else:
                    evals_since_improvement += 1
            
            # If stuck, do a random restart
            if evals_since_improvement >= stagnation_threshold:
                x = rng.random(dim) * span + lo
                f = problem(x)
                used_evals += 1
                if used_evals < self.budget:
                    dominated = False
                    for _, f_old in pareto_front:
                        if np.all(f_old <= f) and np.any(f_old < f):
                            dominated = True
                            break
                    if not dominated:
                        pareto_front.append((x, f))
                        pareto_front = [
                            (x_p, f_p) for x_p, f_p in pareto_front
                            if not (np.all(f <= f_p) and np.any(f < f_p))
                        ]
                evals_since_improvement = 0
                # Reset sigma to explore
                sigma = max(sigma_base * 0.5, sigma_min)
        
        # Return the Pareto front (value is ignored but must not crash)
        return pareto_front
