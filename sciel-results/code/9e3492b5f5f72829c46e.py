"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Multi-start Nelder-Mead with Sobol initialization and dynamic budget allocation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s kill

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def safe_eval(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget or time.time() > deadline:
                return None
            x_clipped = np.clip(x, -5, 5)
            try:
                f_val = float(func(x_clipped))
            except Exception:
                return None
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Generate Sobol points for initial simplex centers
        # We need enough points for n_starts simplices.
        # Each simplex has dim+1 points, but we only need 1 center point from Sobol to start.
        # The other simplex points are generated locally around the center.
        n_starts = 12
        sampler = qmc.Sobol(d=dim, scramble=True)
        # 2^k points. 2^4 = 16 is enough for 12 starts.
        sobol_points = sampler.random(16)
        
        # Map Sobol [0,1]^d to [-4, 4]^d to stay safely inside [-5,5]^5
        # The minimizer is in [-4, 4]^5, so this coverage is ideal.
        sobol_points = sobol_points * 8.0 - 4.0

        # Initialize simplexes
        simplexes = []
        for i in range(n_starts):
            center = sobol_points[i]
            
            # Build initial simplex around center
            # Simplex is a set of dim+1 points.
            # Point 0: center
            # Points 1..dim: center + delta * e_i
            delta = 1.0 # Initial step size
            simplex_points = [center.copy()]
            for d in range(dim):
                p = center.copy()
                p[d] += delta
                simplex_points.append(p)
            
            # Evaluate all points in the simplex
            f_vals = []
            valid = True
            for p in simplex_points:
                f_val = safe_eval(p)
                if f_val is None:
                    valid = False
                    break
                f_vals.append(f_val)
            
            if not valid:
                continue
                
            simplex = {
                'points': np.array(simplex_points),
                'fvals': np.array(f_vals),
                'active': True
            }
            simplexes.append(simplex)

        if not simplexes:
            # Fallback if budget was too small or time ran out immediately
            return best_x, best_f

        # Phase 2: Iterative Nelder-Mead on active simplexes
        # We will cycle through simplexes.
        # To favor the best one, we can use a weighted round-robin or just check which is best periodically.
        # Strategy: 
        # 1. Evaluate all simplexes for a few generations (exploration).
        # 2. Identify the best simplex.
        # 3. Continue Nelder-Mead on the best simplex until budget/time is up (exploitation).
        
        # Let's do a few rounds of Nelder-Mead on all active simplexes first.
        n_rounds_exploration = 5
        
        for round_idx in range(n_rounds_exploration):
            if time.time() > deadline or calls_made >= budget:
                break
            
            for simplex in simplexes:
                if not simplex['active']:
                    continue
                if time.time() > deadline or calls_made >= budget:
                    break
                
                # Perform one Nelder-Mead iteration on this simplex
                points = simplex['points']
                fvals = simplex['fvals']
                
                # Sort points by fval
                order = np.argsort(fvals)
                points = points[order]
                fvals = fvals[order]
                
                # Centroid of all points except the worst
                centroid = np.mean(points[:-1], axis=0)
                worst_point = points[-1]
                
                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - worst_point)
                f_ref = safe_eval(x_ref)
                
                if f_ref is None:
                    simplex['active'] = False
                    continue
                    
                if f_ref < fvals[-2]:
                    # Check for expansion
                    if f_ref < fvals[0]:
                        gamma = 2.0
                        x_exp = centroid + gamma * (centroid - worst_point)
                        f_exp = safe_eval(x_exp)
                        if f_exp is None:
                            simplex['active'] = False
                            continue
                        if f_exp < f_ref:
                            # Accept expansion
                            points[-1] = x_exp
                            fvals[-1] = f_exp
                        else:
                            # Accept reflection
                            points[-1] = x_ref
                            fvals[-1] = f_ref
                    else:
                        # Accept reflection
                        points[-1] = x_ref
                        fvals[-1] = f_ref
                else:
                    # Reflection was not better than second worst
                    # Contraction
                    beta = 0.5
                    x_con = centroid + beta * (worst_point - centroid)
                    f_con = safe_eval(x_con)
                    if f_con is None:
                        simplex['active'] = False
                        continue
                    if f_con < fvals[-1]:
                        points[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            points[i] = points[0] + 0.5 * (points[i] - points[0])
                            f_shrink = safe_eval(points[i])
                            if f_shrink is None:
                                simplex['active'] = False
                                break
                            fvals[i] = f_shrink
                        if simplex['active']:
                            # Re-sort after shrink
                            order = np.argsort(fvals)
                            points = points[order]
                            fvals = fvals[order]
                
                simplex['points'] = points
                simplex['fvals'] = fvals

        # Identify the best simplex
        if not any(s['active'] for s in simplexes):
            return best_x, best_f

        best_simplex_idx = 0
        best_simplex_f = np.inf
        for i, s in enumerate(simplexes):
            if s['active']:
                min_f = np.min(s['fvals'])
                if min_f < best_simplex_f:
                    best_simplex_f = min_f
                    best_simplex_idx = i

        # Phase 3: Focus on the best simplex
        # We can also keep a few secondary simplexes if they are close, but for simplicity and speed,
        # focusing on the single best one often yields the highest AOCC in the later budget phases
        # because NM converges fast.
        
        active_simplex = simplexes[best_simplex_idx]
        
        # Continue NM iterations on this single simplex until budget/time runs out
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            points = active_simplex['points']
            fvals = active_simplex['fvals']
            
            # Sort
            order = np.argsort(fvals)
            points = points[order]
            fvals = fvals[order]
            
            centroid = np.mean(points[:-1], axis=0)
            worst_point = points[-1]
            
            # Reflection
            alpha = 1.0
            x_ref = centroid + alpha * (centroid - worst_point)
            f_ref = safe_eval(x_ref)
            
            if f_ref is None:
                break
                
            if f_ref < fvals[-2]:
                if f_ref < fvals[0]:
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - worst_point)
                    f_exp = safe_eval(x_exp)
                    if f_exp is None:
                        break
                    if f_exp < f_ref:
                        points[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        points[-1] = x_ref
                        fvals[-1] = f_ref
                else:
                    points[-1] = x_ref
                    fvals[-1] = f_ref
            else:
                beta = 0.5
                x_con = centroid + beta * (worst_point - centroid)
                f_con = safe_eval(x_con)
                if f_con is None:
                    break
                if f_con < fvals[-1]:
                    points[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        points[i] = points[0] + 0.5 * (points[i] - points[0])
                        f_shrink = safe_eval(points[i])
                        if f_shrink is None:
                            break
                        fvals[i] = f_shrink
                    if f_shrink is None:
                        break
                    order = np.argsort(fvals)
                    points = points[order]
                    fvals = fvals[order]
                    
                    # If simplex has collapsed, we might be stuck. 
                    # For now, we just continue. If we detect very small spread, we could restart,
                    # but budget is likely low by then.
            
            active_simplex['points'] = points
            active_simplex['fvals'] = fvals

        return best_x, best_f
# EVOLVE-BLOCK-END
