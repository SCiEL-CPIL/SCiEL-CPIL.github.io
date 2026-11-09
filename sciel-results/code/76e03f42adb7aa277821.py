# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Powell's method optimizer for anisotropic functions like Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin vs. 330 s kill
        max_calls = budget

        calls = [0]
        best_fx = float('inf')
        best_x = np.zeros(dim)
        rng = np.random.default_rng(42)

        def f(x):
            nonlocal best_fx, best_x
            if calls[0] >= max_calls:
                return float('inf')
            calls[0] += 1
            try:
                x_clipped = np.clip(x, -5.0, 5.0)
                val = float(func(x_clipped))
            except Exception:
                return float('inf')
            if not np.isfinite(val):
                return float('inf')
            if val < best_fx:
                best_fx = val
                best_x = x_clipped.copy()
            return val

        def remaining():
            return deadline - time.time()

        def line_search(x0, d, f0, max_iter=20):
            """
            Perform 1D line search along direction d starting from x0.
            Uses golden section search.
            Returns best point x, value fx, and step size t.
            """
            if calls[0] >= max_calls:
                return x0, f0, 0.0
            
            # Initial bracketing
            # Try small steps to find direction of decrease
            t = 1.0
            x1 = x0 + t * d
            f1 = f(x1)
            
            if f1 >= f0:
                # Try smaller step or opposite direction? 
                # For Powell, we usually assume we are moving in a direction that initially decreases.
                # If not, we might need to adjust.
                # Let's try t = 0.5
                t = 0.5
                x1 = x0 + t * d
                f1 = f(x1)
                if f1 >= f0:
                    # If even smaller step doesn't work, return current
                    return x0, f0, 0.0
            
            # Now f1 < f0, so minimum is between x0 and x1? Not necessarily, but likely nearby.
            # Let's expand to find a bracket [t_lo, t_hi] where f(t_lo) < f(t_mid) and f(t_hi) > f(t_mid)
            # Actually, simpler: Golden Section Search on [0, t_current_max]
            
            # Expand t to find an upper bound where function increases
            t_hi = t
            f_hi = f1
            while f_hi < f0 and calls[0] < max_calls and remaining() > 1.0:
                t_hi *= 2.0
                x_hi = x0 + t_hi * d
                f_hi = f(x_hi)
                if f_hi >= f0 or t_hi > 10.0: # Don't go too far
                    break
            
            # Bracket is [0, t_hi]
            t_lo = 0.0
            f_lo = f0
            
            # Golden section search
            gr = (np.sqrt(5) - 1) / 2  # ~0.618
            t1 = t_lo + (1 - gr) * (t_hi - t_lo)
            t2 = t_lo + gr * (t_hi - t_lo)
            
            x1 = x0 + t1 * d
            x2 = x0 + t2 * d
            f1 = f(x1)
            f2 = f(x2)
            
            for _ in range(max_iter):
                if calls[0] >= max_calls or remaining() < 1.0:
                    break
                if abs(t_hi - t_lo) < 1e-6 * (1 + abs(t_hi)):
                    break
                
                if f1 < f2:
                    t_hi = t2
                    t2 = t1
                    f2 = f1
                    t1 = t_lo + (1 - gr) * (t_hi - t_lo)
                    x1 = x0 + t1 * d
                    f1 = f(x1)
                else:
                    t_lo = t1
                    t1 = t2
                    f1 = f2
                    t2 = t_lo + gr * (t_hi - t_lo)
                    x2 = x0 + t2 * d
                    f2 = f(x2)
            
            # Best of the evaluated points
            if f1 < f2:
                return x1, f1, t1
            else:
                return x2, f2, t2

        def powell_method(x0, fx0, max_outer_iter=20):
            """
            Powell's Method.
            """
            n = dim
            # Initial directions: coordinate axes
            directions = [np.eye(n)[i] for i in range(n)]
            
            x = x0.copy()
            fx = fx0
            
            for outer_iter in range(max_outer_iter):
                if calls[0] >= max_calls or remaining() < 5.0:
                    break
                
                x_old = x.copy()
                f_old = fx
                
                # Search along each direction
                reductions = []
                for i in range(n):
                    if calls[0] >= max_calls or remaining() < 5.0:
                        break
                    d = directions[i]
                    x_new, fx_new, t = line_search(x, d, fx)
                    reduction = fx - fx_new
                    reductions.append(reduction)
                    x = x_new
                    fx = fx_new
                
                if len(reductions) < n:
                    break
                    
                # Check if we made significant progress
                total_reduction = sum(reductions)
                if total_reduction < 1e-8:
                    break
                
                # Determine if we should add a new direction
                # Find the direction with the smallest reduction
                min_reduction = min(reductions)
                min_idx = reductions.index(min_reduction)
                
                # Condition for new direction: total reduction > 2 * min reduction
                if total_reduction > 2 * min_reduction:
                    # New direction is x - x_old
                    new_dir = x - x_old
                    norm = np.linalg.norm(new_dir)
                    if norm > 1e-12:
                        new_dir /= norm
                        # Replace the direction with the smallest reduction
                        directions[min_idx] = new_dir
                        # Optionally, reorder directions so the new one is last? 
                        # Standard Powell keeps the set but replaces the worst one.
                        # Some implementations rotate the set.
                        # Let's just replace.
                    else:
                        break
                else:
                    # No new direction, check convergence
                    if total_reduction < 1e-6:
                        break

            return x, fx

        # Phase 1: Powell's Method
        # Start at origin
        x0 = np.zeros(dim)
        fx0 = f(x0)
        
        # Allocate most budget to Powell
        # Powell uses ~n * line_search_cost per iteration.
        # With 500 calls, we can afford ~20-50 iterations if line search is efficient.
        x, fx = powell_method(x0, fx0, max_outer_iter=50)
        
        # Phase 2: If we have budget left, do a quick Nelder-Mead polish or Gradient polish
        # Powell might leave some residual error in orthogonal directions if the anisotropy is extreme.
        # A few NM steps can help clean up.
        
        if calls[0] < max_calls - 10 and remaining() > 10.0:
            # Quick NM polish
            # We need a simplex. Use x and x +/- small steps in coordinate directions.
            simplex = [x.copy()]
            f_values = [fx]
            step = 0.1
            for i in range(dim):
                xi = x.copy()
                xi[i] += step
                fi = f(xi)
                simplex.append(xi)
                f_values.append(fi)
                
            if calls[0] < max_calls:
                # Run a few NM steps
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma = 0.5
                
                for _ in range(20): # Limited steps
                    if calls[0] >= max_calls or remaining() < 5.0:
                        break
                    order = np.argsort(f_values)
                    simplex = [simplex[i] for i in order]
                    f_values = [f_values[i] for i in order]
                    
                    if f_values[-1] - f_values[0] < 1e-10:
                        break
                    
                    x_worst = simplex[-1]
                    x_best = simplex[0]
                    x_centroid = np.mean(simplex[:-1], axis=0)
                    
                    x_ref = x_centroid + alpha * (x_centroid - x_worst)
                    f_ref = f(x_ref)
                    
                    if f_values[0] <= f_ref < f_values[-2]:
                        simplex[-1] = x_ref
                        f_values[-1] = f_ref
                    elif f_ref < f_values[0]:
                        x_exp = x_centroid + gamma * (x_ref - x_centroid)
                        f_exp = f(x_exp)
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            f_values[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            f_values[-1] = f_ref
                    else:
                        if f_ref < f_values[-1]:
                            x_con = x_centroid + rho * (x_ref - x_centroid)
                        else:
                            x_con = x_centroid - rho * (x_worst - x_centroid)
                        
                        f_con = f(x_con)
                        if f_con < min(f_ref, f_values[-1]):
                            simplex[-1] = x_con
                            f_values[-1] = f_con
                        else:
                            for i in range(1, dim + 1):
                                simplex[i] = x_best + sigma * (simplex[i] - x_best)
                                f_values[i] = f(simplex[i])
                                
                # Update best
                final_order = np.argsort(f_values)
                if f_values[final_order[0]] < best_fx:
                    best_x = simplex[final_order[0]].copy()
                    best_fx = f_values[final_order[0]]

        if best_x is None or not np.all(np.isfinite(best_x)):
            best_x = np.zeros(dim)
            if calls[0] < max_calls:
                best_fx = f(best_x)

        return best_x, best_fx
# EVOLVE-BLOCK-END
