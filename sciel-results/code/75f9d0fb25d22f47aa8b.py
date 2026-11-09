"""Rewritten optimizer: multi-start Nelder-Mead with deterministic simplex vertices."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead for Rastrigin f3 on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        # Track best
        best_x = None
        best_f = np.inf
        calls = 0

        def safe_func(x):
            nonlocal calls
            if time.time() > deadline:
                return np.inf
            calls += 1
            return func(x)

        # Generate 10 deterministic starting points: vertices of a regular 5-simplex
        # inscribed in [-5,5]^5, plus center
        # For dim=5, a regular 5-simplex has 6 vertices. We'll use those plus
        # additional structured points for diversity.
        
        # Vertices of regular simplex in dim dimensions:
        # Use the standard construction: v_i has -1 in position i and 1/(dim-1) elsewhere,
        # then center and scale.
        n_verts = dim + 1  # 6 for dim=5
        verts = np.zeros((n_verts, dim))
        for i in range(n_verts):
            verts[i] = 1.0 / (dim - 1)
            verts[i, i] = -1.0
        # Center
        verts = verts - verts.mean(axis=0)
        # Scale to fit in [-5, 5]
        max_abs = np.abs(verts).max()
        verts = verts * (4.5 / max_abs)  # keep slightly inside bounds
        # Shift to center of domain
        verts = verts + 0.0  # domain is [-5,5], center is 0

        # Add more diverse starting points: center, and axis-aligned points
        extra_points = []
        extra_points.append(np.zeros(dim))  # center
        for i in range(dim):
            for sign in [1, -1]:
                p = np.zeros(dim)
                p[i] = sign * 3.0
                extra_points.append(p)
        
        all_starts = list(verts) + extra_points
        
        # Budget allocation: divide among starts
        n_starts = len(all_starts)
        # Give more budget to earlier starts (they're likely better positioned)
        # Use a geometric distribution of budget
        base_budget = max(10, budget // (n_starts * 2))
        # Reserve some for final polish
        
        # Nelder-Mead implementation
        def nelder_mead(func, x0, max_iter, deadline):
            dim = len(x0)
            # Initial simplex
            simplex = [x0.copy()]
            f_vals = [func(x0)]
            
            # Extend simplex along each axis
            for i in range(dim):
                new_pt = x0.copy()
                # Step size: 0.05 if x0[i] is not at boundary, else adjust
                if abs(x0[i]) > 4.5:
                    step = 0.5 * np.sign(5.0 - x0[i])
                else:
                    step = 0.05
                new_pt[i] += step
                new_pt = np.clip(new_pt, -5, 5)
                simplex.append(new_pt)
                f_vals.append(func(new_pt))
            
            n = len(simplex)
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Sort simplex by function value
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                
                if f_vals[0] <= f_r < f_vals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                elif f_r < f_vals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_vals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    
                    if f_c < min(f_r, f_vals[-1]):
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_vals[i] = func(simplex[i])
            
            # Return best point
            best_idx = np.argmin(f_vals)
            return simplex[best_idx], f_vals[best_idx]

        # Run multi-start
        for start_idx, x0 in enumerate(all_starts):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            
            # Allocate iterations: earlier starts get more
            remaining_budget = budget - calls
            remaining_starts = n_starts - start_idx
            # Give proportional budget with diminishing returns
            alloc = max(20, int(remaining_budget / remaining_starts * 1.5))
            alloc = min(alloc, remaining_budget)
            
            x_best, f_best = nelder_mead(safe_func, x0, alloc, deadline)
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best
        
        # Final polish with Nelder-Mead on the best point found
        if best_x is not None and calls < budget and time.time() < deadline:
            remaining = budget - calls
            if remaining > 10:
                x_polish, f_polish = nelder_mead(safe_func, best_x, remaining, deadline)
                if f_polish < best_f:
                    best_f = f_polish
                    best_x = x_polish

        if best_x is None:
            # Fallback: return origin
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
