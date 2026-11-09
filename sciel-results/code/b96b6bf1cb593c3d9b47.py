"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def halton_sequence(n, dim):
    """
    Generate n points from the Halton sequence in [0, 1]^dim.
    """
    points = []
    for i in range(1, n + 1):
        p = []
        for j in range(1, dim + 1):
            f = 1.0
            i_copy = i
            while i_copy > 0:
                i_copy, r = divmod(i_copy, j)
                f /= j
                p.append(r * f)
        points.append(p)
    return np.array(points)

class Optimizer:
    """Deterministic Halton sampling followed by multi-start Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. We target 300s to be safe.
        deadline = start_time + 300.0
        
        dim = self.dim
        total_budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # 1. Global Exploration Phase
        # We want to use a significant chunk of the budget for global sampling.
        # With 500 calls, let's use 200 for global sampling.
        # This ensures we have 200 points to evaluate immediately.
        n_global = min(200, total_budget)
        
        # Generate Halton points in [0, 1]^dim
        halton_pts = halton_sequence(n_global, dim)
        
        # Map to [-5, 5]^dim
        # Scale: x = -5 + 10 * h
        global_candidates = -5.0 + 10.0 * halton_pts
        
        # Evaluate all global candidates
        global_results = []
        for i in range(n_global):
            if time.time() > deadline - 10.0: # Leave 10s buffer
                break
            if calls_made >= total_budget:
                break
                
            x = global_candidates[i]
            f = func(x)
            calls_made += 1
            global_results.append((f, x))
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # 2. Local Refinement Phase
        # Select top-k basins from global results
        # Sort by fitness
        global_results.sort(key=lambda t: t[0])
        
        # How many local optimizations to run?
        # We have remaining budget: total_budget - calls_made
        # Nelder-Mead in 5D typically needs ~50-100 evaluations to converge well.
        # Let's allocate the remaining budget to the best few candidates.
        
        remaining_budget = total_budget - calls_made
        if remaining_budget <= 0:
            return best_x, best_f
            
        # Choose top 5 distinct basins (simple check: if points are too close, skip)
        # In Rastrigin, basins are spaced by 2 units. 
        # We'll just take the top 5 from the Halton set, they are likely in different basins 
        # or at least diverse enough.
        
        top_k = 5
        # Ensure we don't try to run more than we have budget for
        # Assume ~80 calls per NM run.
        max_nm_runs = max(1, remaining_budget // 80)
        max_nm_runs = min(max_nm_runs, top_k)
        
        # Select starting points
        start_points = []
        for f, x in global_results[:top_k]:
            start_points.append(x)
            
        # Run Nelder-Mead on selected points
        # We will distribute the remaining budget among these runs.
        # To maximize AOCC, we should run them in parallel? No, sequential.
        # But we can interleave? 
        # Actually, since AOCC is based on the *best* value so far, 
        # and we already have a good best_f from global phase, 
        # the local phase just needs to improve it.
        
        # Strategy: Run NM on the best point first to get a quick improvement,
        # then others.
        
        calls_per_nm = remaining_budget // max_nm_runs
        # If calls_per_nm is too small, NM won't work well.
        if calls_per_nm < 20:
             # Fallback: just random local search or more global sampling
             # For now, if budget is too low, do a simple hill climb on best_x
             pass
        else:
             for idx in range(max_nm_runs):
                 if time.time() > deadline - 5.0:
                     break
                 if calls_made >= total_budget:
                     break
                 
                 x_start = start_points[idx].copy()
                 x_best_nm, f_best_nm, calls_used = self._nelder_mead(func, x_start, calls_per_nm, dim, best_x, best_f)
                 
                 calls_made += calls_used
                 
                 if f_best_nm < best_f:
                     best_f = f_best_nm
                     best_x = x_best_nm.copy()
                     
                 # Update start points for subsequent runs if we found something better?
                 # Not strictly necessary as we are running on pre-selected basins.
        
        return best_x, best_f

    def _nelder_mead(self, func, x0, max_calls, dim, global_best_x, global_best_f):
        """
        Standard Nelder-Mead implementation with call counting.
        Returns (best_x, best_f, calls_made_in_this_run)
        """
        # Initialize simplex
        # 5D -> 6 vertices
        n = dim + 1
        simplex = []
        
        # First vertex is x0
        simplex.append(x0.copy())
        
        # Other vertices
        for i in range(dim):
            v = x0.copy()
            # Step size: 0.05 if x0[i] != 0 else 0.00025 (standard scipy default)
            # But for Rastrigin, basins are 2 units apart. 0.05 is good for local.
            v[i] += 0.05 if x0[i] != 0 else 0.00025
            simplex.append(v)
        
        # Evaluate initial simplex
        f_vals = []
        calls = 0
        
        for i in range(n):
            if calls >= max_calls:
                break
            f = func(simplex[i])
            calls += 1
            f_vals.append(f)
            
            if f < global_best_f:
                global_best_f = f
                global_best_x = simplex[i].copy()
        
        if calls >= max_calls:
            return global_best_x, global_best_f, calls
            
        # Sort simplex
        idx = np.argsort(f_vals)
        simplex = [simplex[i] for i in idx]
        f_vals = [f_vals[i] for i in idx]
        
        # Optimization loop
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage
        
        while calls < max_calls:
            # Check convergence (optional, but good to stop early if stuck)
            # If range of f_vals is very small, break
            if max(f_vals) - min(f_vals) < 1e-6:
                break
                
            # 1. Reflection
            x_centroid = np.mean(simplex[1:], axis=0)
            x_ref = x_centroid + alpha * (x_centroid - simplex[0])
            x_ref = np.clip(x_ref, -5, 5)
            
            f_ref = func(x_ref)
            calls += 1
            
            if f_ref < global_best_f:
                global_best_f = f_ref
                global_best_x = x_ref.copy()
                
            if f_vals[0] <= f_ref < f_vals[-1]:
                # Accept reflection
                simplex[0] = x_ref
                f_vals[0] = f_ref
            elif f_ref < f_vals[0]:
                # Expansion
                x_exp = x_centroid + gamma * (x_ref - x_centroid)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
                calls += 1
                
                if f_exp < global_best_f:
                    global_best_f = f_exp
                    global_best_x = x_exp.copy()
                    
                if f_exp < f_ref:
                    simplex[0] = x_exp
                    f_vals[0] = f_exp
                else:
                    simplex[0] = x_ref
                    f_vals[0] = f_ref
            else:
                # Contraction
                if f_ref < f_vals[-1]:
                    # Outside contraction
                    x_con = x_centroid + rho * (x_ref - x_centroid)
                else:
                    # Inside contraction
                    x_con = x_centroid - rho * (x_ref - x_centroid)
                    
                x_con = np.clip(x_con, -5, 5)
                f_con = func(x_con)
                calls += 1
                
                if f_con < global_best_f:
                    global_best_f = f_con
                    global_best_x = x_con.copy()
                    
                if f_con < min(f_ref, f_vals[-1]):
                    simplex[0] = x_con
                    f_vals[0] = f_con
                else:
                    # Shrinkage
                    for i in range(1, n):
                        simplex[i] = (simplex[0] + simplex[i]) / 2.0
                        f_vals[i] = func(simplex[i])
                        calls += 1
                        if calls >= max_calls:
                            break
                        
                        if f_vals[i] < global_best_f:
                            global_best_f = f_vals[i]
                            global_best_x = simplex[i].copy()
                    if calls >= max_calls:
                        break
            
            # Re-sort
            idx = np.argsort(f_vals)
            simplex = [simplex[i] for i in idx]
            f_vals = [f_vals[i] for i in idx]
            
        return global_best_x, global_best_f, calls

# EVOLVE-BLOCK-END
