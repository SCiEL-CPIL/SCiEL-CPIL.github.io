"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        rng = np.random.default_rng(123)
        
        # Global best tracking
        global_best_x = None
        global_best_f = np.inf
        calls_made = 0
        
        # Nelder-Mead parameters
        # Standard NM parameters, tuned for robustness
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrink
        tol = 1e-12   # Tolerance for stopping
        
        # We will run multiple restarts.
        # Each restart gets a budget slice. If it finishes early (converged), 
        # remaining budget goes to the next restart.
        
        # Initial set of starting points:
        # 1. Center (0,0,0,0,0)
        # 2. Random points in [-4, 4]^d
        # 3. Random points on the boundary
        
        # Let's define a list of initial points to try.
        # Since dim=5, we can afford several restarts.
        # Let's aim for up to 10 restarts max, but stop if budget is exhausted.
        
        initial_points = []
        # Center
        initial_points.append(np.zeros(self.dim))
        
        # Add random points
        for _ in range(10):
            p = rng.uniform(-4.5, 4.5, self.dim)
            initial_points.append(p)
            
        # Add some boundary points
        for _ in range(5):
            p = rng.choice([-4.9, 4.9], self.dim)
            # Mix some randoms
            mask = rng.random(self.dim) < 0.5
            p[mask] = rng.uniform(-4.9, 4.9, np.sum(mask))
            initial_points.append(p)

        # Sort initial points? No, we just iterate.
        # We can prioritize center first as it's often close to the optimum in BBOB if shift is small.
        # But the shift is hidden. 
        # Let's just iterate through the list.
        
        # To be efficient, we can interleave: 
        # Actually, let's just run them sequentially.
        # If a simplex converges fast, it frees up budget.
        
        for i_start in range(len(initial_points)):
            if time.time() > deadline - 5:
                break
            if calls_made >= self.budget:
                break
            
            # Setup for this restart
            x0 = initial_points[i_start]
            
            # Create initial simplex
            # Vertices: x0, and x0 + delta * e_i
            # Choose delta based on box size
            delta = 0.5 
            # Ensure vertices are within bounds
            # If x0 is close to boundary, adjust delta or clip
            
            simplex = [x0.copy()]
            for i in range(self.dim):
                xi = x0.copy()
                # Step size: if x0[i] + delta > 5, use smaller step or go other way
                step = delta
                if xi[i] + step > 5.0:
                    step = 5.0 - xi[i]
                    if step < 1e-6: step = -1e-6 # Should not happen with -4.5..4.5 init
                elif xi[i] - step < -5.0:
                    step = -5.0 - xi[i]
                    if step > -1e-6: step = 1e-6
                
                xi[i] += step
                simplex.append(xi)
            
            # Evaluate initial simplex
            simplex_f = []
            for v in simplex:
                if calls_made >= self.budget:
                    return global_best_x, global_best_f
                if time.time() > deadline - 2:
                    return global_best_x, global_best_f
                
                fv = func(v)
                simplex_f.append(fv)
                calls_made += 1
                
                if fv < global_best_f:
                    global_best_f = fv
                    global_best_x = v.copy()
            
            # Sort simplex by fitness
            # We maintain a list of (f, x)
            # It's easier to work with lists of vectors and values
            # Let's sort indices
            indices = np.argsort(simplex_f)
            simplex = [simplex[i] for i in indices]
            simplex_f = [simplex_f[i] for i in indices]
            
            # Main NM loop for this restart
            # Max iterations per restart to prevent infinite loops on flat spots
            max_iter = 200 
            # Also limit by remaining budget: need at least dim+1 calls for a full cycle?
            # Actually NM uses 1 new point per iteration.
            # So we can do (budget - calls_made) iterations.
            
            for _ in range(max_iter):
                if time.time() > deadline - 2:
                    return global_best_x, global_best_f
                if calls_made >= self.budget:
                    return global_best_x, global_best_f
                
                # 1. Check convergence
                # If range of function values is small, stop
                if max(simplex_f) - min(simplex_f) < tol * (1 + abs(min(simplex_f))):
                    break
                
                # 2. Compute centroid of all vertices except the worst
                # Worst is the last one (index self.dim)
                # Centroid of first self.dim vertices
                centroid = np.mean(simplex[:self.dim], axis=0)
                
                # 3. Reflection
                # x_r = centroid + alpha * (centroid - x_worst)
                x_worst = simplex[self.dim]
                x_r = centroid + alpha * (centroid - x_worst)
                
                # Clip to box
                x_r = np.clip(x_r, -5, 5)
                
                # Evaluate
                f_r = func(x_r)
                calls_made += 1
                if f_r < global_best_f:
                    global_best_f = f_r
                    global_best_x = x_r.copy()
                
                # 4. Decision
                if f_r < simplex_f[0]: # Better than best
                    # Try expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls_made += 1
                    if f_e < global_best_f:
                        global_best_f = f_e
                        global_best_x = x_e.copy()
                    
                    if f_e < f_r:
                        new_point = x_e
                        new_f = f_e
                    else:
                        new_point = x_r
                        new_f = f_r
                        
                elif f_r < simplex_f[-1]: # Better than worst (and not better than best)
                    new_point = x_r
                    new_f = f_r
                    
                else: # Worse than worst
                    # Contraction
                    x_c = centroid + sigma * (x_worst - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_made += 1
                    if f_c < global_best_f:
                        global_best_f = f_c
                        global_best_x = x_c.copy()
                    
                    if f_c < f_r:
                        new_point = x_c
                        new_f = f_c
                    else:
                        # Shrink
                        # Shrink all points towards the best
                        x_best = simplex[0]
                        # Replace all except best
                        for i in range(1, self.dim + 1):
                            simplex[i] = x_best + 0.5 * (simplex[i] - x_best)
                            # Re-evaluate? 
                            # Standard NM shrinks and re-evaluates. 
                            # This is expensive. 
                            # Alternative: just shrink and let the next iteration handle it?
                            # No, we need function values.
                            # Let's re-evaluate the shrunk points.
                            # This consumes dim calls.
                            # To save budget, we might just stop if shrinking is needed repeatedly.
                            # But let's do it properly.
                            if calls_made >= self.budget:
                                return global_best_x, global_best_f
                            f_shrunk = func(simplex[i])
                            simplex_f[i] = f_shrunk
                            if f_shrunk < global_best_f:
                                global_best_f = f_shrunk
                                global_best_x = simplex[i].copy()
                        
                        # After shrinking, we need to re-sort and continue
                        # But this breaks the "1 new point per iteration" assumption.
                        # For simplicity and budget safety, if we need to shrink, 
                        # we can just stop this restart and move to the next one.
                        # This is a heuristic to save budget for more diverse exploration.
                        break 
                
                # Replace the worst point
                simplex[self.dim] = new_point
                simplex_f[self.dim] = new_f
                
                # Re-sort simplex
                indices = np.argsort(simplex_f)
                simplex = [simplex[i] for i in indices]
                simplex_f = [simplex_f[i] for i in indices]

        return global_best_x, global_best_f

# EVOLVE-BLOCK-END
