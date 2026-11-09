"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-objective optimizer using Structured Decomposition and Random Restart Local Search.
    
    Strategy:
    WFG4 is a separable problem with 'deciding' variables (first 2) that determine the 
    Pareto front shape and 'positioning' variables (rest) that determine the distance 
    from the ideal point.
    
    1. Initialize a population of random points.
    2. For each point, perform a local search on the 'positioning' variables (indices 2..11)
       to minimize a scalar aggregation of objectives (Tchebycheff or sum), while keeping 
       'deciding' variables (indices 0..1) fixed. This drives points towards the Pareto front.
    3. Maintain a non-dominated archive.
    4. Use remaining budget for random restarts or further local refinements.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # WFG4 structure:
        # Variables 0, 1 are "deciding" (shape the front)
        # Variables 2..11 are "positioning" (distance from ideal)
        # We will treat indices 0 and 1 as fixed during local search steps
        # and optimize indices 2..11 to push the point towards the front.
        
        deciding_dims = [0, 1]
        positioning_dims = list(range(2, self.dim))
        
        # Archive of non-dominated points
        archive_X = []
        archive_F = []
        
        eval_count = 0
        max_evals = self.budget
        
        def is_dominated(f, F_archive):
            """Check if f is dominated by any point in F_archive."""
            if len(F_archive) == 0:
                return False
            F_arr = np.array(F_archive)
            # f is dominated if there exists f' such that f' <= f and f' != f
            # (minimization)
            dominated = np.any(np.all(F_arr <= f, axis=1) & np.any(F_arr < f, axis=1))
            return dominated

        def add_to_archive(x, f):
            """Add point to archive if non-dominated, remove dominated points."""
            non_local = True
            # Check if new point is dominated by existing archive
            for fa in archive_F:
                if np.all(fa <= f) and np.any(fa < f):
                    non_local = False
                    break
            
            if not non_local:
                return False
            
            # Remove points in archive dominated by new point
            new_archive_X = []
            new_archive_F = []
            for xa, fa in zip(archive_X, archive_F):
                # If fa is dominated by f, remove it
                if np.all(f <= fa) and np.any(f < fa):
                    continue
                new_archive_X.append(xa)
                new_archive_F.append(fa)
            
            new_archive_X.append(x)
            new_archive_F.append(f)
            
            archive_X[:] = new_archive_X
            archive_F[:] = new_archive_F
            return True

        def local_search(x, f):
            """
            Perform local search on positioning variables.
            Uses Nelder-Mead to minimize a scalar function: 
            max(f_i / ref_i) or sum(f_i). 
            Here we use a weighted sum with weights based on current objective values 
            to approximate a scalarization that pushes towards the ideal point.
            """
            # Create a copy to modify
            x_local = x.copy()
            f_current = f.copy()
            
            # We optimize the positioning variables (indices 2..11)
            # Objective: minimize the maximum normalized objective value (Tchebycheff-like)
            # Reference point for normalization: [3.0, 5.0, 7.0] (from task description)
            ref = np.array([3.0, 5.0, 7.0])
            
            def scalar_objective(pos_vars):
                nonlocal eval_count
                if time.time() > deadline:
                    return 1e9
                if eval_count >= max_evals:
                    return 1e9
                
                x_temp = x_local.copy()
                x_temp[positioning_dims] = pos_vars
                
                # Clip to bounds
                x_temp = np.clip(x_temp, lo, hi)
                
                # Evaluate
                f_temp = np.array(problem(x_temp))
                eval_count += 1
                
                # Tchebycheff scalarization: max(f_i / ref_i)
                # This helps in multi-objective context to find points close to the front
                scalar = np.max(f_temp / ref)
                return scalar
            
            # Initial guess for positioning variables
            x0 = x_local[positioning_dims]
            
            # Run Nelder-Mead
            # Options: maxiter to limit evaluations per local search
            # Since we have ~10 positioning dims, NM might use many evals.
            # Limit to ~10-20 evals per restart to allow multiple restarts.
            max_iter_local = 15
            try:
                res = minimize(
                    scalar_objective, 
                    x0, 
                    method='Nelder-Mead', 
                    options={'maxiter': max_iter_local, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                if res.success or res.fun < np.max(f_current / ref):
                    # Update x_local with new positioning vars
                    x_local[positioning_dims] = res.x
                    # Re-evaluate to get consistent f (already counted in scalar_objective, 
                    # but we need the vector f for archive)
                    # Note: The last evaluation in scalar_objective was for res.x
                    # We can just use res.x to update x_local and evaluate once more 
                    # or trust the last call. To be safe and accurate for archive:
                    x_final = np.clip(x_local, lo, hi)
                    f_final = np.array(problem(x_final))
                    eval_count += 1
                    return x_final, f_final
            except Exception:
                pass
            
            return x_local, f_current

        # Main Loop: Random Restart Local Search
        # 1. Generate random point
        # 2. Local search on positioning vars
        # 3. Add to archive
        
        # Initial random population to start
        n_initial = 10
        for i in range(n_initial):
            if eval_count >= max_evals or time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.rand(self.dim)
            f = np.array(problem(x))
            eval_count += 1
            # Perform local search
            x, f = local_search(x, f)
            add_to_archive(x, f)
        
        # Continue with random restarts until budget or time is up
        while eval_count < max_evals and time.time() < deadline:
            # Random restart: new random deciding variables, random positioning
            x = lo + (hi - lo) * np.random.rand(self.dim)
            f = np.array(problem(x))
            eval_count += 1
            
            # Local search to push towards front
            x, f = local_search(x, f)
            
            # Add to archive
            add_to_archive(x, f)
            
            # Optional: If we have time and budget, do a second pass of local search 
            # on the best points in the archive? 
            # For now, keep it simple: RRLS.
            
            # Check if we are stuck (optional)
            if len(archive_X) > 50:
                # We have a good spread, maybe stop early if no improvement in last 20 restarts?
                # Keeping it simple for now to avoid complexity bugs.
                pass

        # Return the archive (or a subset)
        # The evaluator records ALL returned objective vectors.
        # We should return the non-dominated points.
        if len(archive_X) == 0:
            # Fallback: return empty or a dummy? 
            # The task says "Every returned objective vector is recorded".
            # If we return an array of shape (0, n_obj), it might be okay.
            # But usually we return a numpy array of points.
            return np.array([])
        
        return np.array(archive_X)

# EVOLVE-BLOCK-END
