"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-start Nelder-Mead optimizer with adaptive restarts for BBOB f19 (Griewank-Rosenbrock).
    Strategy:
    1. Use diverse starting points (center + random + structured grid).
    2. Run Nelder-Mead simplex from each starting point.
    3. Adaptive simplex size based on progress.
    4. Final polish pass with tighter tolerances on best found point.
    5. Early progress is maximized by quickly converging on good basins.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 300s limit, keep 30s safety margin
        deadline = start_time + 270.0
        
        dim = self.dim
        budget = self.budget
        
        global_best_x = None
        global_best_f = float('inf')
        calls_made = 0
        
        def call_func(x):
            nonlocal calls_made, global_best_x, global_best_f
            if calls_made >= budget:
                return global_best_f
            val = func(x)
            calls_made += 1
            if val < global_best_f:
                global_best_f = val
                global_best_x = x.copy()
            return val

        def check_time():
            return time.time() > deadline or calls_made >= budget

        # --- Phase 1: Generate diverse starting points ---
        starts = []
        
        # Center
        starts.append(np.zeros(dim))
        
        # Structured grid points (corners of sub-boxes)
        # For dim=5, we can use a few structured points
        # Use points at ±2.5 in each coordinate (mid-range)
        for sign in [1, -1]:
            starts.append(sign * 2.5 * np.ones(dim))
        
        # Random points - use more for better coverage
        num_random_starts = 12
        for _ in range(num_random_starts):
            x_rand = np.random.uniform(-5, 5, dim)
            starts.append(x_rand)
        
        # Shuffle starts to mix structured and random
        np.random.shuffle(starts)
        
        # --- Phase 2: Nelder-Mead from each start ---
        nm_results = []  # Store (best_f, best_x) from each NM run
        
        for start_idx, start_x in enumerate(starts):
            if check_time():
                break
            
            # Initialize simplex with adaptive delta
            # Start with larger delta for exploration, smaller for exploitation
            if start_idx < 3:
                delta = 1.0  # Larger for early starts
            elif start_idx < 8:
                delta = 0.5
            else:
                delta = 0.25
            
            simplex = [start_x.copy()]
            for i in range(dim):
                v = start_x.copy()
                v[i] += delta
                simplex.append(v)
            
            # Evaluate initial simplex
            fvals = []
            for v in simplex:
                if check_time():
                    break
                fvals.append(call_func(v))
            
            if check_time():
                break
                
            # Ensure we have dim+1 values
            if len(fvals) < dim + 1:
                continue
            
            # Nelder-Mead Loop
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            
            max_iter = 150  # More iterations per NM run
            local_best_f = fvals[0]
            local_best_x = np.array(simplex[0])
            
            for iter_i in range(max_iter):
                if check_time():
                    break
                
                # Sort simplex by f-values
                indices = np.argsort(fvals)
                simplex = [simplex[i] for i in indices]
                fvals = [fvals[i] for i in indices]
                
                # Track local best
                if fvals[0] < local_best_f:
                    local_best_f = fvals[0]
                    local_best_x = simplex[0].copy()
                
                # Check convergence
                f_range = fvals[-1] - fvals[0]
                if f_range < 1e-12:
                    break
                
                # Adaptive stopping: if simplex is very small
                simplex_size = np.max(np.linalg.norm(np.array(simplex) - np.mean(simplex, axis=0), axis=1))
                if simplex_size < 1e-8:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = call_func(x_ref)
                
                if check_time():
                    break
                
                if fvals[0] <= f_ref < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                elif f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = call_func(x_exp)
                    
                    if check_time():
                        break
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < fvals[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (centroid - simplex[-1])
                    
                    x_con = np.clip(x_con, -5, 5)
                    f_con = call_func(x_con)
                    
                    if check_time():
                        break
                    
                    if f_con < min(f_ref, fvals[-1]):
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = call_func(simplex[i])
                        
                        if check_time():
                            break
            
            nm_results.append((local_best_f, local_best_x.copy()))
        
        # --- Phase 3: Final polish on the best found point ---
        if nm_results and not check_time():
            # Find the best result
            best_idx = np.argmin([r[0] for r in nm_results])
            best_x = nm_results[best_idx][1]
            best_f = nm_results[best_idx][0]
            
            # Run a tight Nelder-Mead polish
            polish_delta = 0.1
            simplex = [best_x.copy()]
            for i in range(dim):
                v = best_x.copy()
                v[i] += polish_delta
                simplex.append(v)
            
            fvals = []
            for v in simplex:
                if check_time():
                    break
                fvals.append(call_func(v))
            
            if not check_time() and len(fvals) == dim + 1:
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma = 0.5
                
                max_polish_iter = 200
                for _ in range(max_polish_iter):
                    if check_time():
                        break
                    
                    indices = np.argsort(fvals)
                    simplex = [simplex[i] for i in indices]
                    fvals = [fvals[i] for i in indices]
                    
                    if fvals[-1] - fvals[0] < 1e-14:
                        break
                    
                    simplex_size = np.max(np.linalg.norm(np.array(simplex) - np.mean(simplex, axis=0), axis=1))
                    if simplex_size < 1e-10:
                        break
                    
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    x_ref = centroid + alpha * (centroid - simplex[-1])
                    x_ref = np.clip(x_ref, -5, 5)
                    f_ref = call_func(x_ref)
                    
                    if check_time():
                        break
                    
                    if fvals[0] <= f_ref < fvals[-2]:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                    elif f_ref < fvals[0]:
                        x_exp = centroid + gamma * (x_ref - centroid)
                        x_exp = np.clip(x_exp, -5, 5)
                        f_exp = call_func(x_exp)
                        
                        if check_time():
                            break
                        
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            fvals[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            fvals[-1] = f_ref
                    else:
                        if f_ref < fvals[-1]:
                            x_con = centroid + rho * (x_ref - centroid)
                        else:
                            x_con = centroid - rho * (centroid - simplex[-1])
                        
                        x_con = np.clip(x_con, -5, 5)
                        f_con = call_func(x_con)
                        
                        if check_time():
                            break
                        
                        if f_con < min(f_ref, fvals[-1]):
                            simplex[-1] = x_con
                            fvals[-1] = f_con
                        else:
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fvals[i] = call_func(simplex[i])
                            
                            if check_time():
                                break
        
        # Ensure global_best is set
        if global_best_x is None:
            global_best_x = np.zeros(dim)
            global_best_f = func(global_best_x)
        
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
