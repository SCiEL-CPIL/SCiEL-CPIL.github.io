import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_y = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def call_func(x):
            nonlocal calls_made, best_x, best_y
            if calls_made >= budget or not time_ok():
                return None
            x = np.clip(x, -5, 5)
            y = func(x)
            calls_made += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # Phase 1: Initial Sampling - diverse random points
        # Use a mix of random and structured points
        n_init = 10
        for _ in range(n_init):
            if not time_ok() or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, d)
            call_func(x)
        
        # If we have very little budget, just return best
        if calls_made >= budget or not time_ok():
            return best_x, best_y

        # Nelder-Mead implementation
        def nelder_mead(start_x, max_iter=100):
            nonlocal best_x, best_y, calls_made
            
            if best_x is None:
                return

            # Initialize simplex
            # Start with a simplex centered at start_x
            # Size of the simplex
            initial_step = 0.5 
            simplex = np.zeros((d + 1, d))
            simplex[0] = start_x
            
            for i in range(d):
                simplex[i+1] = start_x.copy()
                simplex[i+1][i] += initial_step
            
            # Evaluate initial simplex
            f_simplex = np.zeros(d + 1)
            for i in range(d + 1):
                f_val = call_func(simplex[i])
                if f_val is None:
                    return
                f_simplex[i] = f_val
            
            # Sort simplex by function value
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            # Nelder-Mead parameters
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrinkage
            
            for _ in range(max_iter):
                if not time_ok() or calls_made >= budget:
                    break
                
                # Update global best
                if f_simplex[0] < best_y:
                    best_y = f_simplex[0]
                    best_x = simplex[0].copy()
                
                # Calculate centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst_point = simplex[-1]
                worst_f = f_simplex[-1]
                
                # 1. Reflection
                xr = centroid + alpha * (centroid - worst_point)
                xr = np.clip(xr, -5, 5)
                fr = call_func(xr)
                if fr is None:
                    break
                
                if fr < f_simplex[0]:
                    # 2. Expansion
                    xe = centroid + gamma * (centroid - worst_point)
                    xe = np.clip(xe, -5, 5)
                    fe = call_func(xe)
                    if fe is None:
                        break
                    if fe < fr:
                        simplex[-1] = xe
                        f_simplex[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_simplex[-1] = fr
                elif fr < f_simplex[-2]:
                    # 3. Contraction (External)
                    xc = centroid + rho * (worst_point - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = call_func(xc)
                    if fc is None:
                        break
                    if fc < fr:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrinkage
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_val = call_func(simplex[i])
                            if f_val is None:
                                return
                            f_simplex[i] = f_val
                else:
                    # 4. Contraction (Internal)
                    xc = centroid - rho * (centroid - worst_point)
                    xc = np.clip(xc, -5, 5)
                    fc = call_func(xc)
                    if fc is None:
                        break
                    if fc < fr:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrinkage
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_val = call_func(simplex[i])
                            if f_val is None:
                                return
                            f_simplex[i] = f_val
                
                # Re-sort simplex
                order = np.argsort(f_simplex)
                simplex = simplex[order]
                f_simplex = f_simplex[order]
                
                # Convergence check
                if np.ptp(f_simplex) < 1e-10:
                    break

        # Phase 2: Main loop with Nelder-Mead restarts
        # We will use the remaining budget for multiple Nelder-Mead runs
        # Start from the best point found so far, and then from other good points
        
        # Keep track of best points for restarts
        best_points = [(best_y, best_x.copy())] if best_x is not None else []
        
        # Strategy: 
        # 1. Run NM from current best
        # 2. If budget remains, try NM from other diverse points or perturbations of best
        
        # First, run NM from the best point found in initial phase
        if best_x is not None and time_ok() and calls_made < budget:
            nelder_mead(best_x)
        
        # Now, loop while we have budget
        # We will generate new starting points based on:
        # - Perturbations of the global best
        # - Random points in the vicinity of the global best
        # - Occasionally, new random points to escape local minima
        
        attempts = 0
        while time_ok() and calls_made < budget:
            attempts += 1
            
            # Decide where to start the next NM
            r = rng.random()
            
            if r < 0.5 and best_x is not None:
                # Exploit: Perturb global best
                # Noise scale decreases over time
                noise_scale = 0.5 * np.exp(-calls_made / 50.0)
                start_x = best_x + rng.normal(0, noise_scale, d)
            elif r < 0.8 and len(best_points) > 1:
                # Semi-exploit: Perturb a good past point
                idx = rng.integers(1, len(best_points)) # Skip the very best to diversify
                start_x = best_points[idx][1] + rng.normal(0, 0.3, d)
            else:
                # Explore: Random point, but biased towards the center or best region
                if best_x is not None:
                    # Gaussian around best with larger variance
                    start_x = best_x + rng.normal(0, 1.5, d)
                else:
                    start_x = rng.uniform(-5, 5, d)
            
            start_x = np.clip(start_x, -5, 5)
            
            # Run Nelder-Mead
            nelder_mead(start_x)
            
            # Update best points list
            if best_x is not None:
                is_new = True
                for _, px in best_points:
                    if np.linalg.norm(best_x - px) < 0.05:
                        is_new = False
                        break
                if is_new:
                    best_points.insert(0, (best_y, best_x.copy()))
                    # Keep only top 5
                    if len(best_points) > 5:
                        best_points = best_points[:5]
                else:
                    # Update the existing point if it improved
                    for i, (fy, px) in enumerate(best_points):
                        if np.linalg.norm(best_x - px) < 0.05:
                            best_points[i] = (best_y, best_x.copy())
                            break
        
        return best_x, best_y
