import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 270s of 330s limit
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

        # Phase 1: Coarse Exploration
        # Use 10% of budget for random sampling to find good starting points
        explore_calls = max(10, budget // 10)
        candidates = []
        
        # Always include 0
        x0 = np.zeros(dim)
        f0 = call_func(x0)
        candidates.append((f0, x0.copy()))
        
        while calls_made < explore_calls and time.time() < deadline:
            x = np.random.uniform(-5, 5, dim)
            f = call_func(x)
            candidates.append((f, x.copy()))
        
        # Sort candidates by fitness
        candidates.sort(key=lambda x: x[0])
        
        # Select top K diverse candidates for Nelder-Mead
        # We want to start NM from points that are not too close to each other
        # to cover different basins if possible, but primarily focus on the best ones.
        # Let's take the top 3 distinct-ish points.
        # Simple diversity: if distance < 0.5, skip (optional, but helps if landscape is multimodal)
        # For f19 (Griewank-Rosenbrock), it's mostly unimodal but rotated. 
        # Starting from the best few is usually sufficient.
        
        starts = []
        min_dist = 1.0  # Threshold for diversity
        for f, x in candidates:
            if len(starts) >= 3:
                break
            is_diverse = True
            for _, sx in starts:
                if np.linalg.norm(x - sx) < min_dist:
                    is_diverse = False
                    break
            if is_diverse:
                starts.append((f, x))
        
        # If we didn't get enough diverse starts, just take the best ones
        if len(starts) < 3:
            for f, x in candidates:
                if (f, x) not in starts: # Rough check, tuples of arrays don't work well in 'in', so check manually
                    if len(starts) >= 3: break
                    # Check distance again properly
                    diverse = True
                    for _, sx in starts:
                        if np.linalg.norm(x - sx) < min_dist:
                            diverse = False
                            break
                    if diverse:
                        starts.append((f, x))
        
        # If still empty (shouldn't happen), add best
        if not starts:
            starts.append(candidates[0])

        # Phase 2: Nelder-Mead Refinement
        # Use remaining budget for sequential NM runs
        remaining_budget = budget - calls_made
        num_runs = len(starts)
        if num_runs == 0:
            num_runs = 1
            starts.append(candidates[0])
        
        budget_per_run = remaining_budget // num_runs
        
        # Nelder-Mead Parameters
        # Standard defaults adjusted for dimension
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink
        
        for i in range(num_runs):
            if time.time() >= deadline:
                break
            if calls_made >= budget:
                break
            
            # Initialize Simplex
            # Start from the candidate point
            x_center = starts[i][1].copy()
            
            # Create simplex: x_center + step * e_j
            # Step size: initial step for NM. 
            # Since we explored, we can use a smaller step than 1.0, maybe 0.1 or 0.5?
            # 0.5 is reasonable for [-5,5] box.
            step = 0.5
            
            simplex = [x_center.copy()]
            for j in range(dim):
                x_j = x_center.copy()
                x_j[j] += step
                simplex.append(x_j)
            
            # Evaluate initial simplex
            simplex_f = [call_func(x) for x in simplex]
            
            # Check if we already used up budget
            if calls_made >= budget:
                break
            
            # NM Loop
            while calls_made < budget and time.time() < deadline:
                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Convergence check: if spread is small
                # Calculate distance between best and worst
                # Or check gradient norm approximation
                # Simple check: if f_worst - f_best < 1e-10
                if simplex_f[-1] - simplex_f[0] < 1e-10:
                    break
                
                # Calculate centroid of all points except worst
                centroid = np.sum(np.array(simplex[:-1]), axis=0) / dim
                
                # 1. Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = call_func(x_r)
                
                if calls_made >= budget:
                    break
                
                if f_r < simplex_f[0]:
                    # 2. Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = call_func(x_e)
                    
                    if calls_made >= budget:
                        break
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # 3. Contraction
                    # Outside or Inside contraction? Standard is Inside if f_r > f_worst
                    if f_r < simplex_f[-1]:
                        # Inside Contraction
                        x_c = centroid + sigma * (x_r - centroid)
                    else:
                        # Outside Contraction
                        x_c = centroid - sigma * (simplex[-1] - centroid)
                    
                    f_c = call_func(x_c)
                    
                    if calls_made >= budget:
                        break
                    
                    if f_c < min(f_r, simplex_f[-1]):
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # 4. Shrink
                        # Move all points except best towards best
                        for k in range(1, dim + 1):
                            simplex[k] = simplex[0] + sigma * (simplex[k] - simplex[0])
                            simplex_f[k] = call_func(simplex[k])
                            
                            if calls_made >= budget:
                                break
                            if time.time() >= deadline:
                                break
            
            # After NM run, the best point in this simplex might be the global best
            # global_best_x and global_best_f are updated in call_func
            
        return global_best_x, global_best_f
