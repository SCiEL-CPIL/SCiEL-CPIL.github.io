import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        dim = self.dim
        total_budget = self.budget
        
        # Strategy: Bilevel Adaptive Search
        # Phase 1: Scout. Use ~20% of budget to find a good basin.
        # Phase 2: Refine. Use remaining ~80% to converge to the local minimum of that basin.
        # Rationale: f22 has 21 peaks. NM is excellent at navigating the anisotropic valleys 
        # once inside a basin, but poor at global search. Scout identifies the basin, NM exploits it.
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Scout
        # We will do a few random starts, each followed by a short NM run.
        # Let's allocate 100 calls to scouting (20% of 500).
        scout_budget = 100
        n_scouts = 10
        budget_per_scout = scout_budget // n_scouts
        
        global_best_x = None
        global_best_f = float('inf')
        
        for i in range(n_scouts):
            if time.time() > deadline:
                break
            
            # Random start
            x = rng.uniform(-4.5, 4.5, dim)
            
            # Short Nelder-Mead run
            # We implement a lightweight NM here to keep overhead low
            # Or we can just use a few steps of a simple local search to save code complexity?
            # No, NM is key. Let's implement a compact NM.
            
            # Initialize simplex
            simplex = [x.copy() for _ in range(dim + 1)]
            # Perturb first dimension
            simplex[0][0] += 0.5
            
            # Evaluate initial simplex
            f_simplex = np.zeros(dim + 1)
            calls_used = 0
            for j in range(dim + 1):
                f_simplex[j] = func(simplex[j])
                calls_used += 1
                if f_simplex[j] < global_best_f:
                    global_best_f = f_simplex[j]
                    global_best_x = simplex[j].copy()
            
            # Run NM for remaining budget of this scout
            max_nm_iter = budget_per_scout - (dim + 1)
            
            for k in range(max_nm_iter):
                if time.time() > deadline:
                    break
                if calls_used >= scout_budget:
                    break
                    
                # Sort simplex
                idx = np.argsort(f_simplex)
                simplex = [simplex[i] for i in idx]
                f_simplex = f_simplex[idx]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls_used += 1
                
                if f_r < f_simplex[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls_used += 1
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    rho = 0.5
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_used += 1
                    
                    if f_c < f_simplex[-1]:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for j in range(1, dim + 1):
                            simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                            f_simplex[j] = func(simplex[j])
                            calls_used += 1
                            if calls_used >= scout_budget:
                                break
                        
                        if calls_used >= scout_budget:
                            break
                
                # Update global best
                for j in range(dim + 1):
                    if f_simplex[j] < global_best_f:
                        global_best_f = f_simplex[j]
                        global_best_x = simplex[j].copy()
                
                # Check convergence
                if np.max(f_simplex) - np.min(f_simplex) < 1e-10:
                    break
        
        # Phase 2: Refine
        # Use the best point found in Phase 1 as the starting point for a long NM run.
        # Remaining budget: total_budget - calls_used (but we should reserve some for safety)
        # Actually, we should just use the remaining time/budget for a single long NM run.
        
        if global_best_x is None:
            global_best_x = np.zeros(dim)
            global_best_f = func(global_best_x)
            
        # Initialize NM simplex around global_best_x
        x0 = global_best_x.copy()
        
        simplex = [x0.copy() for _ in range(dim + 1)]
        # Standard NM initialization
        for j in range(1, dim + 1):
            simplex[j] = x0.copy()
            simplex[j][j-1] += 0.5 # Step size 0.5 is reasonable for [-5,5]
            
        f_simplex = np.zeros(dim + 1)
        for j in range(dim + 1):
            f_simplex[j] = func(simplex[j])
            if f_simplex[j] < global_best_f:
                global_best_f = f_simplex[j]
                global_best_x = simplex[j].copy()
        
        # Run NM until budget or time runs out
        while time.time() < deadline:
            # Sort simplex
            idx = np.argsort(f_simplex)
            simplex = [simplex[i] for i in idx]
            f_simplex = f_simplex[idx]
            
            # Centroid
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            alpha = 1.0
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            
            if f_r < f_simplex[0]:
                # Expansion
                gamma = 2.0
                x_e = centroid + gamma * (centroid - simplex[-1])
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_simplex[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
            elif f_r < f_simplex[-2]:
                simplex[-1] = x_r
                f_simplex[-1] = f_r
            else:
                # Contraction
                rho = 0.5
                x_c = centroid + rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                
                if f_c < f_simplex[-1]:
                    simplex[-1] = x_c
                    f_simplex[-1] = f_c
                else:
                    # Shrink
                    for j in range(1, dim + 1):
                        simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                        f_simplex[j] = func(simplex[j])
                        
                        if time.time() > deadline:
                            break
                    
                    if time.time() > deadline:
                        break
            
            # Update global best
            for j in range(dim + 1):
                if f_simplex[j] < global_best_f:
                    global_best_f = f_simplex[j]
                    global_best_x = simplex[j].copy()
            
            # Check convergence
            if np.max(f_simplex) - np.min(f_simplex) < 1e-12:
                break
            
            # Safety break if no progress? NM usually converges.
            # We rely on the time deadline.
            
        return global_best_x, global_best_f
