import numpy as np
import time

class Optimizer:
    """
    Optimizer using Nelder-Mead with restarts for BBOB f13 (Sharp Ridge).
    Nelder-Mead is robust for low-dimensional problems and handles ridges well
    when combined with restarts to escape local minima.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0 # Safety margin for 330s limit

        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                x_clipped = np.clip(x, -5.0, 5.0)
                f_val = func(x_clipped)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Global Exploration
        # Sample points to find the basin of attraction.
        n_random = 10
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            f_val = safe_call(x)
            if f_val is None:
                break

        if best_x is None:
            # Fallback to origin if no valid call was made
            x0 = np.zeros(dim)
            f0 = safe_call(x0)
            if f0 is None:
                return best_x, best_f
            best_x = x0
            best_f = f0

        # Phase 2: Nelder-Mead with Restarts
        # We will run Nelder-Mead, and if it stagnates, restart from the best point.
        
        def nelder_mead(x_start, f_start, max_iter=50):
            nonlocal calls, best_f, best_x
            
            # Initialize simplex
            # Start with x_start and small perturbations
            alpha = 0.1
            simplex = np.zeros((dim + 1, dim))
            f_values = np.zeros(dim + 1)
            
            simplex[0] = x_start
            f_values[0] = f_start
            
            for i in range(1, dim + 1):
                x_i = x_start.copy()
                x_i[i-1] += alpha
                # Ensure within bounds
                x_i = np.clip(x_i, -5.0, 5.0)
                simplex[i] = x_i
                f_val = safe_call(x_i)
                if f_val is None:
                    return
                f_values[i] = f_val
            
            # Sort simplex by function value
            def sort_simplex():
                nonlocal simplex, f_values
                idx = np.argsort(f_values)
                simplex = simplex[idx]
                f_values = f_values[idx]
            
            sort_simplex()
            
            # Nelder-Mead Parameters
            alpha_nm = 1.0  # Reflection
            gamma = 2.0     # Expansion
            rho = 0.5       # Contraction
            sigma = 0.5     # Shrink
            
            for it in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    return
                
                # Check convergence
                if np.max(f_values) - np.min(f_values) < 1e-8:
                    break
                
                # Find centroid of the best dim vertices (excluding the worst)
                centroid = np.mean(simplex[:dim], axis=0)
                
                # Reflection
                x_r = centroid + alpha_nm * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = safe_call(x_r)
                if f_r is None:
                    return
                
                if f_r < f_values[0]:
                    # Check for Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = safe_call(x_e)
                    if f_e is None:
                        return
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    # Reflection is good but not the best
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    # Contraction
                    # Outside contraction if f_r < f_values[-1], else inside
                    if f_r < f_values[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = safe_call(x_c)
                    if f_c is None:
                        return
                    
                    if f_c < f_values[-1]:
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_val = safe_call(simplex[i])
                            if f_val is None:
                                return
                            f_values[i] = f_val
                
                sort_simplex()
                
                # Update global best
                if f_values[0] < best_f:
                    best_f = f_values[0]
                    best_x = simplex[0].copy()

        # Run Nelder-Mead with restarts
        # We allocate budget for several Nelder-Mead runs.
        # Each run should use a portion of the remaining budget.
        # Simple strategy: Run Nelder-Mead until it converges or uses ~50% of remaining budget, then restart if improved.
        
        # To be safe and simple, we'll just run a single long Nelder-Mead but with a restart mechanism inside if it stagnates.
        # However, implementing a full restart loop inside NM is complex.
        # Instead, we will do a loop of "NM iterations" where we restart if no improvement for K iterations.
        
        # Let's modify the approach: 
        # 1. Start NM from best_x.
        # 2. If NM stops (converged or budget), check if we have budget left.
        # 3. If yes, perturb best_x and restart NM.
        
        restart_count = 0
        max_restarts = 5
        
        while calls < budget and time.time() < deadline and restart_count < max_restarts:
            # Create a new starting point for this restart
            if restart_count == 0:
                x_start = best_x.copy()
                f_start = best_f
            else:
                # Perturb the best point found so far
                x_start = best_x + rng.normal(0, 0.5, dim)
                x_start = np.clip(x_start, -5.0, 5.0)
                f_start = safe_call(x_start)
                if f_start is None:
                    break
            
            # Run Nelder-Mead for a limited number of iterations to allow restarts
            # We estimate iterations based on remaining budget.
            # Each NM iteration costs ~2-4 function evaluations on average.
            # Let's reserve 100 evaluations per NM run.
            if budget - calls < 20:
                break
                
            # We'll just call the internal NM loop logic.
            # Since the nested function `nelder_mead` doesn't easily expose iteration count or stop conditions for restarts,
            # we will implement a simpler loop here that calls a "step" or just runs a fixed number of iterations.
            
            # For simplicity and robustness, let's just run the standard NM loop defined above, 
            # but we need to break it if it stagnates.
            
            # Let's rewrite the NM loop to be more restart-friendly.
            # We'll use a local simplex and iterate.
            
            simplex = np.zeros((dim + 1, dim))
            f_values = np.zeros(dim + 1)
            
            simplex[0] = x_start
            f_values[0] = f_start
            
            for i in range(1, dim + 1):
                if calls >= budget: break
                x_i = x_start.copy()
                x_i[i-1] += 0.5 # Larger initial step for restarts
                x_i = np.clip(x_i, -5.0, 5.0)
                simplex[i] = x_i
                f_val = safe_call(x_i)
                if f_val is None:
                    break
                f_values[i] = f_val
                
            if calls >= budget:
                break
                
            def sort_simplex_local():
                nonlocal simplex, f_values
                idx = np.argsort(f_values)
                simplex = simplex[idx]
                f_values = f_values[idx]
                
            sort_simplex_local()
            
            alpha_nm = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            stagnation_counter = 0
            max_stagnation = 20 # Restart if no improvement for 20 iterations
            
            for it in range(100): # Max 100 iterations per restart phase
                if calls >= budget or time.time() > deadline:
                    break
                
                if np.max(f_values) - np.min(f_values) < 1e-8:
                    break
                
                centroid = np.mean(simplex[:dim], axis=0)
                
                x_r = centroid + alpha_nm * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = safe_call(x_r)
                if f_r is None:
                    break
                
                if f_r < f_values[0]:
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = safe_call(x_e)
                    if f_e is None:
                        break
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    if f_r < f_values[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = safe_call(x_c)
                    if f_c is None:
                        break
                    
                    if f_c < f_values[-1]:
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_val = safe_call(simplex[i])
                            if f_val is None:
                                break
                            f_values[i] = f_val
                        if calls >= budget:
                            break
                
                sort_simplex_local()
                
                # Update global best
                if f_values[0] < best_f:
                    best_f = f_values[0]
                    best_x = simplex[0].copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    
                if stagnation_counter > max_stagnation:
                    break

            restart_count += 1

        return best_x, best_f
