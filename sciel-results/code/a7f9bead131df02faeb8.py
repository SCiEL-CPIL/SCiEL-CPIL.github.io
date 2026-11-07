import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        
        budget = self.budget
        dim = self.dim
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        def evaluate(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            try:
                val = func(x)
            except Exception:
                return None
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # --- Phase 1: Coordinate Descent with Adaptive Step Sizes ---
        # This exploits the ridge structure of f13 by searching along axes sequentially.
        # It is fast and efficient for separating the "easy" dimensions from the ridge.
        
        # Initialize at center (0,0,0,0,0) which is often a good starting point for BBOB
        x = np.zeros(dim)
        fx = evaluate(x)
        if fx is None:
            return best_x, best_f
            
        # Step sizes for each dimension
        step_sizes = np.ones(dim) * 1.0
        
        # We will use a portion of the budget for coordinate descent.
        # Since CD is fast (1 eval per step), we can afford many iterations.
        # Let's use ~200 evaluations for CD.
        cd_budget = min(200, budget - 1)
        
        # Track best improvement to detect stagnation
        best_cd_f = fx
        no_improve_count = 0
        max_no_improve = 20
        
        for _ in range(cd_budget):
            if calls >= budget or time.time() > deadline:
                break
            
            improved = False
            for i in range(dim):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Try moving in positive direction
                x_trial = x.copy()
                x_trial[i] += step_sizes[i]
                x_trial = np.clip(x_trial, -5, 5)
                
                f_trial = evaluate(x_trial)
                if f_trial is None:
                    break
                
                if f_trial < fx:
                    x = x_trial
                    fx = f_trial
                    improved = True
                    # Increase step size slightly if we improved significantly?
                    # Or keep it constant? For sharp ridge, keeping it or growing slowly is good.
                    step_sizes[i] *= 1.2
                else:
                    # Try moving in negative direction
                    x_trial = x.copy()
                    x_trial[i] -= step_sizes[i]
                    x_trial = np.clip(x_trial, -5, 5)
                    
                    f_trial = evaluate(x_trial)
                    if f_trial is None:
                        break
                    
                    if f_trial < fx:
                        x = x_trial
                        fx = f_trial
                        improved = True
                        step_sizes[i] *= 1.2
                    else:
                        # No improvement in either direction, shrink step size
                        step_sizes[i] *= 0.7
                        # If step size is too small, it's effectively stuck
                        if step_sizes[i] < 1e-3:
                            step_sizes[i] = 1e-3
                
                if time.time() > deadline:
                    break
            
            if not improved:
                no_improve_count += 1
                # If we haven't improved for a while, we might be stuck in a local minimum
                # or the step sizes are too small. 
                # For now, we just continue. If we hit max_no_improve, we break out of CD.
                if no_improve_count >= max_no_improve:
                    break
            else:
                no_improve_count = 0
                if fx < best_cd_f:
                    best_cd_f = fx

        # --- Phase 2: Nelder-Mead Simplex Method ---
        # Now we use the remaining budget to refine the solution using Nelder-Mead.
        # Nelder-Mead is robust and handles the complex geometry of the ridge well.
        # We initialize the simplex around the best point found by CD.
        
        if calls >= budget or time.time() > deadline:
            return best_x, best_f
            
        # Start NM from the current best point
        x_start = x.copy()
        
        # Create initial simplex: center plus points along each axis
        # The initial spread should be small enough to stay near the CD result
        # but large enough to allow movement.
        initial_spread = 0.5
        
        simplex = []
        simplex.append(x_start.copy())  # Center (best point)
        for i in range(dim):
            point = x_start.copy()
            point[i] += initial_spread
            point = np.clip(point, -5, 5)
            simplex.append(point)
        
        # Evaluate all simplex vertices
        simplex_f = []
        for point in simplex:
            # If the point is already evaluated (the center), we should ideally reuse it,
            # but since we don't have a cache, we just re-evaluate.
            # To save calls, we can check if it's close to x_start.
            # Actually, x_start is x, which we just evaluated as fx.
            # But the evaluate function doesn't know that.
            # Let's just call evaluate. It's only 5 calls.
            fx_i = evaluate(point)
            if fx_i is None:
                return best_x, best_f
            simplex_f.append(fx_i)
        
        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma_nm = 0.5  # Shrink
        
        # Run Nelder-Mead until budget or deadline is reached
        while calls < budget and time.time() < deadline:
            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            # Check convergence: if the spread is very small, break
            if np.max(simplex_f) - np.min(simplex_f) < 1e-10:
                break
            
            # Compute centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_reflect = centroid + alpha * (centroid - simplex[-1])
            x_reflect = np.clip(x_reflect, -5, 5)
            f_reflect = evaluate(x_reflect)
            if f_reflect is None:
                break
            
            if f_reflect < simplex_f[0]:
                # Try expansion
                x_expand = centroid + gamma * (x_reflect - centroid)
                x_expand = np.clip(x_expand, -5, 5)
                f_expand = evaluate(x_expand)
                if f_expand is None:
                    break
                
                if f_expand < f_reflect:
                    simplex[-1] = x_expand
                    simplex_f[-1] = f_expand
                else:
                    simplex[-1] = x_reflect
                    simplex_f[-1] = f_reflect
            elif f_reflect < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = x_reflect
                simplex_f[-1] = f_reflect
            else:
                # Contraction
                if f_reflect < simplex_f[-1]:
                    # Outside contraction
                    x_contract = centroid + rho * (x_reflect - centroid)
                else:
                    # Inside contraction
                    x_contract = centroid - rho * (x_reflect - centroid)
                
                x_contract = np.clip(x_contract, -5, 5)
                f_contract = evaluate(x_contract)
                if f_contract is None:
                    break
                
                if f_contract < min(f_reflect, simplex_f[-1]):
                    simplex[-1] = x_contract
                    simplex_f[-1] = f_contract
                else:
                    # Shrink: move all points toward the best
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_new = evaluate(simplex[i])
                        if f_new is None:
                            return best_x, best_f
                        simplex_f[i] = f_new
            
            # Check if we've hit the deadline
            if time.time() > deadline:
                break
        
        return best_x, best_f
