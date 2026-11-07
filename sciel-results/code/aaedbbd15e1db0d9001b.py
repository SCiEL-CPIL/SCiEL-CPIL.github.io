import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
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

        # Phase 1: (1+1)-Evolution Strategy for rapid initial convergence
        # This exploits the ridge structure and provides fast early progress for AOCC.
        x = np.zeros(dim)
        fx = evaluate(x)
        if fx is None:
            return best_x, best_f
            
        sigma = 1.0
        # Use a portion of the budget for ES. 
        # ES is very fast (1 eval per step). We can afford many iterations.
        # Let's use ~150 evaluations for ES to get close to the minimum quickly.
        es_budget = min(150, budget - 1)
        
        for _ in range(es_budget):
            if calls >= budget or time.time() > deadline:
                break
            
            # Generate trial point
            x_trial = x + sigma * np.random.randn(dim)
            x_trial = np.clip(x_trial, -5, 5)
            
            f_trial = evaluate(x_trial)
            if f_trial is None:
                break
            
            if f_trial < fx:
                # Success: move to trial, increase sigma
                x = x_trial
                fx = f_trial
                sigma *= 1.5
            else:
                # Failure: decrease sigma
                sigma *= 0.87
            
            # Safety check
            if time.time() > deadline:
                break

        # Phase 2: Nelder-Mead Simplex Method for refinement
        # NM handles the complex geometry of the ridge well and refines the solution.
        # Initialize simplex around the best point found by ES.
        
        if calls >= budget or time.time() > deadline:
            return best_x, best_f
            
        x_start = x.copy()
        
        # Create initial simplex: center plus points along each axis
        # The initial spread should be small enough to stay near the ES result
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
                        f_new = evaluate(simplex[i])
                        if f_new is None:
                            return best_x, best_f
                        simplex_f[i] = f_new
            
            # Check if we've hit the deadline
            if time.time() > deadline:
                break
        
        return best_x, best_f
