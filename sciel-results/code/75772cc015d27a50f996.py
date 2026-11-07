import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin from 300s limit
        
        x = np.zeros(self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Helper to check time and budget
        def check_time():
            return time.time() > deadline

        def check_budget():
            return calls >= self.budget

        # Phase 1: Nelder-Mead
        # Initialize Simplex
        # x0 is center, x0 + delta*e_i are vertices
        delta = 0.5
        simplex = [x.copy()]
        for i in range(self.dim):
            xi = x.copy()
            xi[i] += delta
            simplex.append(xi)
        
        # Evaluate simplex
        simplex_f = [fx]
        for i in range(1, self.dim + 1):
            if check_time() or check_budget():
                break
            fi = func(simplex[i])
            simplex_f.append(fi)
            calls += 1
            if fi < best_fx:
                best_fx = fi
                best_x = simplex[i].copy()

        # Nelder-Mead Parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage
        
        # Run NM
        while not (check_time() or check_budget()):
            # Sort simplex by fitness
            indices = np.argsort(simplex_f)
            simplex = [simplex[i] for i in indices]
            simplex_f = [simplex_f[i] for i in indices]
            
            # Check convergence: if best and worst are very close, break
            if abs(simplex_f[0] - simplex_f[-1]) < 1e-10:
                break
                
            # Centroid of all but the worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_reflect = centroid + alpha * (centroid - simplex[-1])
            x_reflect = np.clip(x_reflect, -5, 5)
            f_reflect = func(x_reflect)
            calls += 1
            if f_reflect < best_fx:
                best_fx = f_reflect
                best_x = x_reflect.copy()
            
            if check_time() or check_budget():
                break

            # Expansion
            if f_reflect < simplex_f[0]:
                x_expand = centroid + gamma * (x_reflect - centroid)
                x_expand = np.clip(x_expand, -5, 5)
                f_expand = func(x_expand)
                calls += 1
                if f_expand < best_fx:
                    best_fx = f_expand
                    best_x = x_expand.copy()
                
                if f_expand < f_reflect:
                    simplex[-1] = x_expand
                    simplex_f[-1] = f_expand
                else:
                    simplex[-1] = x_reflect
                    simplex_f[-1] = f_reflect
            
            # Contraction
            elif f_reflect < simplex_f[-2]:
                if f_reflect < simplex_f[-1]:
                    # Outside contraction
                    x_contract = centroid + rho * (x_reflect - centroid)
                else:
                    # Inside contraction
                    x_contract = centroid - rho * (x_reflect - centroid)
                
                x_contract = np.clip(x_contract, -5, 5)
                f_contract = func(x_contract)
                calls += 1
                if f_contract < best_fx:
                    best_fx = f_contract
                    best_x = x_contract.copy()
                
                if f_contract < f_reflect:
                    simplex[-1] = x_contract
                    simplex_f[-1] = f_contract
                else:
                    simplex[-1] = x_reflect
                    simplex_f[-1] = f_reflect
            
            # Shrink
            else:
                for i in range(1, self.dim + 1):
                    if check_time() or check_budget():
                        break
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex_f[i] = func(simplex[i])
                    calls += 1
                    if simplex_f[i] < best_fx:
                        best_fx = simplex_f[i]
                        best_x = simplex[i].copy()
                
                # If budget ran out during shrink, we might have updated simplex_f partially
                # We just break the loop
                if check_time() or check_budget():
                    break

        # Phase 2: Adaptive Coordinate Descent
        # Start from best_x found so far
        x_cd = best_x.copy()
        f_cd = best_fx
        
        # Initial step sizes
        steps = np.ones(self.dim) * 0.1
        
        # Cyclic CD
        # We will iterate through coordinates multiple times
        # Limit iterations to avoid infinite loop if step doesn't shrink fast enough
        max_cd_iterations = 1000 # Internal loop limit, but governed by budget/time
        
        for _ in range(max_cd_iterations):
            if check_time() or check_budget():
                break
            
            improved = False
            for i in range(self.dim):
                if check_time() or check_budget():
                    break
                
                # Try positive step
                x_trial = x_cd.copy()
                x_trial[i] += steps[i]
                x_trial = np.clip(x_trial, -5, 5)
                
                f_trial = func(x_trial)
                calls += 1
                
                if f_trial < f_cd:
                    x_cd = x_trial
                    f_cd = f_trial
                    if f_cd < best_fx:
                        best_fx = f_cd
                        best_x = x_cd.copy()
                    steps[i] *= 1.2  # Increase step if successful
                    improved = True
                else:
                    # Try negative step
                    x_trial = x_cd.copy()
                    x_trial[i] -= steps[i]
                    x_trial = np.clip(x_trial, -5, 5)
                    
                    f_trial = func(x_trial)
                    calls += 1
                    
                    if f_trial < f_cd:
                        x_cd = x_trial
                        f_cd = f_trial
                        if f_cd < best_fx:
                            best_fx = f_cd
                            best_x = x_cd.copy()
                        steps[i] *= 1.2
                        improved = True
                    else:
                        # No improvement, shrink step
                        steps[i] *= 0.5
                        
            # If no coordinate improved in a full cycle, and steps are very small, stop
            if not improved:
                if np.all(steps < 1e-6):
                    break
                # Else, maybe try a different starting point? No, just break to save budget
                # Or continue with smaller steps?
                # If steps are not small, maybe we are stuck in a local min or flat area.
                # For f6, if we are close, steps will shrink.
                # If we are far, NM should have moved us closer.
                # If we are stuck, break.
                break

        return best_x, best_fx
