import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 280.0  # Safety margin against 300s/330s limit
        
        dim = self.dim
        budget = self.budget
        
        # Fixed seed for reproducibility
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        eval_count = 0
        
        def time_check():
            return (time.time() - start_time) > timeout

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Strategy:
        # 1. Coarse Exploration: Use ~20% of budget for a stratified random search (LHS-like)
        #    to find good starting points. This is crucial because the "Step Ellipsoidal"
        #    function has large flat regions. Random sampling helps escape these or find
        #    the correct "step" level.
        # 2. Local Refinement: Take the top K candidates from exploration and run
        #    Nelder-Mead on each, distributing the remaining budget.
        
        # Phase 1: Exploration
        # We want to evaluate enough points to get a decent spread.
        # For dim=5, 50 points is a good start.
        n_explore = max(10, budget // 10) 
        # Cap exploration to ensure we have time for optimization
        n_explore = min(n_explore, 50)
        
        # Generate Latin Hypercube points for better coverage than pure random
        # LHS ensures each dimension is divided into n_explore intervals, and one point 
        # is placed in each interval per dimension.
        points = []
        for i in range(n_explore):
            # For each dimension, pick a random position in the i-th bin
            # Bin i is [i/n, (i+1)/n] mapped to [-5, 5]
            x = np.zeros(dim)
            for d in range(dim):
                # Random value in [i/n, (i+1)/n]
                val = (i + rng.random()) / n_explore
                # Map to [-5, 5]
                x[d] = -5.0 + 10.0 * val
            points.append(x)
        
        # Evaluate exploration points
        explore_results = []
        for x in points:
            if eval_count >= budget or time_check():
                break
            f = func(x)
            eval_count += 1
            update_best(x, f)
            explore_results.append((f, x))
        
        if time_check():
            return best_x, best_f

        # Sort exploration results
        explore_results.sort(key=lambda t: t[0])
        
        # Select top K candidates for local optimization
        # K depends on remaining budget. 
        # Nelder-Mead in 5D needs at least dim+1 = 6 points per simplex, 
        # and usually ~10-50 evaluations to converge or stagnate.
        # Let's pick top 3 candidates.
        K = min(3, len(explore_results))
        
        if K == 0:
            return best_x, best_f

        # Phase 2: Local Refinement with Nelder-Mead
        # Remaining budget
        remaining_budget = budget - eval_count
        
        # Allocate budget to each candidate
        # Give more budget to the best candidate? 
        # Or distribute evenly? 
        # Given the "step" nature, the global optimum might be in a different basin.
        # However, the top 1 is likely close. 
        # Let's give 50% to best, 25% to second, 25% to third.
        budgets = []
        for i in range(K):
            if i == 0:
                b = int(remaining_budget * 0.5)
            elif i == 1:
                b = int(remaining_budget * 0.25)
            else:
                b = int(remaining_budget * 0.25)
            # Ensure minimum budget for NM to do anything
            b = max(b, 10)
            budgets.append(b)
        
        # Normalize budgets to sum to remaining_budget
        total_b = sum(budgets)
        if total_b > 0:
            budgets = [int(b * remaining_budget / total_b) for b in budgets]
        
        # Run Nelder-Mead on each candidate
        for i in range(K):
            if eval_count >= budget or time_check():
                break
            
            x_start = explore_results[i][1].copy()
            b_alloc = budgets[i]
            
            # Simple Nelder-Mead implementation
            self._nelder_mead(func, x_start, b_alloc, dim, update_best, eval_count, time_check, start_time)
            # Note: _nelder_mead needs to return eval_count or we track it globally?
            # To keep it simple and avoid passing mutable state around, 
            # let's inline the NM logic or make it return the count used.
            
            # Actually, let's implement a local NM that updates a local counter 
            # and returns the number of evaluations used.
            # But update_best needs to be called.
            # Let's refactor: _nelder_mead returns (evals_used, best_x_local, best_f_local)
            # But we need to call func and track global eval_count.
            
            # Let's just write the NM loop directly here for clarity and control.
            
            # Initialize Simplex
            # Standard initial simplex:
            # x0
            # x0 + delta * e_i
            # delta is usually 0.05 or 0.01 * (max - min) of box?
            # Box is [-5, 5], so range 10. 0.05 * 10 = 0.5.
            delta = 0.5
            
            simplex = [x_start.copy()]
            for d in range(dim):
                x_new = x_start.copy()
                x_new[d] += delta
                simplex.append(x_new)
            
            # Evaluate simplex
            simplex_f = []
            for x in simplex:
                if eval_count >= budget or time_check():
                    break
                f = func(x)
                eval_count += 1
                update_best(x, f)
                simplex_f.append(f)
            
            # If we didn't evaluate the whole simplex, skip this candidate
            if len(simplex_f) < dim + 1:
                continue
                
            simplex = np.array(simplex)
            simplex_f = np.array(simplex_f)
            
            # NM Parameters
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma = 0.5   # Shrink
            
            max_iter = b_alloc - (dim + 1) # Rough estimate of remaining steps
            # Actually, we have b_alloc total for this candidate. 
            # We used dim+1. Remaining is b_alloc - (dim+1).
            # Each NM step costs 1 evaluation usually (reflection), sometimes 2 (expansion/contraction).
            # So max_iter should be conservative.
            max_iter = max(1, (b_alloc - (dim + 1)) // 2)
            
            for _ in range(max_iter):
                if eval_count >= budget or time_check():
                    break
                
                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]
                
                # Calculate centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                worst_x = simplex[-1]
                worst_f = simplex_f[-1]
                
                # Reflection
                x_r = centroid + alpha * (centroid - worst_x)
                f_r = func(x_r)
                eval_count += 1
                update_best(x_r, f_r)
                
                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - worst_x)
                    f_e = func(x_e)
                    eval_count += 1
                    update_best(x_e, f_e)
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    # Just replace worst
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    # Outside contraction
                    x_c = centroid + rho * (worst_x - centroid)
                    f_c = func(x_c)
                    eval_count += 1
                    update_best(x_c, f_c)
                    
                    if f_c < worst_f:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        # Shrink all points except the best towards the best
                        best_x_simplex = simplex[0]
                        for j in range(1, dim + 1):
                            simplex[j] = best_x_simplex + sigma * (simplex[j] - best_x_simplex)
                            if eval_count >= budget or time_check():
                                break
                            f_s = func(simplex[j])
                            eval_count += 1
                            update_best(simplex[j], f_s)
                            simplex_f[j] = f_s
                        # If we broke out of shrink loop early, we might have inconsistent state
                        # But we stop anyway if budget/time is up.

        return best_x, best_f
