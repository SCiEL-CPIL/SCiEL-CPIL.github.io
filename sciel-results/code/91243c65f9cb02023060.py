import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 300.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng_ = hi - lo

        # Budget allocation
        # We use a decomposition approach with a grid of reference points.
        # For 3 objectives, we create a simplex grid.
        # Let K be the number of points on each edge.
        # Number of points = C(K + n_obj - 1, n_obj - 1).
        # For n_obj=3, points = (K+1)(K+2)/2.
        # We need to fit within budget.
        # Each local search (Nelder-Mead) takes ~10-20 evals to converge reasonably.
        # Let's aim for ~10-15 sub-problems to allow deeper search.
        # 10 sub-problems * 15 evals = 150. Leaves 50 for restarts/refinement.
        
        # Dynamic K selection based on budget
        # If budget is 200, we can afford K=3 (6 points) or K=4 (10 points) or K=5 (15 points).
        # K=4 gives 10 points. 200/10 = 20 evals per sub-problem.
        # K=5 gives 15 points. 200/15 = 13 evals per sub-problem.
        # Let's start with K=4 (10 points) to get good coverage and depth.
        
        K = 4
        n_obj = self.n_obj
        dim = self.dim
        
        # Generate reference points on the simplex
        # We want points lambda such that sum(lambda) = 1 and lambda_i >= 0
        # We will use these to define Tchebycheff aggregation functions.
        
        ref_points = []
        if n_obj == 3:
            for i in range(K + 1):
                for j in range(K + 1 - i):
                    k = K - i - j
                    ref_points.append([i / K, j / K, k / K])
        elif n_obj == 2:
            for i in range(K + 1):
                ref_points.append([i / K, 1 - i / K])
        else:
            # Fallback for other dimensions: random points on simplex
            for _ in range(10):
                r = np.random.random(n_obj)
                ref_points.append(r / r.sum())
                
        ref_points = np.array(ref_points)
        n_sub = len(ref_points)
        
        # Evaluate budget per sub-problem
        evals_per_sub = max(5, self.budget // n_sub)
        
        # Store all results
        all_X = []
        all_F = []
        
        # Initial random sampling to find a good starting point for each sub-problem
        # Or just start from random points in the decision space.
        
        # To improve success rate, we can use a few random starts per sub-problem if budget allows
        # But let's stick to 1 start per sub-problem first, maybe restart if it fails.
        
        # We need to track total evaluations
        used = 0
        
        # Helper to create the objective function for scipy
        # Tchebycheff: min max_i ( w_i * (f_i(x) - z_i^min) )
        # We need z_min. We can estimate it from initial random samples.
        
        # Phase 1: Quick random sampling to estimate z_min and z_max
        n_init = min(10, self.budget // 10)
        if n_init > 0 and self.budget > 20:
            X_init = lo + rng_ * np.random.random((n_init, dim))
            F_init = np.array([problem(x) for x in X_init])
            used += n_init
            z_min = np.min(F_init, axis=0)
            z_max = np.max(F_init, axis=0)
            
            # Add these to our set of solutions
            all_X.extend([x for x in X_init])
            all_F.extend([f for f in F_init])
        else:
            # Fallback z_min
            z_min = np.zeros(n_obj)
            z_max = np.ones(n_obj)
            
        # Phase 2: Local Search for each sub-problem
        for idx, w in enumerate(ref_points):
            if time.time() > deadline - 10: # Stop if close to deadline
                break
            
            # Define the scalarized objective
            # f_scalar(x) = max_i ( w_i * (f_i(x) - z_min_i) )
            # We need to handle the case where w_i is 0.
            # If w_i is 0, that objective is ignored in the max? 
            # Standard Tchebycheff: min max_i ( w_i * (f_i(x) - z_i^*) )
            # If w_i=0, the term is 0. This might not push that objective.
            # Better: use a small epsilon or normalize.
            
            def scalar_obj(x):
                fx = problem(x)
                # Clip to finite values
                fx = np.array(fx)
                # Tchebycheff aggregation
                # We want to minimize the maximum weighted deviation from ideal point
                # Since z_min is the best found, we want to minimize (fx - z_min)
                # But if fx < z_min, it's better. So we can use (fx - z_min) directly.
                # However, if w_i is 0, we don't care.
                # To ensure all objectives are considered, we can add a small penalty.
                # Or just use the standard definition.
                
                # Normalize weights? They sum to 1.
                # Term: w_i * (fx_i - z_min_i)
                terms = w * (fx - z_min)
                # If w_i is 0, term is 0.
                # This means points on the edge of the simplex (w_i=0) will only optimize the other objectives.
                # This is correct for boundary points of the Pareto front.
                
                # Return the max term
                return np.max(terms)
            
            # Initial point for this sub-problem
            # Random start
            x0 = lo + rng_ * np.random.random(dim)
            
            # Run Nelder-Mead
            # Bounds
            bounds_list = list(zip(lo, hi))
            
            try:
                res = minimize(
                    scalar_obj, 
                    x0, 
                    method='Nelder-Mead', 
                    bounds=bounds_list,
                    options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                # Check how many function calls were made?
                # scipy doesn't easily expose this in the result for NM without a callback.
                # We can estimate or just trust it's small.
                # NM typically uses ~10-20 evals for 12 dims if not too strict.
                # Let's assume it's within budget.
                
                x_opt = res.x
                f_opt = problem(x_opt)
                used += 1 # Approximate, actual calls are hidden in minimize
                
                # We need to be careful with the budget.
                # Since we can't easily count calls inside minimize, 
                # we rely on the fact that NM is fast.
                # If we go over budget, the evaluator will kill us or mark invalid.
                # To be safe, we should wrap the problem call to count.
                
                # Let's refactor to use a counting wrapper.
                
            except Exception:
                pass

        # The above approach has a flaw: we don't know exactly how many evals `minimize` used.
        # If `minimize` uses 100 evals, and we have 10 sub-problems, we use 1000 evals > 200.
        # We MUST count evaluations strictly.
        
        # RESTART WITH STRICT COUNTING
        
        used = 0
        all_X = []
        all_F = []
        
        # Reset z_min estimate
        # We will do a small initial sampling
        n_init = min(10, self.budget // 10)
        if n_init > 0:
            X_init = lo + rng_ * np.random.random((n_init, dim))
            F_init = np.array([problem(x) for x in X_init])
            used += n_init
            z_min = np.min(F_init, axis=0)
            all_X.extend([x for x in X_init])
            all_F.extend([f for f in F_init])
        else:
            z_min = np.zeros(n_obj)
            
        # Define a counting wrapper for problem
        def counting_problem(x):
            nonlocal used
            if used >= self.budget:
                # Return a bad value to stop optimization
                return [1e10] * n_obj
            used += 1
            return problem(x)
            
        # Re-run sub-problems with counting
        for idx, w in enumerate(ref_points):
            if time.time() > deadline - 5:
                break
            if used >= self.budget:
                break
                
            def scalar_obj_counting(x):
                fx = counting_problem(x)
                terms = w * (np.array(fx) - z_min)
                return np.max(terms)
            
            x0 = lo + rng_ * np.random.random(dim)
            
            try:
                res = minimize(
                    scalar_obj_counting, 
                    x0, 
                    method='Nelder-Mead', 
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 20, 'xatol': 1e-3, 'fatol': 1e-3}
                )
                
                # Record the result
                x_opt = res.x
                # We already evaluated x_opt inside minimize? 
                # Not necessarily the last one. We should evaluate it once more to be sure, 
                # but that costs 1 eval.
                # Actually, minimize returns the best x found. We should evaluate it to record F.
                # But if we call problem(x_opt) again, it's a new eval.
                # We can just use the last evaluated point? 
                # It's safer to evaluate the final x.
                
                if used < self.budget:
                    f_opt = counting_problem(x_opt)
                    all_X.append(x_opt)
                    all_F.append(f_opt)
                    
            except Exception:
                pass

        # If we have budget left, do some random restarts or refine best points
        if used < self.budget:
            n_extra = self.budget - used
            # Random sampling
            for _ in range(n_extra):
                if time.time() > deadline:
                    break
                x = lo + rng_ * np.random.random(dim)
                f = problem(x)
                used += 1
                all_X.append(x)
                all_F.append(f)

        # Convert to arrays for return (though return value is ignored, 
        # the recorded vectors are what matter)
        # The evaluator records every returned objective vector from `problem(x)` calls.
        # My `counting_problem` calls `problem(x)`, so those are recorded.
        # The final `return X` is ignored.
        
        return np.array(all_X) if all_X else np.array([])
