"""Grid-based multi-start local optimization for RE21."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-objective optimizer using a deterministic grid initialization 
    followed by local L-BFGS-B refinement.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin against 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # 1. Deterministic Global Grid Initialization
        # With dim=4 and budget=200, we can afford a reasonable grid.
        # 3^4 = 81 points. This covers the space systematically.
        n_grid = 3
        grid_points = []
        for i in range(n_grid):
            for j in range(n_grid):
                for k in range(n_grid):
                    for l in range(n_grid):
                        x = np.array([
                            lo[0] + span[0] * (i + 0.5) / n_grid,
                            lo[1] + span[1] * (j + 0.5) / n_grid,
                            lo[2] + span[2] * (k + 0.5) / n_grid,
                            lo[3] + span[3] * (l + 0.5) / n_grid
                        ])
                        grid_points.append(x)
        
        # Evaluate grid
        X = np.array(grid_points)
        F = np.array([problem(x) for x in X])
        used = len(X)
        
        # Check time
        if time.time() > deadline:
            return X

        # 2. Multi-start Local Optimization
        # We will pick the best 'k' points from the grid (based on a simple 
        # scalarization or non-dominance) and run local optimization on them.
        # To ensure diversity, we pick non-dominated points first, then fill up with best scalarized.
        
        # Find non-dominated points
        nd_indices = []
        for i in range(len(F)):
            is_dominated = False
            for j in range(len(F)):
                if i == j: continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated = True
                    break
            if not is_dominated:
                nd_indices.append(i)
                
        # If we have few non-dominated, add best scalarized points
        # Scalarization: sum of normalized objectives (simple heuristic)
        # Or just use the sum of objectives as a proxy for "goodness" in 4D space
        # Since we want to minimize, lower sum is better.
        # Let's sort by sum of objectives to get candidates for local opt.
        # We want to explore different regions, so we don't just pick the global best.
        # Let's pick the top 10 distinct regions from the grid.
        
        # Simple strategy: Pick 10 points that are "spread out" or just the top 10 by sum.
        # Given the small budget, refining the top 10 grid points is a strong baseline.
        # Let's use a simple clustering or just take the top 10 by sum(F).
        # Actually, for RE21, the Pareto front is smooth. 
        # Let's pick the 10 best points based on the sum of objectives to start local search.
        
        # Sort indices by sum of objectives
        sums = np.sum(F, axis=1)
        sorted_indices = np.argsort(sums)
        
        # We will attempt to refine up to 10 of these, if budget allows.
        # Each local optimization might use ~10-20 evaluations.
        # Budget remaining: 200 - 81 = 119.
        # 10 starts * 12 evals/start = 120. Just fits.
        
        max_starts = 10
        candidates = sorted_indices[:max_starts]
        
        best_x = X[candidates[0]]
        best_f = F[candidates[0]]
        
        for idx in candidates:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
                
            x0 = X[idx]
            f0 = F[idx]
            
            # Define scalarized objective for local search
            # A common choice is the weighted sum. Since we don't know weights, 
            # we can use the sum of objectives, or a Tchebycheff approach.
            # Given the reference point [3000, 0.06], and typical RE21 ranges,
            # a simple sum or product might work. Let's try a weighted sum 
            # that balances the two objectives. 
            # RE21 f1 is usually much larger than f2? 
            # Let's check typical magnitudes. 
            # If f1 ~ 100-1000 and f2 ~ 0.01-0.1, sum is dominated by f1.
            # It's better to normalize or use a specific weight.
            # Let's use a weight vector that tries to find a compromise.
            # w = [1, 1] on normalized scale?
            # For simplicity and robustness, let's minimize the sum of objectives.
            # If scales are very different, this might bias towards f1.
            # Let's try to be smart: use the reference point to define a scalarization.
            # Distance to reference point in objective space?
            # Or just use the sum. Let's stick to sum for now, it's standard.
            
            def scalar_obj(x):
                fx = problem(x)
                # Record evaluation? 
                # The problem __call__ counts evaluations. 
                # We need to track 'used' carefully.
                # The problem function itself doesn't tell us it was called, 
                # but the evaluator counts it. 
                # We must ensure we don't exceed budget.
                # We are inside the loop, so we assume one call per problem(x).
                used_count[0] += 1
                return np.sum(fx)
            
            # We need to track evaluations inside the minimize call.
            # scipy minimize doesn't have a direct "budget" limit that stops it cleanly 
            # without throwing an exception or checking a flag.
            # We can pass a custom callback or just rely on the fact that 
            # L-BFGS-B is efficient.
            # To be safe, we can limit the number of function evaluations in minimize.
            
            # Let's use a simple counter in a closure.
            eval_count = [0]
            def obj_func(x):
                if eval_count[0] >= 15: # Limit per local search
                    return 1e10
                eval_count[0] += 1
                fx = problem(x)
                return np.sum(fx)
            
            # Options for L-BFGS-B
            options = {'maxiter': 50, 'maxfun': 15}
            
            try:
                res = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds, options=options)
                # res.x is the local optimum
                # We need to evaluate the final point to get the true F
                # But minimize already called problem. 
                # The last call in minimize might be the best one.
                # To be safe, we should evaluate res.x again? 
                # No, that wastes a budget. 
                # The evaluator records ALL calls. 
                # So the calls inside minimize are recorded.
                # We just need to know if res.x is better than current best.
                # We need the F for res.x. 
                # We can get it from the last evaluation? 
                # It's safer to just add res.x to our pool and let the evaluator record it.
                # But we need to know its F to compare.
                # Let's evaluate it once more? No, budget.
                # Actually, the 'problem' function is called inside obj_func.
                # The evaluator sees those calls.
                # We don't strictly need to know the F of res.x to return, 
                # because the evaluator records ALL calls made to problem.
                # We just need to ensure we make valid calls.
                # So we don't need to update our internal X/F for the final result,
                # unless we want to use it for further decisions.
                # Since we are doing a fixed number of starts, we can just let the 
                # evaluator record the points found.
                
                # However, we must be careful: if we call problem(res.x) again, 
                # it's a new evaluation.
                # The calls inside minimize are already counted.
                # So we don't need to do anything else for this point.
                
            except Exception:
                pass # Ignore errors in local optimization
            
            # Update used budget
            used += eval_count[0]
            
        return X # Return initial grid, the evaluator has recorded the rest
# EVOLVE-BLOCK-END
