"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective optimizer using Multi-Start Gradient Descent."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 330s limit, leave 30s buffer
        deadline = start_time + 300
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # RE34 specific knowledge: 
        # Objective 1: Structural weight (minimize)
        # Objective 2: Frequency (maximize -> minimize -f)
        # Objective 3: Displacement (minimize)
        
        # We don't have the exact formula, but we can use the black-box function.
        # However, gradient-based methods need gradients. 
        # If we use scipy.optimize.minimize with method='L-BFGS-B', it approximates gradients 
        # or uses finite differences if not provided. 
        # Since we don't have the analytical gradient, we must use a method that handles 
        # black-box functions well or provide finite difference gradients.
        # L-BFGS-B with finite differences is expensive per step.
        # Alternative: Use a derivative-free method like Nelder-Mead or Powell?
        # But the task implies we can use SciPy. 
        # Let's try a hybrid: 
        # 1. Initial random sampling to get a rough sense of the landscape.
        # 2. Use local search from promising points.
        
        # Actually, a pure gradient descent with finite differences might be too slow 
        # if the function is noisy or if we have to evaluate many times per step.
        # Let's look at the budget: 200 evaluations.
        # A single L-BFGS-B run might take 10-50 evaluations.
        # So we can do ~4-20 local optimizations.
        
        # Strategy:
        # 1. Generate a set of initial points.
        # 2. For each point, run a local minimization on a weighted sum of objectives?
        #    No, weighted sum misses non-convex parts.
        # 3. Better: Run local minimization on each objective individually?
        #    Or use a multi-start approach on a scalarized objective?
        
        # Let's try a different angle: 
        # The RE34 problem is smooth. 
        # We can use `scipy.optimize.minimize` with `method='L-BFGS-B'`.
        # To handle multi-objective, we can use the "epsilon-constraint" method or 
        # simply run multiple local searches with different weighting schemes 
        # and then filter for non-dominated points.
        
        # Given the small budget (200), let's try:
        # 1. Randomly sample 20 points.
        # 2. For each, run L-BFGS-B on a weighted sum of objectives with random weights.
        # 3. Collect all evaluated points and return the non-dominated set.
        
        # However, L-BFGS-B with finite differences will call the function 
        # (dim+1) times per gradient step. With dim=5, that's 6 calls per step.
        # If it takes 20 steps, that's 120 calls per optimization.
        # That's too many for 200 total budget if we do multiple restarts.
        
        # Alternative: Use `method='Powell'` which is derivative-free and often 
        # efficient for low-dimensional problems.
        # Or `Nelder-Mead`.
        
        # Let's use a strategy that balances exploration and exploitation:
        # 1. Sample 10 random points.
        # 2. For each, run a local search using Powell's method on a weighted sum.
        # 3. To cover the front, use different weight vectors.
        
        # Wait, the reference point is [1705.0, 12.0, 0.3].
        # This suggests the scales are very different.
        # We should normalize the objectives before weighting?
        # Or just use the raw values.
        
        # Let's try a simpler, robust approach:
        # 1. Generate 50 random points.
        # 2. Use these as seeds for local searches.
        # 3. But 50 * (cost of local search) might exceed 200.
        
        # Let's estimate cost of Powell's method for 5D.
        # It might take 20-50 evaluations per run.
        # So we can afford ~4-10 local searches.
        
        # Let's do 5 local searches with different weight vectors.
        # Weights:
        # [1, 0, 0], [0, 1, 0], [0, 0, 1] (extremes)
        # [0.5, 0.25, 0.25], [0.25, 0.5, 0.25], etc.
        
        # Actually, let's use a more sophisticated initialization:
        # Latin Hypercube Sampling (LHS) for the initial points to ensure spread.
        
        n_starts = 8
        # Generate weight vectors on the simplex
        weights = []
        for i in range(n_starts):
            # Random weights
            w = np.random.rand(self.n_obj)
            w /= np.sum(w)
            weights.append(w)
        
        # Ensure we include extreme points
        weights[0] = np.array([1.0, 0.0, 0.0])
        weights[1] = np.array([0.0, 1.0, 0.0])
        weights[2] = np.array([0.0, 0.0, 1.0])
        weights[3] = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])
        
        all_points = []
        all_objs = []
        used_evals = 0
        
        # We need to track evaluations carefully.
        # scipy.optimize.minimize does not have a direct "callback" for every function call 
        # that returns the value in a way that is easy to count globally without wrapping.
        # We will wrap the problem function to count evaluations.
        
        eval_count = 0
        max_budget = self.budget
        
        def wrapped_problem(x):
            nonlocal eval_count
            if eval_count >= max_budget:
                return np.array([0, 0, 0]) # Dummy
            eval_count += 1
            # Clip to bounds as per problem description? 
            # "problem(x) clips a finite length-5 vector to the bounds"
            # So we can just pass x, but L-BFGS-B handles bounds.
            # Powell does not handle bounds natively well, so we might need to clip x before passing?
            # The problem function itself clips.
            return problem(x)
        
        for w in weights:
            if time.time() > deadline - 10:
                break
            if eval_count >= max_budget:
                break
            
            # Random start point
            x0 = np.random.uniform(lo, hi)
            
            # Define objective to minimize: weighted sum
            def obj_func(x):
                f = wrapped_problem(x)
                return np.dot(f, w)
            
            try:
                # Use Powell's method, it's derivative-free and handles non-smoothness better
                # than L-BFGS-B if gradients are approximated poorly.
                # However, L-BFGS-B is usually faster for smooth problems.
                # Let's try L-BFGS-B with finite differences.
                # To limit evaluations, we can set maxiter.
                res = minimize(obj_func, x0, method='L-BFGS-B', bounds=list(zip(lo, hi)), 
                               options={'maxiter': 50, 'maxfun': 50})
                
                # Collect the best point found by the optimizer
                # Note: res.fun is the scalar value, we need the vector.
                # We need to evaluate res.x again to get the vector? 
                # That consumes 1 more eval.
                # Or we can store the last few points.
                
                if eval_count < max_budget:
                    f_res = problem(res.x)
                    eval_count += 1
                    all_points.append(res.x)
                    all_objs.append(f_res)
                    
            except Exception:
                pass
        
        # If we have budget left, do some random sampling to fill gaps
        while eval_count < max_budget and time.time() < deadline - 5:
            x = np.random.uniform(lo, hi)
            f = problem(x)
            eval_count += 1
            all_points.append(x)
            all_objs.append(f)
            
        if len(all_objs) == 0:
            return np.array([])
            
        # Filter for non-dominated points
        F = np.array(all_objs)
        P = np.array(all_points)
        
        n_points = len(F)
        is_dominated = np.zeros(n_points, dtype=bool)
        
        for i in range(n_points):
            for j in range(n_points):
                if i == j:
                    continue
                # If j dominates i, mark i as dominated
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated[i] = True
                    break
        
        # Keep non-dominated points
        # Also filter out points worse than reference point?
        # "Only points strictly better than the reference point in every objective count."
        # Reference: [1705.0, 12.0, 0.3]
        ref = np.array([1705.0, 12.0, 0.3])
        
        valid_mask = ~is_dominated
        # Strictly better means < ref in all objectives
        valid_mask &= np.all(F < ref, axis=1)
        
        if not np.any(valid_mask):
             # If no points are valid, return all non-dominated? 
             # Or just return the best found?
             # The scorer ignores points not better than ref.
             # So returning them is harmless, but we should try to return something valid if possible.
             # If none are valid, the score is 0 for this seed.
             # Let's return the non-dominated set anyway, maybe some are close.
             valid_mask = ~is_dominated
             
        return P[valid_mask]

# EVOLVE-BLOCK-END
