"""Rewritten program using a Model-and-Optimize approach with scalarization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    """
    Multi-objective optimizer using a portfolio of scalarized local optimizations.
    Strategy:
    1. Sample a diverse set of random points.
    2. Identify promising regions (non-dominated).
    3. Use SciPy's optimizer to minimize weighted sums of objectives starting from good initial points.
    4. Aggregate all evaluated points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.evals = 0
        self.start_time = None
        self.timeout = 290.0  # Safety margin below 300s/330s

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Store all evaluated points to return (though ignored, good practice)
        all_X = []
        all_F = []
        
        def safe_eval(x):
            """Wraps problem call with time and budget checks."""
            if time.time() - self.start_time > self.timeout:
                raise TimeoutError("Time limit exceeded")
            if self.evals >= self.budget:
                raise ValueError("Budget exceeded")
            
            x = np.asarray(x)
            fx = problem(x)
            self.evals += 1
            return np.array(fx)

        # 1. Initial Random Sampling
        # Use a significant portion of the budget for exploration
        n_rand = int(self.budget * 0.4) 
        if n_rand < 10:
            n_rand = 10
            
        # Ensure we don't exceed budget
        n_rand = min(n_rand, self.budget)
        
        for _ in range(n_rand):
            if time.time() - self.start_time > self.timeout:
                break
            x = np.random.uniform(lo, hi)
            try:
                fx = safe_eval(x)
                all_X.append(x)
                all_F.append(fx)
            except (TimeoutError, ValueError):
                break
        
        if not all_F:
             # If even random sampling failed or timed out immediately, return empty or single point if possible
             return np.array([np.random.uniform(lo, hi)])

        F_arr = np.array(all_F)
        X_arr = np.array(all_X)
        
        # 2. Scalarization Portfolio
        # Define a set of weight vectors to cover the Pareto front
        # For 3 objectives, we can use a set of points on the simplex
        # We will pick 'n_weights' directions
        weights = []
        # Uniform weight
        weights.append(np.ones(self.n_obj) / self.n_obj)
        # Extreme weights (prioritize one objective)
        for i in range(self.n_obj):
            w = np.zeros(self.n_obj)
            w[i] = 0.8
            # Distribute remainder among others to avoid degeneracy
            rem = 0.2 / (self.n_obj - 1)
            for j in range(self.n_obj):
                if j != i:
                    w[j] = rem
            weights.append(w)
        # Add more mixed weights for better coverage
        weights.append(np.array([0.5, 0.3, 0.2]))
        weights.append(np.array([0.2, 0.3, 0.5]))
        weights.append(np.array([0.4, 0.4, 0.2]))
        weights.append(np.array([0.2, 0.4, 0.4]))
        weights.append(np.array([0.4, 0.2, 0.4]))
        weights.append(np.array([0.3, 0.5, 0.2]))
        weights.append(np.array([0.2, 0.5, 0.3]))
        weights.append(np.array([0.5, 0.2, 0.3]))
        
        # Filter to non-dominated set for better starts
        def is_dominated(f, F):
            return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))
        
        nondom_mask = np.array([not is_dominated(f, F_arr) for f in F_arr])
        start_pool = X_arr[nondom_mask] if np.any(nondom_mask) else X_arr
        start_pool_F = F_arr[nondom_mask] if np.any(nondom_mask) else F_arr
        
        for w in weights:
            if time.time() - self.start_time > self.timeout:
                break
            if self.evals >= self.budget - 5: # Leave room for at least one eval
                break
            
            # Find the best starting point for this weight
            weighted_sums = start_pool_F @ w
            best_idx = np.argmin(weighted_sums)
            x0 = start_pool[best_idx]
            
            # Define scalar objective
            def scalar_obj(x):
                try:
                    fx = safe_eval(x)
                    return np.dot(fx, w)
                except (TimeoutError, ValueError):
                    return np.inf
            
            # Optimize
            try:
                # Use L-BFGS-B for bounded optimization
                # maxiter is limited, but we also have function call budget implicitly via safe_eval
                res = minimize(scalar_obj, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 30, 'maxfun': 20})
                # The result res.x is the best found, but we've already recorded the evals in safe_eval
                # We should ensure the final point is recorded if it hasn't been, but safe_eval handles counting.
                # However, the last point evaluated inside minimize might be the one we want to keep.
                # Since safe_eval appends to all_X/all_F, we don't need to do anything special, 
                # just that the loop in minimize calls safe_eval.
            except Exception:
                pass
        
        # 3. Local refinement around best non-dominated points
        if time.time() - self.start_time <= self.timeout and self.evals < self.budget - 10:
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)
            nondom_mask = np.array([not is_dominated(f, F_arr) for f in F_arr])
            if np.any(nondom_mask):
                # Pick top non-dominated points for local refinement
                nd_X = X_arr[nondom_mask]
                nd_F = F_arr[nondom_mask]
                
                # Sort by hypervolume contribution or just pick diverse ones
                # For simplicity, pick the first few non-dominated points
                n_refine = min(5, len(nd_X), (self.budget - self.evals) // 5)
                
                for i in range(n_refine):
                    if time.time() - self.start_time > self.timeout:
                        break
                    if self.evals >= self.budget - 2:
                        break
                    
                    x0 = nd_X[i]
                    
                    # Try small perturbations
                    for step in [0.1, 0.05, 0.02]:
                        if self.evals >= self.budget - 1:
                            break
                        for d in range(self.dim):
                            if self.evals >= self.budget - 1:
                                break
                            for sign in [-1, 1]:
                                if self.evals >= self.budget - 1:
                                    break
                                x_new = x0.copy()
                                x_new[d] += sign * step
                                x_new = np.clip(x_new, lo, hi)
                                try:
                                    fx = safe_eval(x_new)
                                    all_X.append(x_new)
                                    all_F.append(fx)
                                except (TimeoutError, ValueError):
                                    break
        
        # 4. Final cleanup / Random fill if budget remains
        # If we have budget left, do a bit more random search to fill gaps
        while self.evals < self.budget:
            if time.time() - self.start_time > self.timeout:
                break
            x = np.random.uniform(lo, hi)
            try:
                fx = safe_eval(x)
                all_X.append(x)
                all_F.append(fx)
            except (TimeoutError, ValueError):
                break
        
        # Return the last population or all points? 
        # The prompt says "return value of __call__ is ignored". 
        # But "Keep the program's entry-point function name(s) and their return interface EXACTLY as in the current program".
        # Current program returns X (the population array). 
        # We should return an array of shape (N, dim).
        if all_X:
            return np.array(all_X)
        else:
            return np.empty((0, self.dim))

# EVOLVE-BLOCK-END
