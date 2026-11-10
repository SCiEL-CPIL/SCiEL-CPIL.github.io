import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280  # Safety margin
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        # State tracking
        X = []
        F = []
        evals_used = 0
        
        def evaluate(x):
            nonlocal evals_used
            if evals_used >= self.budget:
                return None
            fx = problem(x)
            X.append(x)
            F.append(fx)
            evals_used += 1
            return fx

        # 1. Initialization with Latin Hypercube Sampling (LHS)
        # Use a moderate number of initial points to cover the space well
        n_init = 60
        if self.budget < n_init:
            n_init = max(10, self.budget // 2)
            
        # Generate LHS points
        # qmc.LatinHypercube requires scipy >= 1.7
        try:
            sampler = qmc.LatinHypercube(d=self.dim, seed=0)
            # Draw n_init points
            # We need to map [0,1] to [lo, hi]
            lhs_points = sampler.random(n=n_init)
            X_init = lo + (hi - lo) * lhs_points
        except Exception:
            # Fallback to random
            X_init = lo + (hi - lo) * np.random.random((n_init, self.dim))

        for i in range(n_init):
            if evals_used >= self.budget:
                break
            evaluate(X_init[i])
        
        if evals_used >= self.budget:
            return X

        # Convert to arrays for easier manipulation
        X_arr = np.array(X)
        F_arr = np.array(F)
        
        # 2. Hypervolume-guided Local Search
        # We will iteratively pick the point with the highest marginal hypervolume contribution
        # and refine it using Nelder-Mead.
        
        # Reference point for hypervolume calculation
        # The task specifies reference point [3.0, 5.0, 7.0] for scoring, but for local search
        # we can use a dynamic reference or the same one. Using the scoring reference is safe.
        ref_point = np.array([3.0, 5.0, 7.0])
        
        # Helper to calculate hypervolume contribution of a point given a set of non-dominated points
        def hv_contribution(point_f, non_dom_F, ref):
            # point_f: (n_obj,)
            # non_dom_F: (n, n_obj)
            # ref: (n_obj,)
            # Calculate the volume added by point_f to the union of non_dom_F
            
            # If point is dominated by any in non_dom_F, contribution is 0
            for f in non_dom_F:
                if np.all(f <= point_f) and np.any(f < point_f):
                    return 0.0
            
            # Hypervolume of the set {point_f} union non_dom_F minus Hypervolume of non_dom_F
            # Since n_obj is small (3), we can use a simple 3D hypervolume algorithm
            # or just calculate the box contribution if it's the "highest" point.
            # For 3 objectives, the contribution of a non-dominated point p is the volume of the box
            # [p1, ref1] x [p2, ref2] x [p3, ref3] minus the parts covered by other points.
            
            # Simple approach: Calculate HV of set S and HV of set S+{p}
            # This is O(n^2) or similar per call, which might be slow.
            # Better: Calculate the specific box volume for p, then subtract overlaps.
            
            # Box for p
            box_vol = 1.0
            for k in range(self.n_obj):
                if point_f[k] >= ref[k]:
                    return 0.0
                box_vol *= (ref[k] - point_f[k])
            
            # Subtract volume covered by other non-dominated points
            # This is complex. Let's use a simpler heuristic for selection:
            # Pick the point that is "most extreme" in a direction not yet covered,
            # or just pick a random non-dominated point and refine it.
            # Given the budget constraints, a robust strategy is:
            # 1. Get non-dominated set.
            # 2. Pick the point with the largest distance to the "ideal" point (min of mins) 
            #    or just pick one that is not yet refined.
            
            return box_vol # Placeholder, we will use a simpler selection below

        # Simpler Selection Strategy:
        # 1. Identify non-dominated points.
        # 2. Select the one that maximizes the distance to the centroid of the non-dominated set,
        #    or simply cycle through them.
        # 3. Apply Nelder-Mead to minimize the sum of objectives or a weighted sum.
        #    Since we want to minimize all, we can minimize the Tchebycheff function or just 
        #    use a weighted sum with weights derived from the point's position.
        
        # Let's use a weighted sum method where weights are inversely proportional to the 
        # current objective values (normalized), or just use equal weights and rely on 
        # Nelder-Mead to find a local Pareto point.
        # Actually, minimizing a weighted sum w1*f1 + w2*f2 + w3*f3 with different weights 
        # allows us to explore different parts of the Pareto front.
        
        # We will maintain a list of "active" points to refine.
        # We will pick a non-dominated point, generate weights based on its "direction" 
        # (distance from ideal point), and run Nelder-Mead.
        
        ideal_point = np.min(F_arr, axis=0)
        
        # Track which points we have tried to refine
        # We will keep a list of indices of points in the current non-dominated set
        # and pick the one with the highest "potential" (e.g., farthest from ideal point)
        
        # To avoid re-refining the same point, we can keep a counter or just 
        # accept that Nelder-Mead might converge to the same point.
        
        # Let's try a different approach:
        # Use the remaining budget to perform a series of local searches.
        # Each local search uses a small sub-budget (e.g., 10-20 evals).
        
        sub_budget = 15
        max_iterations = (self.budget - evals_used) // sub_budget
        
        for _ in range(max_iterations):
            if evals_used >= self.budget or time.time() > deadline:
                break
            
            # Get current non-dominated set
            non_dom_idx = self._get_non_dominated_indices(F_arr)
            
            if len(non_dom_idx) == 0:
                # If no non-dominated points (shouldn't happen), pick random
                idx = np.random.randint(len(F_arr))
            else:
                # Pick the point that is farthest from the ideal point in normalized space
                # or just pick the one with the largest hypervolume contribution (approx)
                # Let's pick the one that maximizes the sum of normalized distances to ideal
                scores = []
                for idx in non_dom_idx:
                    f = F_arr[idx]
                    # Normalize by range (ref - ideal)
                    # Avoid division by zero
                    denom = np.maximum(ref_point - ideal_point, 1e-6)
                    dist = np.sum((ref_point - f) / denom) # Higher is better (closer to ref? No, closer to ideal is better)
                    # We want to explore points that are "extreme". 
                    # Let's pick the one with the largest distance to the centroid of the non-dom set.
                    scores.append(-np.sum(f**2)) # Arbitrary, let's use random selection from non-dom for diversity
                    
                # Random selection from non-dominated set to ensure diversity
                idx = non_dom_idx[np.random.randint(len(non_dom_idx))]
            
            x_start = X_arr[idx].copy()
            
            # Define objective function for Nelder-Mead
            # We want to minimize a weighted sum. 
            # Weights can be chosen to push the solution towards a different part of the front.
            # Let's use weights that are inversely related to the current objective values,
            # normalized.
            f_start = F_arr[idx]
            weights = 1.0 / (f_start + 1e-6)
            weights /= np.sum(weights)
            
            def objective(x):
                if evals_used >= self.budget:
                    return np.inf
                fx = problem(x)
                evals_used += 1
                # Record this point
                X.append(x)
                F.append(fx)
                F_arr = np.vstack([F_arr, fx.reshape(1, -1)])
                X_arr = np.vstack([X_arr, x.reshape(1, -1)])
                return np.dot(weights, fx)
            
            # Run Nelder-Mead
            # We need to be careful with the budget. 
            # scipy.optimize.minimize with Nelder-Mead uses multiple evaluations per iteration.
            # We can limit the number of function calls by using options.
            # However, the objective function above increments evals_used and updates F_arr.
            # This is messy because F_arr is being updated inside the loop.
            
            # Let's refactor: 
            # The objective function should just call problem and count.
            # We will collect the results after the optimization.
            
            # Reset local tracking for this optimization
            local_evals = 0
            local_X = []
            local_F = []
            
            def local_objective(x):
                nonlocal local_evals
                if local_evals >= sub_budget:
                    return np.inf
                fx = problem(x)
                local_evals += 1
                local_X.append(x)
                local_F.append(fx)
                return np.dot(weights, fx)
            
            try:
                res = minimize(
                    local_objective, 
                    x_start, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 5, # Limit iterations
                        'xatol': 1e-4,
                        'fatol': 1e-4
                    }
                )
            except Exception:
                pass
            
            # Add all evaluated points from this local search to the global set
            for i in range(len(local_X)):
                if evals_used < self.budget:
                    X.append(local_X[i])
                    F.append(local_F[i])
                    evals_used += 1
                else:
                    break
            
            # Update global arrays
            if local_X:
                F_arr = np.vstack([F_arr, np.array(local_F)])
                X_arr = np.vstack([X_arr, np.array(local_X)])
        
        return X

    def _get_non_dominated_indices(self, F):
        n = len(F)
        non_dom = []
        for i in range(n):
            is_dom = False
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dom = True
                    break
            if not is_dom:
                non_dom.append(i)
        return non_dom

    def _dominates(self, a, b):
        return np.all(a <= b) and np.any(a < b)
