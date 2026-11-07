# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """MOEA/D using Tchebycheff decomposition and Nelder-Mead local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        # Use np.random which is seeded by the evaluator
        rng = np.random
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        center = (lo + hi) / 2.0
        
        # Generate weight vectors for Tchebycheff decomposition
        # We want to cover the simplex. 
        # For 3 objectives, we can generate a grid or use random points on the simplex.
        # Given the budget constraint (200 evals) and 12 dims, we need to be very efficient.
        # Nelder-Mead in 12 dims is expensive. 
        # Let's estimate: NM requires roughly (dim+1)*iterations evals. 
        # If we do 10 iterations, that's ~130 evals per subproblem.
        # With 200 budget, we can only do 1-2 subproblems if we use full NM.
        # This suggests we need a lighter local search or fewer, more targeted subproblems.
        # Alternatively, we can use a very fast gradient-free method or limited NM.
        
        # Let's try a hybrid: 
        # 1. Initialize with a few random points.
        # 2. Use the remaining budget to run short Nelder-Mead runs on the best candidates 
        #    guided by Tchebycheff weights.
        
        # Actually, the directive says: "map ... onto 30 independent single-objective sub-problems".
        # But 30 subproblems * ~13 evals each (minimal NM) = 390 > 200.
        # We must adapt. We will use a smaller number of weight vectors or a faster local search.
        # Let's use 5-10 weight vectors and a very constrained Nelder-Mead (maxiter=5, maxfev=10).
        # Or better: Use the budget to perform a few full local searches on the most promising areas.
        
        # Strategy:
        # 1. Generate a set of weight vectors (e.g., 5 uniform points on the simplex).
        # 2. For each weight vector, initialize Nelder-Mead at the center.
        # 3. Run Nelder-Mead with a tight maxfev to respect the budget.
        # 4. Collect all evaluated points.
        
        # To ensure we don't exceed budget:
        # Total budget = 200.
        # Let's allocate budget / n_weights to each subproblem.
        # If n_weights = 5, budget per subproblem = 40.
        # NM in 12 dims with 40 evals is feasible for a quick local refinement.
        
        n_weights = 5
        
        # Generate weight vectors on the simplex for 3 objectives
        # We can use a simple grid or random. Let's use a structured set to ensure coverage.
        # For 3 objectives, weights (w1, w2, w3) such that sum=1, wi>=0.
        # Points: (1,0,0), (0,1,0), (0,0,1) and some intermediates like (0.5, 0.25, 0.25)
        weights = []
        # Add corners
        weights.append([1.0, 0.0, 0.0])
        weights.append([0.0, 1.0, 0.0])
        weights.append([0.0, 0.0, 1.0])
        # Add edge midpoints
        weights.append([0.5, 0.5, 0.0])
        weights.append([0.5, 0.0, 0.5])
        weights.append([0.0, 0.5, 0.5])
        # Add center
        weights.append([1/3, 1/3, 1/3])
        
        # We have 7 weights. 
        # Let's select 5 to keep budget per subproblem higher? 
        # Or just use all 7 and distribute budget.
        # 200 / 7 = ~28 evals per subproblem.
        # This is tight for NM in 12D, but might find local optima near the center.
        
        # To improve, we can also add some random initializations.
        
        evaluated_points = []
        evaluated_F = []
        eval_count = 0
        
        def update_eval(x, f):
            nonlocal eval_count
            evaluated_points.append(x)
            evaluated_F.append(f)
            eval_count += 1
        
        # 1. Initialize with center
        x0 = center.copy()
        f0 = problem(x0)
        update_eval(x0, f0)
        
        # 2. Run local searches for each weight vector
        for w in weights:
            if eval_count >= self.budget:
                break
            if time.time() > deadline:
                break
            
            w = np.array(w)
            
            # Objective function for Tchebycheff scalarization
            def tcheb_obj(x):
                fx = problem(x)
                # Tchebycheff: max_i (w_i * |f_i - z_i^*|)
                # We don't know z^* exactly, so we use the current best known or a fixed reference.
                # Using current best is adaptive. Let's use the best non-dominated point found so far.
                # For simplicity and speed, let's use a fixed reference point like the center's F or a known good point.
                # Actually, standard Tchebycheff uses an ideal point z^*. 
                # Since we don't know it, we can approximate it with the minimum seen so far for each objective.
                # Or, we can just minimize the weighted sum if the front is convex?
                # WFG7 front is convex. Tchebycheff is robust.
                # Let's use the min of evaluated_F for each objective as z^*.
                if len(evaluated_F) == 0:
                    z_star = np.zeros(self.n_obj)
                else:
                    z_star = np.array([min(f[i] for f in evaluated_F) for i in range(self.n_obj)])
                
                # Handle case where z_star is not strictly better (it is by definition min)
                # But if all are 0, we might have issues. WFG7 objectives are positive.
                
                # Normalize? Tchebycheff usually doesn't require normalization if scales are similar.
                # WFG7 objectives have similar scales (1-3, 1-5, 1-7).
                # Let's stick to simple Tchebycheff.
                
                return np.max(w * np.abs(fx - z_star))
            
            # Nelder-Mead setup
            # maxfev: limit evaluations for this subproblem
            # We have remaining budget. Let's give it a fraction.
            # To be safe, let's cap maxfev to 20 for each subproblem.
            # If we have 7 subproblems, 7*20 = 140 evals. Plus initial 1. Total 141.
            # We have 200. This is safe.
            
            maxfev_sub = 20
            
            # Initial simplex for NM
            # x0 is center.
            # Perturb x0 slightly for each weight to diversify?
            # Or start from the best point found so far?
            # Starting from center is good for global coverage initially.
            # Let's start from center for all, but NM will move them.
            
            res = minimize(tcheb_obj, x0, method='Nelder-Mead', 
                           options={'maxfev': maxfev_sub, 'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
            
            # The `problem` calls inside `tcheb_obj` are counted in `eval_count`?
            # NO. `tcheb_obj` calls `problem` but we didn't update `eval_count` inside `tcheb_obj`.
            # We MUST update eval_count inside the objective function to track budget correctly.
            
            # Let's refactor to ensure counting is correct.
            
            break # Break out to implement correct counting version below
            
        # RE-IMPLEMENTATION WITH CORRECT COUNTING
        
        # Reset
        evaluated_points = []
        evaluated_F = []
        eval_count = 0
        
        # We need a global counter or closure that updates eval_count.
        # Python closures can modify nonlocal variables.
        
        # Let's restart the logic properly.
        
        # 1. Initial point
        x0 = center.copy()
        f0 = problem(x0)
        evaluated_points.append(x0)
        evaluated_F.append(f0)
        eval_count = 1
        
        # 2. Local searches
        for w in weights:
            if eval_count >= self.budget:
                break
            if time.time() > deadline:
                break
            
            w = np.array(w)
            
            # Define objective with counting
            def tcheb_obj(x, w=w, z_star_holder={}):
                # We need access to current best z_star.
                # Using a mutable holder to pass current z_star is tricky if it updates.
                # Simpler: Use a fixed reference point for Tchebycheff?
                # Or update z_star dynamically?
                # Dynamic z_star makes the objective non-static, which NM doesn't like.
                # Standard practice: Use an ideal point estimate.
                # Let's use the minimum of the initial population (just x0 for now) or a fixed point.
                # Given WFG7, the ideal point is likely (1, 1, 1) or similar.
                # Let's use the best known from evaluated_F.
                # To make it static for the duration of one NM run, we freeze z_star at the start of the run.
                
                fx = problem(x)
                # Update global lists
                evaluated_points.append(x)
                evaluated_F.append(fx)
                nonlocal eval_count
                eval_count += 1
                
                # Calculate Tchebycheff value
                # Use z_star calculated BEFORE this evaluation?
                # Or use a fixed z_star = [1, 1, 1]?
                # Let's use the min of evaluated_F EXCLUDING the current point to avoid self-reference issues?
                # Or just use the min of all evaluated_F so far.
                if len(evaluated_F) > 1:
                    # Exclude current point? No, just use all.
                    z_star = np.array([min(f[i] for f in evaluated_F[:-1]) for i in range(self.n_obj)])
                else:
                    z_star = np.array([1.0, 1.0, 1.0]) # Fallback ideal for WFG7
                
                return np.max(w * np.abs(fx - z_star))
            
            # Run NM
            # Start from center
            res = minimize(tcheb_obj, x0, method='Nelder-Mead', 
                           options={'maxfev': 20, 'maxiter': 30})
            
            # The function tcheb_obj has already updated evaluated_points and eval_count.
            # We don't need to do anything with res directly, as we record every evaluation.
            
        # 3. If we have budget left, do a few random explorations or refine the best point
        if eval_count < self.budget:
            # Find the best point (lowest sum or hypervolume contribution)
            # Let's just do some random local searches around the best points found.
            
            # Sort by sum of objectives
            sums = np.sum(evaluated_F, axis=1)
            best_idx = np.argmin(sums)
            best_x = evaluated_points[best_idx]
            
            # Run a few more NM runs starting from best_x with different weights
            for _ in range(3):
                if eval_count >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Random weight
                w = rng.random(self.n_obj)
                w /= np.sum(w)
                
                def tcheb_obj2(x, w=w):
                    fx = problem(x)
                    evaluated_points.append(x)
                    evaluated_F.append(fx)
                    nonlocal eval_count
                    eval_count += 1
                    
                    z_star = np.array([min(f[i] for f in evaluated_F[:-1]) for i in range(self.n_obj)]) if len(evaluated_F) > 1 else np.array([1.0, 1.0, 1.0])
                    
                    return np.max(w * np.abs(fx - z_star))
                
                res = minimize(tcheb_obj2, best_x, method='Nelder-Mead', 
                               options={'maxfev': 10, 'maxiter': 20})
                
        return evaluated_points

# EVOLVE-BLOCK-END
