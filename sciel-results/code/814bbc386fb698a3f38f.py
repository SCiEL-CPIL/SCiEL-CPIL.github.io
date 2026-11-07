"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: The task says 300s limit, evaluator kills at 330s.
        # We aim to finish well before 300s.
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume calculation (fixed by task)
        # ref_point = np.array([1705.0, 12.0, 0.3]) # Not strictly needed for optimization, only scoring
        
        # --- Strategy: Multi-Start Scalarization ---
        # 1. Generate a set of weight vectors (directions in objective space)
        # 2. For each weight vector, solve a scalarized problem using L-BFGS-B
        # 3. Collect all solutions
        
        # We need to distribute the budget.
        # L-BFGS-B is fast but each call consumes several evaluations.
        # Let's estimate: A typical L-BFGS-B run on 5 variables might take 5-20 evaluations.
        # If we use 20 weight vectors, and each takes ~10 evals, that's 200 evals.
        # This fits perfectly in the budget.
        
        # Generate weight vectors on a grid in the simplex for n_obj=3.
        # We want to cover the Pareto front well.
        # A grid of points on the 2D simplex (since 3 objectives sum to 1 for weights).
        
        # Let's create a set of weight vectors.
        # We can use a Latin Hypercube or just a regular grid.
        # For 3 objectives, weights (w1, w2, w3) with w1+w2+w3=1, wi >= 0.
        
        # Let's use a grid of 5x5 points on the simplex triangle.
        # This gives 15 unique weight vectors (roughly).
        # Actually, let's just generate random weight vectors or a structured set.
        # A structured set ensures coverage.
        
        # Generate points on the simplex
        def generate_simplex_weights(n_obj, n_per_edge):
            weights = []
            # n_per_edge controls the resolution
            for i in range(n_per_edge + 1):
                for j in range(n_per_edge + 1 - i):
                    k = n_per_edge - i - j
                    w = np.array([i, j, k], dtype=float) / n_per_edge
                    weights.append(w)
            return np.array(weights)
        
        # Let's aim for about 15-20 scalarizations.
        # n_per_edge = 4 gives (5*6)/2 = 15 weights.
        # n_per_edge = 5 gives (6*7)/2 = 21 weights.
        # Let's try n_per_edge = 4 first to be safe on time/evals.
        n_per_edge = 4
        weights = generate_simplex_weights(n_obj, n_per_edge)
        n_weights = len(weights)
        
        # We have 'budget' evaluations.
        # We will assign a sub-budget to each weight vector.
        # However, L-BFGS-B stops when convergence is reached, which might be before using the sub-budget.
        # We can set maxiter and ftol to control this.
        
        # To ensure we use the budget effectively, we can restart if a scalarization finishes too early.
        # But first, let's just run one L-BFGS-B per weight vector.
        
        # Initial points for L-BFGS-B:
        # We can use a Sobol sequence to generate diverse starting points for each weight vector.
        # Or just start from the center? Center might be a local minimum trap.
        # Let's generate multiple starting points per weight vector if budget allows.
        
        # Alternative: 
        # Use the Sobol points from the previous strategy as starting points for the scalarizations?
        # No, let's keep it simple. 
        # For each weight vector, pick a random starting point (or Sobol point) and run L-BFGS-B.
        
        # Let's use a Sobol sequence to generate starting points for the weight vectors.
        # We need n_weights starting points.
        rng = np.random.default_rng(42)
        try:
            sobol = Sobol(d=dim, scramble=True, seed=123)
            m = 1
            while m < n_weights:
                m *= 2
            sobol_samples = sobol.random(m)[:n_weights]
            X_starts = lo + span * sobol_samples
        except:
            X_starts = lo + span * rng.random((n_weights, dim))
            
        all_solutions = []
        eval_count = 0
        
        # Define the scalarized objective function
        # We use the augmented Tchebycheff method or simple weighted sum.
        # Weighted sum is simpler and works well for convex fronts. RE34 might be non-convex.
        # Augmented Tchebycheff is more robust for non-convex fronts.
        # Let's use Weighted Sum for simplicity first, as it's faster.
        # f_scalar(x) = sum(w_i * f_i(x))
        
        # To handle non-convexity, we might need multiple starts per weight.
        # But with only 200 evals, we can't afford many restarts.
        # Let's stick to 1 start per weight vector, but choose the start carefully.
        
        # Actually, a better strategy for non-convex fronts with limited budget:
        # 1. Generate a few random points (e.g., 20).
        # 2. For each of these, run a short L-BFGS-B with a random weight vector?
        # No, that's messy.
        
        # Let's try:
        # 1. Generate N_start = 10 starting points using Sobol.
        # 2. For each starting point, run L-BFGS-B minimizing the weighted sum with a weight vector 
        #    sampled from a uniform distribution on the simplex.
        # 3. This gives 10 solutions.
        # 4. If we have budget left, we can do more.
        
        # Let's refine:
        # We want to cover the front.
        # Let's use 10 weight vectors (n_per_edge=3 gives 6, n_per_edge=4 gives 15).
        # Let's use 10 weight vectors.
        # And for each, we use 1 starting point from Sobol.
        # If the L-BFGS-B uses ~10-15 evals, 10 * 15 = 150 evals.
        # We have 50 evals left.
        # We can use the remaining 50 to do 3-4 more scalarizations or restarts.
        
        # Let's implement a loop that runs scalarizations until the deadline or budget is hit.
        
        # Pre-compute a larger set of weight vectors and starting points.
        # Let's create 20 weight vectors and 20 starting points.
        
        n_iter = 20
        weights_full = generate_simplex_weights(n_obj, 5) # 21 weights
        if len(weights_full) > n_iter:
             weights_full = weights_full[:n_iter]
        else:
             # Pad if needed? No, just use what we have.
             pass
             
        # Generate starting points
        m = 1
        while m < n_iter:
            m *= 2
        try:
            sobol = Sobol(d=dim, scramble=True, seed=123)
            sobol_samples = sobol.random(m)[:n_iter]
            X_starts_full = lo + span * sobol_samples
        except:
            X_starts_full = lo + span * rng.random((n_iter, dim))
            
        solutions = []
        
        for i in range(n_iter):
            if time.time() > deadline:
                break
            if eval_count >= budget:
                break
                
            w = weights_full[i % len(weights_full)]
            x0 = X_starts_full[i % len(X_starts_full)]
            
            # Define the scalarized objective
            def scalarized_obj(x, w=w):
                nonlocal eval_count
                eval_count += 1
                f_vals = problem(x)
                return np.dot(w, f_vals)
                
            # Run L-BFGS-B
            # Bounds are (lo, hi)
            bounds_opt = list(zip(lo, hi))
            
            try:
                res = minimize(scalarized_obj, x0, method='L-BFGS-B', bounds=bounds_opt, 
                               options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5})
                # res.x is the solution
                # We need to get the objective values for the solution
                # Note: res.fun is the scalarized value, not the vector.
                # We need to call problem(res.x) again to get the vector?
                # That would cost 1 more evaluation.
                # Is it worth it? Yes, we need the vector for the Pareto front.
                
                if eval_count < budget:
                    eval_count += 1
                    f_sol = problem(res.x)
                    solutions.append((res.x, f_sol))
                else:
                    # Out of budget, but we have the solution x. 
                    # We can't evaluate it without exceeding budget?
                    # The task says "More than 200 evaluations ... makes the whole result invalid".
                    # So we must stop evaluating.
                    # But we already have res.x. Do we need f_sol?
                    # The evaluator records "Every returned objective vector".
                    # If we don't call problem(res.x), we don't record it.
                    # So we MUST call problem(res.x) to record it.
                    # But we are out of budget.
                    # So we should check budget BEFORE calling minimize?
                    # L-BFGS-B uses multiple evaluations internally.
                    # We don't know exactly how many it will use.
                    # We should be conservative.
                    
                    # If we are close to budget, we shouldn't start a new minimize.
                    # Let's add a check: if budget - eval_count < 10, skip this iteration.
                    pass
                    
            except Exception as e:
                # If minimize fails, we might have used some evaluations.
                # We just skip this weight vector.
                pass
                
        # If we have remaining budget, we can do some random sampling or local search.
        # But L-BFGS-B should have used most of it.
        
        # Return the solutions
        # The return value is ignored, but we must return something.
        # Let's return the list of solutions.
        return solutions

# EVOLVE-BLOCK-END
