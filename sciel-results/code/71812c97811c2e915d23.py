"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """MOEA/D-style optimizer using weight vectors and local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 310.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Generate weight vectors for decomposition
        # We want a diverse set of weight vectors to cover the Pareto front.
        # For 3 objectives, a simplex-lattice approach is good.
        # Let's use a moderate number of subproblems to fit within budget.
        # Budget is 200. If we use 20 subproblems, we have 10 evaluations per subproblem on average?
        # Or we can use a smaller number of subproblems and more iterations.
        # Let's try N=10 subproblems.
        n_subproblems = 10
        
        # Generate weight vectors on the simplex for n_obj=3
        # Simplex lattice with resolution r.
        # Number of points is C(r + n_obj - 1, n_obj - 1).
        # For n_obj=3, C(r+2, 2).
        # If r=4, C(6,2) = 15.
        # If r=3, C(5,2) = 10.
        # Let's generate a dense set and sample down to n_subproblems if needed, 
        # or just use a fixed grid.
        
        # Simpler: generate random weight vectors on the simplex and normalize.
        # But structured is better for coverage.
        # Let's use a fixed set of weight vectors that covers the corners and edges well.
        
        # For 3 objectives, we can define weights manually or use a lattice.
        # Let's use a lattice with resolution 4 (15 points) and pick 10 if needed, 
        # or just use 15 subproblems. 200 / 15 ~ 13 evals per subproblem.
        
        # Let's stick to n_subproblems = 15 for better coverage.
        n_sub = 15
        
        # Generate simplex lattice points
        def simplex_lattice(n_obj, resolution):
            points = []
            def backtrack(idx, remaining, current):
                if idx == n_obj - 1:
                    current.append(remaining)
                    points.append(current[:])
                    current.pop()
                    return
                for i in range(remaining + 1):
                    current.append(i)
                    backtrack(idx + 1, remaining - i, current)
                    current.pop()
            backtrack(0, resolution, [])
            return np.array(points) / resolution
        
        weights = simplex_lattice(self.n_obj, 4)
        # We have 15 points.
        # If n_sub > len(weights), we'd need to generate more, but 15 is fine.
        # If n_sub < len(weights), we sample.
        if n_sub < len(weights):
            indices = np.random.choice(len(weights), n_sub, replace=False)
            weights = weights[indices]
        elif n_sub > len(weights):
            # Pad with random weights if needed, but 15 is likely enough
            pass
            
        n_sub = len(weights)
        
        # 2. Initialize population
        # Each subproblem gets one initial solution.
        # Use random initialization within bounds.
        rng = np.random.default_rng(42)
        
        pop = np.zeros((n_sub, self.dim))
        for i in range(n_sub):
            pop[i] = lo + (hi - lo) * rng.random(self.dim)
            
        F = np.array([problem(x) for x in pop])
        eval_count = n_sub
        
        # Ideal point (minimums) and Nadir point (maximums) for normalization
        # We don't know the true ideal/nadir, so we use the current population's extremes
        # or just use the raw weighted sum if normalization is unstable.
        # For MOEA/D with Tchebycheff or Weighted Sum, normalization helps.
        # Let's use Weighted Sum as it's simpler and robust for convex fronts (WFG4 is non-convex, 
        # but Weighted Sum still works, just might miss parts. Tchebycheff is better for non-convex).
        # Let's use Tchebycheff distance.
        
        def tchebycheff_distance(f, w, ideal):
            # d(x, w) = max_i (w_i * |f_i - ideal_i|)
            # Since we minimize, ideal is the min.
            # We need to estimate ideal. Let's use the min of F so far.
            return np.max(w * np.abs(f - ideal))
        
        # 3. Iteration loop
        # We have budget - eval_count evaluations left.
        # Strategy: In each iteration, pick a subproblem, generate an offspring, 
        # and update the neighbor subproblems if it's better.
        
        # Neighborhood: For each subproblem, define neighbors.
        # Simple approach: All other subproblems are neighbors? 
        # Or K nearest based on weight vector distance.
        # Let's use K=5 nearest neighbors in weight space.
        
        # Precompute neighbor lists
        neighbors = []
        for i in range(n_sub):
            dists = np.linalg.norm(weights - weights[i], axis=1)
            # Sort by distance, take top K (excluding self)
            sorted_indices = np.argsort(dists)
            neigh = [j for j in sorted_indices if j != i][:5]
            neighbors.append(neigh)
        
        # Estimate ideal point dynamically
        ideal = np.min(F, axis=0)
        
        # Main loop
        while eval_count < self.budget:
            if time.time() > deadline:
                break
                
            # Pick a subproblem to evolve
            # Random selection is simple. Weighted selection by hypervolume contribution is complex.
            # Let's use random selection for simplicity and speed.
            i = rng.integers(0, n_sub)
            
            # Generate offspring
            # Mutation: Gaussian perturbation around the current solution
            # Or crossover with a neighbor?
            # Let's use a simple mutation: x_new = x_i + noise
            # Scale noise by (hi - lo)
            
            # Adaptive step size? Start with 0.1 * (hi-lo) and decay?
            # Or just fixed 0.1.
            step_size = 0.1
            x_new = pop[i].copy()
            noise = rng.normal(0, step_size, self.dim) * (hi - lo)
            x_new = x_new + noise
            x_new = np.clip(x_new, lo, hi)
            
            F_new = problem(x_new)
            eval_count += 1
            
            # Update ideal point
            ideal = np.minimum(ideal, F_new)
            
            # Check if F_new is better than current solution of subproblem i
            # Using Tchebycheff distance
            d_current = tchebycheff_distance(F[i], weights[i], ideal)
            d_new = tchebycheff_distance(F_new, weights[i], ideal)
            
            if d_new < d_current:
                pop[i] = x_new
                F[i] = F_new
                
                # Update neighbors
                for j in neighbors[i]:
                    d_j_current = tchebycheff_distance(F[j], weights[j], ideal)
                    d_j_new = tchebycheff_distance(F_new, weights[j], ideal)
                    if d_j_new < d_j_current:
                        pop[j] = x_new
                        F[j] = F_new
            
            # Also, we might want to do a local search around the best solution found so far?
            # Or just let the MOEA/D loop run.
            
        # Final evaluation of the best non-dominated points?
        # The problem records ALL returned objective vectors.
        # Wait, the contract says: "Every returned objective vector is recorded".
        # This implies I should return a list of vectors?
        # "The return value of __call__ is ignored."
        # "Every returned objective vector is recorded" -> This phrasing is tricky.
        # Usually, in these tasks, the `problem(x)` call itself might be logged, 
        # OR the return value of `__call__` is a list of vectors that are logged.
        # Let's re-read: "The evaluator calls np.random.seed(seed), creates Optimizer... and calls it once. 
        # ... Every returned objective vector is recorded and the return value of __call__ is ignored."
        
        # This is contradictory. "Return value is ignored" vs "Every returned objective vector is recorded".
        # Interpretation A: The evaluator logs the outputs of `problem(x)` calls.
        # Interpretation B: The evaluator expects `__call__` to return a list of vectors, 
        # and it logs those, but ignores the *type* or *structure* if it's not a list?
        # Or maybe "return value is ignored" means the *object* returned is ignored, 
        # but if it's a list of vectors, those vectors are extracted and recorded?
        
        # Looking at the seed program:
        # `return X` where X is an array of points.
        # If the evaluator records "returned objective vectors", it likely evaluates the returned points?
        # No, that would exceed budget.
        
        # Let's look at the seed program again.
        # It returns `X` (the decision variables).
        # If the evaluator records "returned objective vectors", it must be evaluating `X`?
        # But "More than 200 evaluations... makes the whole result invalid".
        # If I evaluate 200 points in the loop, and then return X, does the evaluator evaluate X again?
        # If so, that's 400 evaluations.
        
        # Alternative Interpretation:
        # The `problem` object passed to `__call__` is a wrapper that logs every call.
        # "Every returned objective vector is recorded" refers to the return values of `problem(x)` calls.
        # "The return value of __call__ is ignored" means I don't need to return anything specific, 
        # but the side-effect of calling `problem` is what counts.
        
        # This is the most standard interpretation for these "black box" evaluators.
        # The evaluator wraps `problem` to count calls and store results.
        # So, I just need to call `problem(x)` up to 200 times.
        # The points I call `problem` on are the ones recorded.
        
        # Therefore, the strategy is to call `problem` on the best points found.
        # The current MOEA/D loop calls `problem` on every offspring.
        # So all offspring are recorded.
        # This is good for coverage.
        
        # However, if I want to maximize hypervolume, I want the *recorded* set to be good.
        # If I record 200 random-ish points, the hypervolume is determined by the non-dominated subset of those 200.
        # The MOEA/D approach generates a spread of points, which is good.
        
        # One optimization: If I find a very good point, I should ensure it's in the recorded set.
        # Since every `problem` call is recorded, this is automatic.
        
        return None # Return value is ignored

# EVOLVE-BLOCK-END
