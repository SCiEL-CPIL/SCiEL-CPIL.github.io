"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using LHS initialization and multi-start local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # 1. Generate a diverse set of weight vectors for scalarization
        # We use a deterministic grid on the simplex to ensure coverage.
        # For 3 objectives, we can use a grid of size k.
        # Let's use a reasonably dense grid to cover the front well.
        # Number of weights: ~30-50
        # We will use 36 weight vectors (6x6 grid on the 2-simplex)
        
        # Generate points on the 2-simplex (3 objectives)
        # Using a simple grid approach
        # u, v in [0, 1] such that u+v <= 1
        # w0 = 1-u-v, w1=u, w2=v
        # To get uniform coverage, we can sample or use a grid
        # Grid:
        # Let's use a triangular grid
        # Number of points per side: 6 -> 6+5+4+3+2+1 = 21 points?
        # Or just random uniform on simplex
        # Let's use a deterministic set of weights based on a grid
        n_weights = 36
        weights = []
        # We will use a grid of u and v
        # u in [0, 1], v in [0, 1-u]
        # To make it symmetric, we can use a grid of size 6x6 and filter
        # Or just use a fixed set of 36 points
        # Let's use a simple deterministic generation
        # We want weights that cover the simplex well
        # We can use the Dirichlet distribution with a fixed seed for reproducibility
        # Or a grid
        # Let's use a grid of 6x6 points in the 2D triangle
        # u = i/5, v = j/5 for i,j in 0..5 such that u+v <= 1
        # This gives 21 points.
        # Let's add more by using a finer grid or random points
        # Let's use 36 random points with a fixed seed for diversity
        rng = np.random.default_rng(42)
        for _ in range(n_weights):
            # Dirichlet distribution with alpha=1 gives uniform distribution on simplex
            w = rng.dirichlet([1, 1, 1])
            weights.append(w)
        weights = np.array(weights)

        # 2. Allocate budget
        # We have 200 evaluations.
        # Phase 1: Latin Hypercube Sampling (LHS) for global exploration.
        # Phase 2: Local refinement of the best points found in Phase 1.
        
        # LHS size: Let's use 50 points for LHS.
        n_lhs = 50
        if n_lhs > budget:
            n_lhs = budget
            
        # Generate LHS points
        # LHS ensures that each dimension is divided into n_lhs equal-probability intervals
        # and exactly one sample is placed in each interval per dimension.
        # This provides good coverage of the search space.
        
        # We will use a simple LHS implementation
        # For each dimension, we assign a random permutation of indices 0..n_lhs-1
        # Then map index i to (i + 0.5) / n_lhs (center of the interval)
        # Then scale to bounds
        
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            # Map to [0, 1)
            lhs_points[:, d] = (perm + 0.5) / n_lhs
        # Scale to bounds
        lhs_points = lo + (hi - lo) * lhs_points

        # Track used evaluations
        used = 0
        # Store all evaluated points and their objective values
        # We need to keep track of non-dominated points for refinement
        # Since the evaluator records every call, we don't need to return them,
        # but we need to know which points to refine.
        
        # We will store the LHS points and their values
        lhs_values = np.zeros((n_lhs, n_obj))
        
        # Evaluate LHS points
        for i in range(n_lhs):
            if time.time() > deadline:
                break
            if used >= budget:
                break
            x = lhs_points[i]
            f = problem(x)
            lhs_values[i] = f
            used += 1

        # Identify non-dominated points from the LHS set
        # A point is non-dominated if no other point is better in all objectives
        # and at least one is better in one objective.
        
        def is_dominated(f, F):
            # f is a single point (n_obj,)
            # F is a matrix (n, n_obj)
            # Returns True if f is dominated by any point in F
            # Dominance: F[j] <= f for all j, and F[j] < f for at least one j
            # Since we are minimizing, lower is better.
            # F[j] <= f means F[j] is at least as good as f in all objectives.
            # F[j] < f in at least one objective means F[j] is strictly better in at least one.
            
            # Check if any point in F dominates f
            # F <= f (element-wise)
            # and F < f (element-wise) for at least one row
            
            # Vectorized check
            # leq = np.all(F <= f, axis=1)
            # lt = np.any(F < f, axis=1)
            # dominated = np.any(leq & lt)
            
            # But we must exclude the point itself if it's in F?
            # The LHS points are unique, so we can just check all.
            # However, if we check against the whole set, we might find itself.
            # But since it's unique, it won't be strictly better than itself.
            # So it's fine.
            
            leq = np.all(F <= f, axis=1)
            lt = np.any(F < f, axis=1)
            return np.any(leq & lt)

        # Find non-dominated indices
        non_dominated_indices = []
        for i in range(n_lhs):
            if time.time() > deadline:
                break
            # Check if lhs_values[i] is dominated by any other point in lhs_values
            # We can check against all points, but we can optimize by checking only against the current non-dominated set
            # For simplicity, let's check against all
            if not is_dominated(lhs_values[i], lhs_values):
                non_dominated_indices.append(i)
        
        # If no non-dominated points (should not happen), use all
        if not non_dominated_indices:
            non_dominated_indices = list(range(n_lhs))
            
        # Sort non-dominated points by their scalarized value with a uniform weight
        # to pick the "best" ones for refinement
        uniform_weight = np.ones(n_obj) / n_obj
        scalar_values = lhs_values[non_dominated_indices] @ uniform_weight
        # Sort indices by scalar value
        sorted_indices = np.argsort(scalar_values)
        # Take the top few for refinement
        # We have budget - used evaluations left.
        # Each refinement uses ~10-20 evaluations.
        # Let's refine the top 5-10 non-dominated points.
        
        remaining_budget = budget - used
        # Each refinement run:
        # - Initial simplex: dim+1 = 5 points
        # - Max iterations: ~10
        # Total ~15 evaluations per run.
        # Let's refine up to min(10, remaining_budget // 15) points
        n_refine = min(10, max(0, remaining_budget // 15))
        
        # If n_refine is 0, we can do a final random sampling or just return
        if n_refine > 0:
            # Select the top n_refine non-dominated points
            # We need to map back to the original indices
            top_indices = [non_dominated_indices[i] for i in sorted_indices[:n_refine]]
            
            for idx in top_indices:
                if time.time() > deadline:
                    break
                if used >= budget:
                    break
                
                x0 = lhs_points[idx]
                
                # Define the scalarized objective for local search
                # We use a weight vector that points towards the "center" of the front
                # or just use the uniform weight for simplicity
                # Actually, using a specific weight for each refinement might be better
                # to explore different parts of the front.
                # But since we are refining points already found, using the uniform weight
                # or a weight based on the point's position might be good.
                # Let's use the uniform weight for now.
                
                def scalar_obj(x, w=uniform_weight):
                    nonlocal used
                    if used >= budget:
                        return 0.0
                    f = problem(x)
                    used += 1
                    return np.dot(w, f)

                # Nelder-Mead options
                # We want to stay within bounds.
                # Nelder-Mead doesn't handle bounds well.
                # We can clip the input to the scalar_obj.
                
                def bounded_scalar_obj(x):
                    x_clipped = np.clip(x, lo, hi)
                    return scalar_obj(x_clipped)

                options = {
                    'maxiter': 10,
                    'xatol': 1e-6,
                    'fatol': 1e-6
                }

                try:
                    res = minimize(bounded_scalar_obj, x0, method='Nelder-Mead', options=options)
                except Exception:
                    continue

        # If we have budget left, we can do a final random sampling or refine more
        # But the above should use most of the budget.
        
        # The problem statement says: "Every returned objective vector is recorded"
        # The evaluator calls problem(x) and records the result.
        # My scalar_obj calls problem(x) and increments used.
        # So all these calls are recorded.
        
        # Return something (ignored)
        return None
# EVOLVE-BLOCK-END
