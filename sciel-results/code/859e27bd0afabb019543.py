"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    """Adaptive Weighted-Sum Scalarization with L-BFGS-B for Multi-objective Optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin against 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # 1. Global Coverage: Latin Hypercube Sampling
        # Use ~30% of budget for initial global sampling to ensure we don't miss any regions
        n_lhs = max(10, int(budget * 0.3))
        
        # Generate LHS points
        lhs_points = np.zeros((n_lhs, dim))
        for j in range(dim):
            u = np.random.rand(n_lhs)
            points = (np.arange(n_lhs) + u) / n_lhs
            np.random.shuffle(points)
            lhs_points[:, j] = points
            
        # Scale to bounds
        X_lhs = lo + (hi - lo) * lhs_points
        
        # Evaluate LHS points
        F = np.zeros((n_lhs, n_obj))
        eval_count = 0
        for i in range(n_lhs):
            if time.time() > deadline:
                break
            try:
                F[i] = problem(X_lhs[i])
                eval_count += 1
            except Exception:
                F[i] = np.full(n_obj, 1e6) # Penalty

        if eval_count >= budget or time.time() > deadline:
            return None

        # 2. Local Refinement with Weighted-Sum Scalarization
        # We will iteratively pick a non-dominated point, create a weighted sum scalarization
        # tangent to the front at that point (or simply towards that point), and minimize it
        # using L-BFGS-B. This allows us to explore the front systematically.

        remaining_budget = budget - eval_count
        
        # Helper to check dominance
        def is_dominated(point, points):
            for p in points:
                if np.all(p <= point) and np.any(p < point):
                    return True
            return False

        # Initialize list of known non-dominated points
        # We keep a list of (x, f) pairs
        archive = []
        for i in range(n_lhs):
            if not is_dominated(F[i], F):
                archive.append((X_lhs[i].copy(), F[i].copy()))
        
        # If archive is empty (unlikely but possible with bad luck), add best point
        if not archive:
            idx = np.argmin(np.sum(F, axis=1))
            archive.append((X_lhs[idx].copy(), F[idx].copy()))

        # Define scalarization objective
        # We want to minimize a weighted sum. To ensure we move along the front,
        # we can use the weight vector derived from the normal to the hyperplane
        # defined by the current non-dominated point and the reference point,
        # or simply use random weights biased towards the current point's direction.
        
        # A simpler and robust approach: 
        # For each iteration, pick a point from the archive.
        # Define weights w such that the weighted sum is minimized near that point.
        # Actually, standard weighted sum minimization finds a specific point on the front 
        # for a given weight vector. To explore the whole front, we should vary the weights.
        
        # Strategy:
        # 1. Sample random weight vectors w (normalized to sum to 1).
        # 2. Minimize sum(w_i * f_i(x)) using L-BFGS-B.
        # 3. If the result is non-dominated, add to archive.
        
        # How many scalarizations can we run?
        # L-BFGS-B might take several function evaluations per run.
        # Let's estimate ~10-20 evals per run.
        # We have ~70% of budget left.
        # If budget=200, n_lhs=60, remaining=140.
        # We can run about 7-10 scalarizations.
        
        # To be more efficient, we can use the archive points to guide the weights.
        # However, random weights on the simplex are a good baseline for convex fronts.
        # RE37 is a standard test problem, often with a convex or slightly non-convex front.
        
        # Let's allocate a fixed number of scalarization runs based on remaining budget.
        # Assume ~15 evals per run.
        n_runs = max(1, int(remaining_budget / 15))
        
        # Generate random weight vectors
        # We can also include weights derived from the archive to ensure we refine known good areas.
        weight_vectors = []
        
        # Add some random weights
        for _ in range(n_runs):
            w = np.random.rand(n_obj)
            w /= np.sum(w)
            weight_vectors.append(w)
        
        # Sort weight vectors to ensure systematic coverage? 
        # Or just run them.
        
        for w in weight_vectors:
            if time.time() > deadline:
                break
            
            # Define scalar objective
            def scalar_obj(x):
                try:
                    f = problem(x)
                    eval_count += 1
                    return np.dot(w, f)
                except Exception:
                    return 1e6
            
            # Choose a starting point.
            # Using the best point from the archive for the current weight might help convergence.
            # Let's find the archive point that minimizes the weighted sum.
            best_start_idx = 0
            best_start_val = np.inf
            for i, (x_a, f_a) in enumerate(archive):
                val = np.dot(w, f_a)
                if val < best_start_val:
                    best_start_val = val
                    best_start_idx = i
            
            x0 = archive[best_start_idx][0].copy()
            
            # Run L-BFGS-B
            try:
                res = minimize(scalar_obj, x0, method='L-BFGS-B', bounds=list(zip(lo, hi)), options={'maxiter': 20, 'maxfun': 30})
            except Exception:
                continue
            
            x_opt = res.x
            f_opt = np.array(problem(x_opt))
            eval_count += 1
            
            # Check if non-dominated
            if not is_dominated(f_opt, [f for _, f in archive]):
                archive.append((x_opt.copy(), f_opt.copy()))
                
        return None
# EVOLVE-BLOCK-END
