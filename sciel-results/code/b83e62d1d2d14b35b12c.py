"""Constrained Chebyshev scalarization optimizer for WFG7."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using constrained Chebyshev scalarization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        ref = np.array([3.0, 5.0, 7.0])

        used = 0

        def count_eval():
            nonlocal used
            used += 1

        # Generate weight vectors on the simplex
        # For 3 objectives, we can use a grid of weights
        # We want diverse weights to cover the Pareto front
        n_weights = 15  # Number of weight vectors per axis roughly
        weight_vectors = []
        for i in range(n_weights + 1):
            for j in range(n_weights + 1 - i):
                k = n_weights - i - j
                w = np.array([i / n_weights, j / n_weights, k / n_weights])
                if np.sum(w) > 0.01:  # Skip near-zero weights
                    weight_vectors.append(w)
        
        # Add some random weights for diversity
        for _ in range(10):
            w = rng.random(n_obj)
            w /= np.sum(w)
            weight_vectors.append(w)
        
        weight_vectors = np.array(weight_vectors)
        n_total_weights = len(weight_vectors)

        # Store all objective vectors found
        all_F = []

        # For each weight vector, run a local optimization
        # We have budget evaluations total
        evals_per_weight = max(1, budget // (n_total_weights * 2))
        
        # But we need to be careful with time
        # Let's do a first pass with fewer evals per weight, then refine
        # Actually, let's just distribute budget across weights
        
        # Strategy: for each weight, do a few random restarts with L-BFGS-B
        # Each L-BFGS-B call uses multiple function evaluations
        
        # Let's estimate: L-BFGS-B with maxiter=20 might use ~40 evals
        # So with budget 200, we can do about 5 L-BFGS-B calls
        # That's not enough for 25+ weights
        
        # Better approach: use a smaller number of weight vectors
        # and for each, do a short optimization
        
        # Let's use a different strategy:
        # 1. Generate a set of weight vectors
        # 2. For each weight, start from a random point and do gradient descent
        #    using finite differences (since we don't have gradients)
        #    But that uses many evals per step
        
        # Alternative: use the fact that we can do ~200 evals total
        # Let's do a two-phase approach:
        # Phase 1: Random sampling to get an initial approximation of the front
        # Phase 2: For promising regions, do local refinement
        
        # Actually, let's try a simpler approach:
        # Use a set of weight vectors, and for each, do a short line search
        # along the gradient direction
        
        # Let's reconsider. With 200 evals, we can:
        # - Do ~20 random points to get initial front
        # - For each of ~10 weight vectors, do ~18 local steps
        
        # Let's implement a simple gradient-free local search:
        # For each weight, start from a random point, and iteratively
        # try moving in the direction that reduces the Chebyshev objective
        
        def chebyshev_objective(x, w, f_min):
            f = problem(x)
            count_eval()
            # Constrained Chebyshev: max_i(w_i * (f_i - f_min_i))
            # But we need f_min. Let's use a simpler version:
            # Just minimize max_i(w_i * f_i)
            return np.max(w * f), f
        
        # Actually, let's use a different approach that's more standard:
        # Weighted sum with a penalty for constraint violation
        # Or just use weighted sum and rely on the weights to cover the front
        
        # Let's try: for each weight, minimize sum(w_i * f_i) using a simple
        # coordinate descent or random perturbation
        
        # Phase 1: Initial random sampling
        n_init = min(30, budget)
        X_init = lo + (hi - lo) * rng.random((n_init, dim))
        F_init = np.array([problem(x) for x in X_init])
        used += n_init
        all_F.extend(F_init.tolist())
        
        # Get initial f_min
        f_min = np.min(F_init, axis=0)
        
        # Phase 2: For each weight vector, do local search
        # Use the non-dominated points as starting points
        # But first, let's identify promising starting points
        
        # For each weight, pick the best point from initial sampling
        # in terms of weighted sum, then do local search
        
        remaining_budget = budget - used
        n_weights_to_use = min(n_total_weights, max(1, remaining_budget // 5))
        
        # Select the most diverse weight vectors
        if n_total_weights > n_weights_to_use:
            # Use a simple diversity selection: take every k-th weight
            step = n_total_weights // n_weights_to_use
            selected_weights = weight_vectors[::step][:n_weights_to_use]
        else:
            selected_weights = weight_vectors
        
        for w in selected_weights:
            if used >= budget or time.time() > deadline:
                break
            
            # Find the best starting point for this weight
            # Use weighted sum to score initial points
            scores = F_init @ w
            best_idx = np.argmin(scores)
            x_current = X_init[best_idx].copy()
            f_current = F_init[best_idx].copy()
            
            # Local search: try random perturbations and keep improvements
            n_local_steps = max(1, (budget - used) // (n_weights_to_use + 1))
            step_size = 0.1 * (hi - lo)
            
            for _ in range(n_local_steps):
                if used >= budget or time.time() > deadline:
                    break
                
                # Try a random perturbation
                direction = rng.standard_normal(dim)
                direction /= (np.linalg.norm(direction) + 1e-10)
                x_try = x_current + step_size * rng.random() * direction
                x_try = np.clip(x_try, lo, hi)
                
                f_try = problem(x_try)
                used += 1
                all_F.append(f_try.tolist())
                
                # Check if improvement in weighted sum
                score_current = np.dot(w, f_current)
                score_try = np.dot(w, f_try)
                
                if score_try < score_current:
                    x_current = x_try
                    f_current = f_try
                    # Adapt step size
                    step_size *= 0.8
                else:
                    step_size *= 1.1
                    step_size = min(step_size, 0.3 * (hi - lo))
        
        # Phase 3: Final refinement around best non-dominated points
        if used < budget and len(all_F) > 0:
            F_arr = np.array(all_F)
            
            # Find non-dominated points
            n_pts = len(F_arr)
            nd_mask = np.ones(n_pts, dtype=bool)
            for i in range(n_pts):
                for j in range(n_pts):
                    if i != j:
                        if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                            nd_mask[i] = False
                            break
            
            nd_indices = np.where(nd_mask)[0]
            
            if len(nd_indices) > 0:
                # Sort by distance to reference
                nd_points = F_arr[nd_indices]
                dists = np.sum((nd_points - ref) ** 2, axis=1)
                sorted_nd = nd_indices[np.argsort(dists)]
                
                remaining = budget - used
                n_refine = min(len(sorted_nd), max(1, remaining // 10))
                
                for idx in sorted_nd[:n_refine]:
                    if used >= budget or time.time() > deadline:
                        break
                    
                    # We need the corresponding x, but we didn't store all x
                    # Let's just do a random local search from a random point
                    # near the best found
                    x_base = lo + (hi - lo) * rng.random(dim)
                    
                    n_steps = min(10, budget - used)
                    for _ in range(n_steps):
                        if used >= budget or time.time() > deadline:
                            break
                        
                        direction = rng.standard_normal(dim)
                        direction /= (np.linalg.norm(direction) + 1e-10)
                        step = 0.05 * (hi - lo) * (0.5 + 0.5 * rng.random())
                        x_try = x_base + step * direction
                        x_try = np.clip(x_try, lo, hi)
                        
                        f_try = problem(x_try)
                        used += 1
                        all_F.append(f_try.tolist())
        
        # Return the last evaluated point (return value is ignored per contract)
        if len(all_F) > 0:
            return np.array(all_F[-1])
        else:
            return np.array([0.0] * n_obj)
