"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective optimizer using weighted-sum L-BFGS-B + evolutionary refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1705.0, 12.0, 0.3])
        
        used = 0
        solutions = []  # list of (x, f)
        
        def eval_point(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            used += 1
            return x, f
        
        def time_check():
            return time.time() > deadline
        
        # Phase 1: Latin Hypercube Sampling for initial spread
        n_lhs = 30
        X_lhs = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            perm = rng.permutation(n_lhs)
            X_lhs[:, d] = (perm + rng.random(n_lhs)) / n_lhs
        X_lhs = lo + X_lhs * (hi - lo)
        
        for x in X_lhs:
            if used >= self.budget or time_check():
                break
            x_c, f = eval_point(x)
            solutions.append((x_c, f))
        
        # Phase 2: Weighted-sum L-BFGS-B optimization
        # Generate diverse weight vectors on the 2-simplex (3 objectives)
        # Use stratified sampling on the simplex for good coverage
        n_weights = 40
        
        # Stratified sampling on simplex: divide into regions
        # For 3 objectives, use a grid of (w1, w2) with w3 = 1-w1-w2
        weight_vectors = []
        # Grid points
        n_grid = 6
        for i in range(n_grid + 1):
            for j in range(n_grid + 1 - i):
                w1 = i / n_grid
                w2 = j / n_grid
                w3 = 1.0 - w1 - w2
                if w3 >= 0:
                    weight_vectors.append(np.array([w1, w2, w3]))
        
        # Add some random weights for diversity
        for _ in range(10):
            w = rng.dirichlet(np.ones(self.n_obj))
            weight_vectors.append(w)
        
        # Shuffle and take up to n_weights
        rng.shuffle(weight_vectors)
        weight_vectors = weight_vectors[:n_weights]
        
        # For each weight vector, run L-BFGS-B from a few starting points
        # Use best non-dominated points as starting points
        for w in weight_vectors:
            if used >= self.budget or time_check():
                break
            
            # Select starting points: use a few random solutions + best
            start_points = []
            
            # Add a few random starting points
            n_starts = 3
            for _ in range(n_starts):
                if used >= self.budget or time_check():
                    break
                x0 = lo + rng.random(self.dim) * (hi - lo)
                x0, f0 = eval_point(x0)
                start_points.append(x0)
                solutions.append((x0, f0))
            
            # Also add some existing solutions as starting points
            if len(solutions) > 0:
                # Pick a few diverse existing solutions
                n_existing_starts = min(2, len(solutions))
                idxs = rng.choice(len(solutions), size=n_existing_starts, replace=False)
                for idx in idxs:
                    start_points.append(solutions[idx][0].copy())
            
            # Run L-BFGS-B from each starting point
            for x0 in start_points:
                if used >= self.budget or time_check():
                    break
                
                def objective(x):
                    nonlocal used
                    x = np.clip(x, lo, hi)
                    f = np.array(problem(x))
                    used += 1
                    return np.dot(w, f)
                
                def objective_no_count(x):
                    x = np.clip(x, lo, hi)
                    f = np.array(problem(x))
                    return np.dot(w, f)
                
                # Use scipy minimize with L-BFGS-B
                # We need to be careful: each function call counts as an evaluation
                # L-BFGS-B may call the objective many times
                # We'll limit by checking budget in the objective
                bounds_scipy = list(zip(lo, hi))
                
                try:
                    res = minimize(
                        objective,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds_scipy,
                        options={'maxiter': 50, 'maxfun': 30}
                    )
                    if time_check() or used >= self.budget:
                        break
                    # Evaluate the result
                    x_res = np.clip(res.x, lo, hi)
                    x_res, f_res = eval_point(x_res)
                    solutions.append((x_res, f_res))
                except Exception:
                    pass
        
        # Phase 3: Use remaining budget for targeted refinement
        # Find the current Pareto front and do local search around it
        if used < self.budget and not time_check():
            # Get non-dominated solutions
            n_sols = len(solutions)
            if n_sols > 0:
                F_arr = np.array([f for _, f in solutions])
                dominated = np.zeros(n_sols, dtype=bool)
                for i in range(n_sols):
                    for j in range(n_sols):
                        if i != j:
                            if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                                dominated[i] = True
                                break
                nd_idx = np.where(~dominated)[0]
                
                # For each non-dominated point, do a small local search
                for idx in nd_idx:
                    if used >= self.budget or time_check():
                        break
                    
                    x_center = solutions[idx][0]
                    # Do a few random perturbations and local optimization
                    for _ in range(2):
                        if used >= self.budget or time_check():
                            break
                        # Random perturbation
                        scale = 0.1 * (hi - lo)
                        x_pert = x_center + scale * rng.standard_normal(self.dim)
                        x_pert = np.clip(x_pert, lo, hi)
                        x_pert, f_pert = eval_point(x_pert)
                        solutions.append((x_pert, f_pert))
                        
                        # Local L-BFGS-B from this point with a random weight
                        w = rng.dirichlet(np.ones(self.n_obj))
                        
                        def objective2(x):
                            nonlocal used
                            x = np.clip(x, lo, hi)
                            f = np.array(problem(x))
                            used += 1
                            return np.dot(w, f)
                        
                        bounds_scipy = list(zip(lo, hi))
                        try:
                            res = minimize(
                                objective2,
                                x_pert,
                                method='L-BFGS-B',
                                bounds=bounds_scipy,
                                options={'maxiter': 30, 'maxfun': 20}
                            )
                            if time_check() or used >= self.budget:
                                break
                            x_res = np.clip(res.x, lo, hi)
                            x_res, f_res = eval_point(x_res)
                            solutions.append((x_res, f_res))
                        except Exception:
                            pass
        
        # Return the last solution (return value is ignored by evaluator)
        if len(solutions) > 0:
            return solutions[-1][0]
        return lo.copy()
# EVOLVE-BLOCK-END
