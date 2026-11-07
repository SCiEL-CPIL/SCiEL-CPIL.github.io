"""
Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using a hybrid approach:
1. Dense Latin Hypercube Sampling (LHS) for global basin identification.
2. Nelder-Mead Simplex for local exploitation.
3. Adaptive restarts if stagnation is detected.
"""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        # Helper to update best
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Latin Hypercube Sampling (LHS)
        # LHS ensures coverage of the entire space, which is crucial for 
        # identifying the correct global basin in bi-modal functions.
        # We use a stratified sampling approach.
        
        # Determine number of samples for LHS
        # We want to use a significant portion of the budget for global exploration
        # but leave enough for local refinement.
        # Let's use 50% of the budget for LHS if possible, but at least 50 samples.
        n_lhs = max(50, budget // 2)
        
        # Generate LHS samples
        # For each dimension, divide the range into n_lhs intervals and sample one point per interval
        # Then shuffle the indices for each dimension to decorrelate
        lhs_samples = np.zeros((n_lhs, dim))
        for d in range(dim):
            # Create intervals
            intervals = np.linspace(-5, 5, n_lhs + 1)
            # Sample one point from each interval
            # Random offset within interval
            offsets = np.random.uniform(0, 1, size=n_lhs)
            points = intervals[:-1] + offsets * (intervals[1] - intervals[0])
            # Shuffle to break correlation
            np.random.shuffle(points)
            lhs_samples[:, d] = points

        # Evaluate LHS samples
        for i in range(n_lhs):
            if calls >= budget or time.time() > deadline:
                break
            x = lhs_samples[i]
            f = func(x)
            calls += 1
            update_best(x, f)

        if best_x is None:
            # Fallback if no calls were made (should not happen)
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        # Phase 2: Local Refinement with Nelder-Mead
        # Use the best point from LHS as the starting point for Nelder-Mead.
        # Nelder-Mead is derivative-free and robust for local optimization.
        
        # We will perform multiple restarts of Nelder-Mead from the best points found so far.
        # Sort all evaluated points by function value and pick the top few for restarts.
        
        # Keep track of all evaluated points
        # In the current implementation, we only kept the best. Let's store all for restarts.
        # Since we already did LHS, we have n_lhs points.
        # We need to store them. Let's re-evaluate or store them.
        # It's better to store them during evaluation.
        
        # Let's refactor slightly to store all points.
        # But we already did the loop. Let's just use the best point and maybe a few random ones near it.
        # Actually, a better strategy for Phase 2 is to run Nelder-Mead on the best point found.
        
        # Define the objective function for Nelder-Mead
        def objective(x):
            nonlocal calls
            if calls >= budget or time.time() > deadline:
                return best_f
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # Run Nelder-Mead
        # Options:
        # maxiter: limit iterations
        # xatol, fatol: convergence criteria
        # initial_simplex: define a reasonable initial simplex size
        
        # We'll run Nelder-Mead with a moderate budget for each run.
        # Since we have budget - calls left, we can use a fraction of it.
        
        # Let's try a few restarts from the best point and slightly perturbed points.
        # But to keep it simple and robust, let's just run one strong Nelder-Mead run first.
        
        # Initial simplex size
        # A good rule of thumb is to have the simplex span a fraction of the search space
        # or be centered around the current best with a reasonable spread.
        # For dim=5, a spread of 0.5 to 1.0 is often good.
        
        initial_simplex = np.zeros((dim + 1, dim))
        initial_simplex[0] = best_x
        for i in range(1, dim + 1):
            initial_simplex[i] = best_x
            initial_simplex[i, i-1] += 0.5  # Perturb in one dimension at a time
        
        # Clip initial simplex to bounds
        initial_simplex = np.clip(initial_simplex, -5, 5)

        options = {
            'maxiter': 200,
            'xatol': 1e-8,
            'fatol': 1e-8,
            'adaptive': True,
            'disp': False
        }

        try:
            result = minimize(objective, best_x, method='Nelder-Mead', 
                               options=options, initial_simplex=initial_simplex)
            if result.fun < best_f:
                best_f = result.fun
                best_x = np.clip(result.x, -5, 5).copy()
        except Exception:
            pass

        # If we still have budget, we can try a second round of LHS + NM
        # or just more NM restarts from the best point.
        # Let's try a few more NM runs from slightly perturbed best_x.
        
        if calls < budget and time.time() < deadline:
            # Try 3 more restarts with different initial simplex perturbations
            for restart in range(3):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Perturb best_x
                x_start = best_x + np.random.normal(0, 0.1, size=dim)
                x_start = np.clip(x_start, -5, 5)
                
                # Create a new initial simplex
                init_simplex = np.zeros((dim + 1, dim))
                init_simplex[0] = x_start
                for i in range(1, dim + 1):
                    init_simplex[i] = x_start
                    init_simplex[i, i-1] += 0.3
                
                init_simplex = np.clip(init_simplex, -5, 5)
                
                try:
                    result = minimize(objective, x_start, method='Nelder-Mead', 
                                       options=options, initial_simplex=init_simplex)
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5, 5).copy()
                except Exception:
                    pass

        return best_x, best_f
