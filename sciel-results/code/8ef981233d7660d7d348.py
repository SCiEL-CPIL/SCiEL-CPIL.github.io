"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid Global-Local Optimizer using Hooke-Jeeves Pattern Search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        
        # State tracking
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        def check_time():
            return time.time() > deadline
        
        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            if check_time():
                return None
            try:
                val = func(x)
            except Exception:
                return None
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Global Coarse Grid Search (Deterministic)
        # In 5D, a 3^5 = 243 point grid is feasible within 500 calls.
        # This ensures we find the basin of attraction for the rotated Rastrigin.
        # We use a slightly expanded grid to cover [-5, 5] robustly.
        # Points: -5, 0, 5 is too coarse. Let's use 4 points per dimension? 4^5 = 1024 > 500.
        # 3 points per dimension: 3^5 = 243.
        # To be safe against rotation, let's use a mix.
        
        # Strategy: 
        # 1. Center (1 call)
        # 2. Axis-aligned coarse grid (3^5 = 243 calls)
        # 3. Remaining budget for local refinement.
        
        # Initialize with center
        x0 = np.zeros(n)
        f0 = evaluate(x0)
        
        if f0 is None:
            return best_x, best_f

        # Generate 3^5 grid points
        # Range: [-4, 0, 4] covers most of the search space where minima lie in [-4,4]
        # Actually, the problem says minimizer in [-4,4]. 
        # Let's use grid points: -4, 0, 4.
        grid_vals = [-4.0, 0.0, 4.0]
        
        # Generate all combinations
        grid_points = np.array(np.meshgrid(*[grid_vals]*n)).T.reshape(-1, n)
        
        # Shuffle to spread risk? No, deterministic is fine for reproducibility, 
        # but random order might hit the good spot earlier for AOCC.
        # Let's shuffle the grid points to improve early progress (AOCC).
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(grid_points))
        grid_points = grid_points[perm]
        
        for p in grid_points:
            if calls_made >= budget or check_time():
                break
            f = evaluate(p)
            if f is None:
                break
            
        # Phase 2: Local Refinement using Hooke-Jeeves
        # Use remaining budget to refine the best point found.
        
        # Parameters for HJ
        step_sizes = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
        
        current_x = best_x.copy()
        current_f = best_f
        
        # Adaptive step size logic
        # We will cycle through step sizes, reducing them if no improvement.
        
        # To maximize AOCC, we want rapid initial drop. 
        # The grid search likely found a decent spot. 
        # Now we do a local pattern search.
        
        # We'll implement a simple coordinate descent with adaptive steps.
        # For each dimension, try +/- step.
        
        # To be efficient, we'll use a "ratchet" approach:
        # Start with larger steps, shrink when stuck.
        
        active_step_idx = 0
        current_step = step_sizes[0]
        
        # We will loop until budget or time runs out
        # Each "iteration" tries to improve in all dimensions.
        
        # Optimization: Vectorize the HJ logic if possible? 
        # No, sequential is safer for budget counting.
        
        # Let's use a more aggressive local search: 
        # Nelder-Mead is good, but HJ is deterministic and simple.
        # Given 5D, HJ is very fast per call.
        
        while calls_made < budget and not check_time():
            improved = False
            for i in range(n):
                if calls_made >= budget or check_time():
                    break
                
                # Try positive step
                x_try = current_x.copy()
                x_try[i] += current_step
                f_try = evaluate(x_try)
                
                if f_try is not None and f_try < current_f:
                    current_x = x_try
                    current_f = f_try
                    improved = True
                else:
                    # Try negative step
                    x_try = current_x.copy()
                    x_try[i] -= current_step
                    f_try = evaluate(x_try)
                    
                    if f_try is not None and f_try < current_f:
                        current_x = x_try
                        current_f = f_try
                        improved = True
                    else:
                        # If no improvement in this direction, we might shrink step
                        # But we check shrinkage after full pass or specific condition
                        pass
            
            if not improved:
                # Shrink step size
                active_step_idx += 1
                if active_step_idx >= len(step_sizes):
                    # Convergence reached or steps too small
                    break
                current_step = step_sizes[active_step_idx]
            else:
                # If improved, keep step size or potentially expand? 
                # For safety and smoothness, keep step size.
                # If we want to be faster, we could expand, but shrinking is safer for AOCC tail.
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
