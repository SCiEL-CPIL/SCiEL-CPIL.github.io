"""Model-and-Optimize Optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer using a two-phase strategy:
    1. Coarse global screening with short local optimizations.
    2. Intensive Powell's method on the best candidates.
    3. Final polish with Nelder-Mead.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 300s limit, keep 20s safety margin
        deadline = start_time + 280.0
        
        dim = self.dim
        budget = self.budget
        
        global_best_x = None
        global_best_f = float('inf')
        calls_made = 0
        best_x_history = []

        def call_func(x):
            nonlocal calls_made, global_best_x, global_best_f
            if calls_made >= budget:
                return global_best_f
            val = func(x)
            calls_made += 1
            if val < global_best_f:
                global_best_f = val
                global_best_x = x.copy()
                best_x_history.append(x.copy())
            return val

        # --- Phase 1: Coarse Global Screening ---
        # Generate diverse starting points to cover the space
        # 1. Center
        # 2. Points biased towards 1.0 and -1.0
        # 3. Stratified random points
        # 4. Pure random points
        
        starts = []
        rng = np.random.default_rng(42)
        
        # Center
        starts.append(np.zeros(dim))
        
        # Biased points (common locations for minima in BBOB)
        starts.append(np.ones(dim) * 1.0)
        starts.append(np.ones(dim) * -1.0)
        starts.append(np.ones(dim) * 2.0)
        starts.append(np.ones(dim) * -2.0)
        
        # Stratified points: 2x2x2x2x2 = 32 points
        # This provides good coverage without too many points
        # We will take a subset if budget is tight, but 32 is small enough
        # Let's generate 4 points per dimension level
        levels = [-4.0, -2.0, 0.0, 2.0, 4.0]
        # We can't do 5^5 = 3125.
        # Let's use a sparse grid or random stratification.
        # Simple approach: 20 random points from uniform distribution
        for _ in range(20):
            starts.append(rng.uniform(-5, 5, dim))
            
        # Sort starts by some heuristic? No, just run them.
        # We will run a very short optimization from each start to "screen" them.
        
        num_starts = len(starts)
        calls_per_screen = 5
        candidates = []
        
        for start_x in starts:
            if time.time() > deadline or calls_made >= budget:
                break
            
            # If we are running low on budget, skip remaining starts
            if calls_made + calls_per_screen > budget:
                break

            try:
                # Use Powell's method for a very short run
                # maxfev=5 is very short, but it might move in a good direction
                from scipy.optimize import minimize
                res = minimize(
                    call_func, 
                    start_x, 
                    method='Powell', 
                    options={'maxfev': calls_per_screen, 'maxiter': 10}
                )
                # Add the result and the starting point itself to candidates
                candidates.append(res.x)
                candidates.append(start_x)
            except Exception:
                candidates.append(start_x)

        # Deduplicate candidates roughly
        unique_candidates = []
        for c in candidates:
            if not any(np.linalg.norm(c - uc) < 0.5 for uc in unique_candidates):
                unique_candidates.append(c)
        
        # Evaluate all unique candidates to ensure we have their true values
        # Note: call_func updates global_best, so we just need to ensure they are evaluated
        # However, minimize already evaluated res.x. start_x might not be evaluated if minimize failed immediately?
        # Actually, minimize evaluates the starting point first.
        # So all candidates are likely evaluated.
        
        # Select top K candidates for intensive refinement
        # We need to sort by function value.
        # Since call_func updates global_best, we don't have a direct list of (x, f) pairs for all candidates.
        # We can re-evaluate the top few unique candidates to get their scores, or just pick a few diverse ones.
        # Better: Keep track of the best few points found during screening.
        
        # Let's re-evaluate the unique candidates to get their scores for sorting
        # This costs extra calls, but we have budget.
        # However, we must be careful not to exceed budget.
        # Let's just pick the first 5 unique candidates that are "good" (i.e., not too far from global_best_x if we have one, or just diverse).
        
        # Actually, a simpler heuristic:
        # Run intensive Powell's on the global_best_x found so far, and 2 other diverse points.
        
        top_candidates = []
        if global_best_x is not None:
            top_candidates.append(global_best_x)
            
        # Add 2 other diverse points from unique_candidates
        for c in unique_candidates:
            if np.linalg.norm(c - global_best_x) > 1.0:
                top_candidates.append(c)
                if len(top_candidates) >= 3:
                    break
        
        # Ensure we have at least 1 candidate
        if not top_candidates:
            top_candidates.append(np.zeros(dim))

        # --- Phase 2: Intensive Refinement ---
        # Allocate remaining budget to these candidates
        # Reserve ~50 calls for final polish
        remaining_budget = budget - calls_made - 50
        if remaining_budget < 0:
            remaining_budget = max(0, budget - calls_made)
            
        calls_per_refine = max(10, remaining_budget // len(top_candidates))
        
        for b_start in top_candidates:
            if time.time() > deadline or calls_made >= budget:
                break
            
            # If we are running low on budget, skip
            if calls_made + calls_per_refine > budget - 50:
                # Use whatever is left
                calls_per_refine = max(5, budget - 50 - calls_made)
                if calls_per_refine <= 0:
                    break

            try:
                # Use Powell's method for intensive refinement
                res = minimize(
                    call_func, 
                    b_start, 
                    method='Powell', 
                    options={'maxfev': calls_per_refine, 'maxiter': 100}
                )
            except Exception:
                pass

        # --- Phase 3: Final Polish ---
        # Use Nelder-Mead for high precision on the global best
        if global_best_x is not None and calls_made < budget and time.time() < deadline:
            polish_budget = min(50, budget - calls_made)
            if polish_budget > 5:
                try:
                    res = minimize(
                        call_func, 
                        global_best_x, 
                        method='Nelder-Mead', 
                        options={
                            'xatol': 1e-10, 
                            'fatol': 1e-12, 
                            'maxfev': polish_budget, 
                            'maxiter': 200
                        }
                    )
                    if res.fun < global_best_f:
                        global_best_f = res.fun
                        global_best_x = res.x.copy()
                except Exception:
                    pass

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
