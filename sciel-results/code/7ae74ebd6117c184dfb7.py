"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """DFO Optimizer using Random Search + Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        d = self.dim
        budget = self.budget
        
        eval_count = 0
        best_f = np.inf
        best_x = None

        # Helper to evaluate and track
        def evaluate(x):
            nonlocal eval_count, best_f, best_x
            if eval_count >= budget:
                return np.inf
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Global Exploration with Random Sampling
        # We use 10% of budget for initial global search, minimum 10 points, max 50
        n_global = min(50, max(10, budget // 10))
        
        # Sample points uniformly in [-4, 4]^d to stay safely inside [-5, 5]^d
        # But we can sample in [-5, 5] as func clips. Let's use [-4.5, 4.5] to be safe.
        candidates = []
        for _ in range(n_global):
            x = rng.uniform(-4.5, 4.5, d)
            candidates.append(x)
        
        # Evaluate all candidates
        f_scores = []
        for x in candidates:
            fx = evaluate(x)
            f_scores.append(fx)
            
        if eval_count >= budget:
            return best_x, best_f

        # Sort candidates by fitness
        sorted_indices = np.argsort(f_scores)
        
        # Phase 2: Local Refinement on Top Candidates
        # We will spend the remaining budget on refining the best 1-3 candidates.
        # Since NM is cheap per iteration in low dims, we can afford several restarts.
        
        # Determine how many top candidates to refine
        # If we have lots of budget, refine more.
        n_refine = 3
        
        # For each top candidate, run Nelder-Mead
        # We allocate remaining budget among them.
        remaining_budget = budget - eval_count
        budget_per_refine = remaining_budget // n_refine
        
        # Ensure at least 10 evaluations per refine if possible
        if budget_per_refine < 10:
            n_refine = 1
            budget_per_refine = remaining_budget

        for i in range(min(n_refine, len(sorted_indices))):
            if eval_count >= budget:
                break
            
            start_idx = sorted_indices[i]
            x0 = candidates[start_idx].copy()
            
            # Define wrapper for Nelder-Mead
            # NM in SciPy uses its own internal call count logic, but we need to strictly limit calls.
            # We will use a custom loop or careful options.
            # SciPy's minimize with method='Nelder-Mead' stops on maxiter or tol.
            # We need to count calls.
            
            def nm_wrapper(x):
                nonlocal eval_count
                if eval_count >= budget:
                    return best_f # Return current best to stop progress if budget hit
                return evaluate(x)

            # Initial simplex size
            # Standard NM uses initial_simplex which is x0 + 0.05 * x0 (or 0.00025 if x0 is 0)
            # We can provide initial_simplex for better control.
            # Let's use a scaled simplex around x0.
            # Scale 0.5 seems reasonable for a function with range [-5,5].
            # But NM handles scaling internally somewhat.
            # Let's just rely on default or provide a small perturbation.
            
            # To improve convergence, we can restart NM from the current best if it gets stuck.
            # But for now, let's just run NM on the top candidates.
            
            try:
                # Use xatol and fatol to stop early if converged
                # maxiter is the max number of function evaluations? No, it's iterations.
                # In 5D, 1 iteration is ~6 calls.
                # So maxiter = budget_per_refine // 6 is a safe upper bound.
                max_iter = max(10, budget_per_refine // 6)
                
                res = minimize(nm_wrapper, x0, method='Nelder-Mead', 
                               options={'maxiter': max_iter, 
                                        'xatol': 1e-6, 
                                        'fatol': 1e-8,
                                        'adaptive': True})
                
                # Update best if NM found something better
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
                    
            except Exception:
                pass

        # Phase 3: Final Polish on the Global Best
        # If we still have budget left, do one more NM run from the absolute best point found so far.
        if eval_count < budget - 5 and best_x is not None:
            remaining = budget - eval_count
            max_iter = max(10, remaining // 6)
            
            def final_wrapper(x):
                nonlocal eval_count
                if eval_count >= budget:
                    return best_f
                return evaluate(x)
                
            try:
                # Perturb best_x slightly to escape potential local minima if NM got stuck
                # Or just run NM again. NM is deterministic given start point.
                # If it converged, running again might not help.
                # Let's try a small random perturbation to restart exploration.
                x_start = best_x + rng.normal(0, 0.1, d)
                x_start = np.clip(x_start, -5, 5)
                
                res = minimize(final_wrapper, x_start, method='Nelder-Mead', 
                               options={'maxiter': max_iter, 
                                        'xatol': 1e-8, 
                                        'fatol': 1e-10,
                                        'adaptive': True})
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
                    
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
