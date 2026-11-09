"""Hybrid LHS + Nelder-Mead + L-BFGS-B Optimizer for BBOB f20."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Hybrid optimizer combining:
    1. Latin Hypercube Sampling (LHS) for broad initial exploration.
    2. Nelder-Mead local search for robust refinement in multimodal landscapes.
    3. L-BFGS-B for high-precision convergence in the final basin.
    
    This approach is designed to maximize AOCC by ensuring rapid early 
    improvement (via LHS) and deep late-stage convergence (via L-BFGS-B).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 295  # Safety margin for 300s limit
        
        # Fixed seed for reproducibility in evaluation, though instances are distinct
        rng = np.random.default_rng(42)
        
        # State tracking
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Helper to track best
        def update_best(x, f_val):
            nonlocal best_x, best_f
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
        
        # Phase 1: Strategic Exploration with Latin Hypercube Sampling
        # LHS ensures that each dimension is uniformly sampled, providing better
        # coverage than pure random sampling for the same number of points.
        # We allocate ~20% of the budget to this phase.
        n_explore = max(10, budget // 5)  # At least 10, 20% of budget
        if n_explore > budget:
            n_explore = budget
            
        # Generate LHS points
        # For LHS, we divide each dimension into n_explore bins and pick one point per bin
        lhs_points = np.zeros((n_explore, dim))
        for d in range(dim):
            # Create bins: [0, 1/n, 2/n, ...]
            bins = np.linspace(0, 1, n_explore + 1)
            # Randomly permute bin indices to ensure stratification
            perm = rng.permutation(n_explore)
            # Pick a uniform random point within each bin
            for i in range(n_explore):
                low = bins[perm[i]]
                high = bins[perm[i] + 1]
                lhs_points[i, d] = rng.uniform(low, high)
        
        # Map from [0, 1]^dim to [-5, 5]^dim
        lhs_points = lhs_points * 10.0 - 5.0
        
        # Evaluate LHS points
        f_vals_lhs = np.zeros(n_explore)
        for i in range(n_explore):
            if time.time() > deadline:
                break
            x = lhs_points[i]
            f_val = func(x)
            calls_made += 1
            f_vals_lhs[i] = f_val
            update_best(x, f_val)
            
        if best_x is None:
            # Fallback if no calls were made (should not happen)
            return np.zeros(dim), 0.0

        # Phase 2: Adaptive Local Refinement with Nelder-Mead
        # Select the top K candidates from the exploration phase.
        # Nelder-Mead is robust to the multimodality of Schwefel's function
        # and does not require gradient information.
        k_refine = min(5, n_explore)  # Refine top 5
        top_indices = np.argsort(f_vals_lhs)[:k_refine]
        
        remaining_budget = budget - calls_made
        # Allocate budget for Nelder-Mead
        # Nelder-Mead typically needs O(dim * iterations) calls.
        # With dim=5, let's allow ~200 calls per local search if budget permits.
        # But we have multiple starts. Let's split remaining budget among top candidates.
        nm_calls_per_start = max(20, remaining_budget // (k_refine * 2))
        
        nm_results = []
        for idx in top_indices:
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break
                
            x0 = lhs_points[idx].copy()
            
            # Define objective function wrapper for Nelder-Mead
            def nm_func(x):
                nonlocal calls_made
                if calls_made >= budget or time.time() > deadline:
                    return np.inf
                x_clipped = np.clip(x, -5, 5)
                f_val = func(x_clipped)
                calls_made += 1
                update_best(x_clipped, f_val)
                return f_val
            
            try:
                res = minimize(
                    nm_func, 
                    x0, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 100, 
                        'xatol': 1e-4, 
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
                if res.x is not None and res.fun < np.inf:
                    nm_results.append(res.x)
            except Exception:
                pass

        # Phase 3: Precision Polishing with L-BFGS-B
        # L-BFGS-B uses gradient approximation and converges very fast near the optimum.
        # This is crucial for the late-stage AOCC score.
        
        # Collect candidates for polishing:
        # 1. Current global best
        # 2. Best results from Nelder-Mead
        candidates = [best_x]
        for res_x in nm_results:
            if res_x is not None:
                # Avoid duplicates
                if not any(np.allclose(res_x, c) for c in candidates):
                    candidates.append(res_x)
        
        # Also add the best LHS point if not already there
        best_lhs_idx = np.argmin(f_vals_lhs)
        if not any(np.allclose(lhs_points[best_lhs_idx], c) for c in candidates):
            candidates.append(lhs_points[best_lhs_idx])

        # Shuffle candidates to ensure we don't always polish the same one first if multiple are good
        # But prioritize the current best first? Actually, L-BFGS-B is cheap in terms of calls if it converges fast.
        # Let's polish the top 2 distinct candidates.
        
        # Sort candidates by their known best function value (approximate)
        # We don't have exact f for all candidates in the list if they came from NM, 
        # but we know best_x is the global best.
        
        # Allocate remaining budget to L-BFGS-B
        remaining_budget = budget - calls_made
        if remaining_budget > 0:
            # Try to polish the best candidate first
            # L-BFGS-B maxiter is limited to ensure we don't blow the budget
            # Each iteration does 1-2 function calls for gradient estimation (finite diff).
            # So maxiter should be conservative.
            max_iter_lbfgs = max(10, remaining_budget // 10)
            
            for x0 in candidates[:2]:  # Polish top 2
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break
                
                # Define objective function wrapper for L-BFGS-B
                def lbfgs_func(x):
                    nonlocal calls_made
                    if calls_made >= budget or time.time() > deadline:
                        return np.inf
                    x_clipped = np.clip(x, -5, 5)
                    f_val = func(x_clipped)
                    calls_made += 1
                    update_best(x_clipped, f_val)
                    return f_val
                
                try:
                    # Use tight tolerances for precision
                    res = minimize(
                        lbfgs_func, 
                        x0, 
                        method='L-BFGS-B', 
                        bounds=[(-5, 5)] * dim,
                        options={
                            'maxiter': max_iter_lbfgs,
                            'ftol': 1e-15, 
                            'gtol': 1e-10,
                            'maxls': 20
                        }
                    )
                    if res.x is not None and res.fun < np.inf:
                        # Update best if improved
                        update_best(res.x, res.fun)
                except Exception:
                    pass

        return best_x, best_f

# EVOLVE-BLOCK-END
