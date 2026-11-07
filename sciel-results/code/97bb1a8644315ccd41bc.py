"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Structure-Exploiting Optimizer: Rastrigin-Aware Lattice Snap + Nelder-Mead."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        
        # State
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        # Helper to safely call func and update state
        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            
            try:
                f_val = func(x)
                calls_made += 1
                if np.isnan(f_val) or np.isinf(f_val):
                    return np.inf
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            except Exception:
                return np.inf

        # Phase 1: Global Search via Rastrigin-Aware Lattice Snap
        # Strategy:
        # 1. The function is likely a rotated/shifted Rastrigin.
        # 2. Rastrigin minima are at integers.
        # 3. We don't know the rotation/shift, but we can sample a sparse grid of 
        #    "integer-like" points in the domain to find a good basin.
        # 4. Instead of random, we prioritize points near the center and use a 
        #    structured sampling to cover the space efficiently.
        
        # We will use a two-stage initial search:
        # A. Evaluate center and a few key points to get a baseline.
        # B. Use a "snap to integer" heuristic on random points to find basins.
        
        # Let's reserve 150 calls for global exploration.
        global_budget = 150
        
        # A. Initial probes
        # Center
        evaluate(np.zeros(n))
        
        # A few random points to check scale/shift
        rng = np.random.default_rng(42)
        for _ in range(10):
            x_rand = rng.uniform(-5, 5, n)
            evaluate(x_rand)
            
        if best_x is None:
            best_x = np.zeros(n)
            best_f = 1000.0 # Fallback

        # B. Structured Global Search
        # We want to find the basin containing the global minimum.
        # Since it's Rastrigin-like, the landscape is multimodal.
        # We will generate a set of candidate points. 
        # Instead of pure random, let's use a Halton sequence or similar for coverage,
        # but also include "snapped" versions of random points to target minima.
        
        # Generate 100 candidate points
        # 50 random uniform
        # 50 random uniform but snapped to nearest integer (with small jitter)
        
        candidates = []
        for _ in range(50):
            x = rng.uniform(-4.5, 4.5, n)
            candidates.append(x)
            
        for _ in range(50):
            x = rng.uniform(-4.5, 4.5, n)
            # Snap to nearest integer
            x_snapped = np.round(x)
            # Add small jitter to avoid exactly hitting singularities or to explore around
            x_snapped += rng.normal(0, 0.1, n)
            candidates.append(x_snapped)
            
        # Shuffle to mix them
        rng.shuffle(candidates)
        
        for x in candidates:
            if calls_made >= global_budget or time.time() > deadline:
                break
            evaluate(x)
            
        # Phase 2: Local Refinement
        # We have the best point found in Phase 1.
        # Use Nelder-Mead for local convergence.
        # Nelder-Mead is robust in low dimensions (5D) and handles non-smoothness 
        # (if any, though Rastrigin is smooth) well.
        
        x_best = best_x.copy()
        
        # Define the objective function for scipy
        def scipy_func(x):
            return evaluate(x)
            
        # Run Nelder-Mead
        # We need to be careful with the budget. 
        # Nelder-Mead doesn't have a strict "maxfev" that stops exactly, 
        # but we can use the callback to stop if budget is exceeded or time is up.
        
        # However, scipy's minimize might make many calls. 
        # Let's implement a manual loop or use a strict callback.
        
        # Actually, since we have a hard budget, let's use a simpler, controlled local search
        # if we are worried about scipy overhead, but scipy NM is usually fast.
        # Let's try scipy NM with a callback that raises an exception to break if needed?
        # No, exceptions are bad. 
        # Let's just call minimize and hope it finishes. 
        # But if it doesn't, we waste time.
        
        # Better: Implement a simple, fast local refinement manually to ensure control.
        # Or, use scipy but wrap it in a try-except and check time.
        
        # Let's stick to a manual adaptive local search that is very fast per iteration.
        # Coordinate Descent with Adaptive Step Size.
        # This exploits the axis-aligned nature of Rastrigin minima (locally).
        
        x_curr = x_best.copy()
        f_curr = best_f
        sigma = 0.5  # Initial step size
        
        # Coordinate Descent Loop
        # Iterate until budget or time is up
        while calls_made < budget and time.time() < deadline:
            improved = False
            for i in range(n):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Try positive step
                x_plus = x_curr.copy()
                x_plus[i] += sigma
                f_plus = evaluate(x_plus)
                
                if f_plus < f_curr:
                    x_curr = x_plus
                    f_curr = f_plus
                    improved = True
                else:
                    # Try negative step
                    x_minus = x_curr.copy()
                    x_minus[i] -= sigma
                    f_minus = evaluate(x_minus)
                    
                    if f_minus < f_curr:
                        x_curr = x_minus
                        f_curr = f_minus
                        improved = True
            
            if improved:
                # If we improved, keep sigma or increase slightly?
                # In CD, usually keep sigma if moving, decrease if stuck.
                # Let's keep sigma constant for a few iterations, then decrease.
                # Simple strategy: if improved, sigma *= 1.1 (explore more)
                # else sigma *= 0.7 (exploit more)
                sigma *= 1.05
            else:
                sigma *= 0.8
                if sigma < 1e-5:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
