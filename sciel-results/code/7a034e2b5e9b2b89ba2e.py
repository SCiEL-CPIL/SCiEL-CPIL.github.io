"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid Global Scan + Local Nelder-Mead Refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # Safety margin for 300s limit
        
        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        call_lock = [0]  # Using list for mutable int in closure if needed, but nonlocal is fine

        def eval_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            # Ensure x is a valid numpy array of correct shape
            x = np.asarray(x, dtype=np.float64).flatten()
            if len(x) != dim:
                return np.inf
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def safe_eval(x):
            """Wrapper for scipy minimize that counts calls and respects budget."""
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            return eval_func(x)

        # Phase 1: Structured Initial Scan for Early Progress
        # 1. Origin
        eval_func(np.zeros(dim))
        
        # 2. Cardinal directions at small radii
        for radius in [0.5, 1.0, 2.0]:
            for i in range(dim):
                x_pos = np.zeros(dim)
                x_pos[i] = radius
                eval_func(x_pos)
                
                x_neg = np.zeros(dim)
                x_neg[i] = -radius
                eval_func(x_neg)
        
        # 3. Random Global Scan
        # Allocate ~80 calls for random global exploration
        global_scan_budget = 80
        for _ in range(global_scan_budget):
            if calls_made >= budget or time.time() > deadline:
                break
            # Sample in [-4, 4] to stay away from boundaries where function might be worse
            x = rng.uniform(-4.0, 4.0, dim)
            eval_func(x)

        # Phase 2: Local Refinement using Nelder-Mead
        # We will try to refine the best few points found so far.
        # Keep track of the best points found
        # Since we only track the single best, we need to store top candidates.
        # Let's modify eval_func to store top K? Or just refine the current best?
        # Refining only the best is risky if it's a local minimum.
        # Let's store the top 5 best points from Phase 1.
        
        # Re-implementing storage of top points for Phase 1 results is complex.
        # Instead, let's just run Nelder-Mead on the current best_x.
        # If it improves, great. If not, we might be stuck.
        # To mitigate, we can do a few random restarts + local refinement.

        # Let's run a loop of: Pick a candidate (best or random), run local opt.
        candidates = []
        if best_x is not None:
            candidates.append(best_x.copy())
        
        # Add a few random candidates for diversification
        for _ in range(3):
            if calls_made < budget:
                candidates.append(rng.uniform(-4.0, 4.0, dim))
        
        for start_x in candidates:
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Define objective for scipy
            # We need to ensure scipy doesn't exceed our budget
            # We will use maxfev to limit it
            remaining = budget - calls_made
            if remaining < 10:
                break
            
            # Limit local search to at most 50% of remaining budget or 100 calls, whichever is smaller
            max_local_calls = min(100, remaining)
            
            try:
                res = minimize(
                    safe_eval, 
                    start_x, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 50,
                        'maxfev': max_local_calls,
                        'xatol': 1e-8,
                        'fatol': 1e-8
                    }
                )
                
                # Update best if scipy found something better
                # Note: safe_eval already updates best_x/best_f if it finds better
                # But we should check res.fun just in case
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                # If minimize fails, continue
                pass

        # Phase 3: Polish with small random steps if budget remains
        if best_x is not None and calls_made < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            remaining = budget - calls_made
            
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = eval_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
