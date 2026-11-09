"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Start Adaptive (1+1) Evolution Strategy for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        calls_used = 0
        best_x = None
        best_f = np.inf

        def call_func(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= budget:
                return np.inf
            calls_used += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def run_es(x0, sigma0, max_iter, success_prob=0.375):
            """
            Adaptive (1+1)-ES with 1/5-th success rule.
            x0: initial point
            sigma0: initial step size
            max_iter: max iterations for this run
            """
            nonlocal calls_used, deadline, best_x, best_f
            
            x = x0.copy()
            sigma = sigma0
            f_x = call_func(x)
            
            # If the initial point is already the global best, update
            if f_x < best_f:
                best_f = f_x
                best_x = x.copy()

            for i in range(max_iter):
                if time.time() > deadline or calls_used >= budget:
                    break
                
                # Generate offspring
                z = np.random.randn(dim)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5, 5)
                
                f_trial = call_func(x_trial)
                
                if f_trial < f_x:
                    # Success: move to offspring
                    x = x_trial
                    f_x = f_trial
                    # Increase step size (less aggressive than 1/5 rule to avoid divergence)
                    sigma *= 1.05 
                else:
                    # Failure: keep current, decrease step size
                    sigma *= 0.95
                
                # Prevent sigma from becoming too small or too large
                sigma = np.clip(sigma, 1e-6, 10.0)

            return x, f_x

        # 1. Generate diverse initial points
        # The Rosenbrock function has a minimum at (1, 1, ..., 1).
        # The valley is hard to navigate. We should sample points that are likely 
        # to be in or near the valley, or at least provide a good gradient signal.
        
        # Strategy: 
        # - One point at the origin (0,0,0) - common bad start but standard.
        # - One point at (1,1,...,1) - the known global min location. Even with rotation/shift,
        #   this might be close or provide a strong signal if the shift is small.
        # - Random points in the box.
        # - Points along the diagonal.
        
        rng = np.random.default_rng(42)
        
        starts = []
        # Standard origin
        starts.append(np.zeros(dim))
        
        # Theoretical minimum location
        starts.append(np.ones(dim))
        
        # Random points
        for _ in range(5):
            starts.append(rng.uniform(-4, 4, dim))
            
        # Diagonal points (often helpful in Rosenbrock)
        for val in [-2, 0, 2]:
            starts.append(np.full(dim, val))

        # 2. Run ES from each start
        # We allocate budget dynamically. 
        # First, do a quick evaluation of all starts to find the best initial point.
        
        best_start_x = None
        best_start_f = np.inf
        
        for s in starts:
            if time.time() > deadline or calls_used >= budget:
                break
            s = np.clip(s, -5, 5)
            f_s = call_func(s)
            if f_s < best_start_f:
                best_start_f = f_s
                best_start_x = s.copy()
        
        # If we have budget left, run a longer ES from the best start
        # And potentially shorter ES from other promising starts if budget allows.
        
        # For simplicity and robustness, run a main ES from the best start.
        # Then, if budget remains, run secondary ES from other starts.
        
        if best_start_x is not None:
            # Main run: use remaining budget for iterations
            # Rough estimate: 1 call per iter. 
            # We've used len(starts) calls so far.
            remaining_calls = budget - calls_used
            # Leave a few calls for safety
            max_iter_main = int(remaining_calls * 0.8)
            if max_iter_main > 0:
                x_main, f_main = run_es(best_start_x, 0.5, max_iter_main)
                
                # If we have significant budget left, try another start
                remaining_calls = budget - calls_used
                if remaining_calls > 50:
                    # Pick the second best start
                    second_best_x = None
                    second_best_f = np.inf
                    for s in starts:
                        if np.linalg.norm(s - best_start_x) < 1e-6:
                            continue
                        # We already evaluated these, so we know their values? 
                        # No, call_func updates best_f but doesn't store per-start values easily in this scope.
                        # Let's just pick a random different start from the list.
                        pass
                    
                    # Simpler: just pick starts[0] if it wasn't the best, or starts[1]
                    alt_start = starts[0] if not np.allclose(starts[0], best_start_x) else starts[1]
                    alt_start = np.clip(alt_start, -5, 5)
                    # Re-evaluate to get current f (it might have been improved by previous ES? No, ES is local)
                    # Actually, we can just run ES from it.
                    max_iter_alt = int(remaining_calls * 0.5)
                    if max_iter_alt > 10:
                        run_es(alt_start, 0.5, max_iter_alt)

        return best_x, best_f
# EVOLVE-BLOCK-END
