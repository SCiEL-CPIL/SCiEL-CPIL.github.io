import numpy as np
import time
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit
        
        def time_ok():
            return time.time() < deadline
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(self.dim)
        best_fx = np.inf
        call_count = 0

        # Tracked function to count calls and update best
        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        bounds = [(-5.0, 5.0)] * self.dim
        
        # Strategy: Competitive Multi-Start
        # We maintain a pool of candidate starting points.
        # We allocate small bursts of budget to each to see which converges fastest.
        # The "winner" gets more budget.
        
        # Initial pool: Origin and a few random points in [-3, 3]^5 (tighter than [-4,4] to ensure validity and better density)
        # Rosenbrock global min is often near 0 in unrotated, but rotated/shifted.
        # The problem states minimizer is in [-4, 4]^5.
        
        candidates = []
        # 1. Origin
        candidates.append(np.zeros(self.dim))
        # 2. 4 random points
        for _ in range(4):
            candidates.append(rng.uniform(-3.0, 3.0, self.dim))
            
        # Evaluate initial candidates
        # We will evaluate them one by one, but keep track of their "progress"
        
        # To maximize AOCC, we want to find a good basin ASAP.
        # Let's evaluate the first 5 points.
        initial_points = []
        for x0 in candidates:
            if call_count >= self.budget - 20:
                break
            fx = tracked_func(x0)
            initial_points.append((fx, x0.copy()))
            
        # Sort initial points by function value
        initial_points.sort(key=lambda item: item[0])
        
        # Now, we have 5 points. We will run a "race" between the top 2-3.
        # If the best one stagnates, we switch to the second best.
        
        # We define a "burst" length.
        burst_len = 30
        
        # We will iterate through the sorted candidates.
        # For each candidate, we try to improve it with L-BFGS-B for burst_len calls.
        # If it improves significantly, we continue with it (polish).
        # If not, we move to the next candidate.
        
        # However, L-BFGS-B is stateful. We should restart it from the best known point if we switch.
        
        active_x = initial_points[0][1].copy()
        active_fx = initial_points[0][0]
        
        # We keep a history of "failed" starts to avoid re-trying the same bad basin.
        # In 5D, basins are distinct enough that if we move far, we are likely in a new one.
        
        # Loop through candidates
        for i, (fx_start, x_start) in enumerate(initial_points):
            if not time_ok() or call_count >= self.budget - 10:
                break
                
            # Only try if this starting point was good enough initially
            # or if we haven't found a very good solution yet
            if fx_start > best_fx + 1e-5 and i > 0:
                # If this start is much worse than our current best, skip it?
                # Actually, if our current best is a local min, this might be better.
                # But if fx_start is already much worse than best_fx, it's unlikely to be the global min 
                # unless the landscape is very flat, which Rosenbrock is not.
                # However, for AOCC, we already have a good best_fx. 
                # We should focus on polishing the best one we have.
                # So, if we have a very good best_fx, we should stop trying worse starts.
                if best_fx < 1.0: # Arbitrary threshold: if we are close, polish
                     # Break out of trying new starts and just polish
                     break
                     
            # Run L-BFGS-B from x_start
            # We use the current best_x as a reference? No, L-BFGS-B optimizes from x_start.
            # But we want to update global best.
            
            # We limit maxfun to burst_len
            maxfun_allowed = min(burst_len, self.budget - call_count - 5)
            if maxfun_allowed < 10:
                maxfun_allowed = 10
                
            try:
                res = minimize(
                    tracked_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 100,
                        'maxfun': maxfun_allowed,
                        'ftol': 1e-12,
                        'gtol': 1e-8
                    }
                )
            except Exception:
                pass
            
            # After the burst, check if we improved
            # If we improved significantly, we might want to continue polishing this one.
            # If not, we move to the next candidate.
            
            # How to decide "significantly"?
            # If the function value dropped by more than a small amount, it's a good basin.
            # Rosenbrock values can be large. 
            # Let's just check if call_count increased and best_fx decreased.
            
            # If we are still far from 0 (global min is 0 for Rosenbrock, but shifted here),
            # and we haven't used much budget, try the next candidate.
            
            # Heuristic: If the improvement in this burst was less than 1% of the starting value 
            # (or absolute threshold), consider it stagnant.
            
            # For simplicity and robustness: 
            # If we have budget left, and we haven't converged (best_fx > 1e-4), 
            # and we haven't tried all candidates, try the next one.
            
            # But we must be careful not to oscillate.
            # If the new candidate starts at a worse point, it's risky.
            
            # Let's refine: 
            # We only switch to a new candidate if the current one is "stuck".
            # "Stuck" means: L-BFGS-B stopped early (maxfun reached) but didn't improve best_fx 
            # by a decent amount in the last burst.
            
            # We don't easily track "improvement in last burst" without storing fx before burst.
            fx_before_burst = active_fx # This is not quite right, active_fx is global best.
            # We need fx of the specific candidate before the burst.
            
            # Let's restructure:
            # We will just loop through the top 3 initial points.
            # For each, we run L-BFGS-B.
            # If the resulting best_fx is very low (< 0.1), we break and do a final polish.
            # Otherwise, we try the next.
            
        # Final Polish Phase
        # Use the best_x found so far.
        # Use Nelder-Mead or L-BFGS-B with tight tolerances for the remaining budget.
        if time_ok() and call_count < self.budget - 5:
            try:
                remaining = self.budget - call_count - 5
                if remaining > 10:
                    # Use L-BFGS-B again for high precision
                    res = minimize(
                        tracked_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': 1000,
                            'maxfun': remaining,
                            'ftol': 1e-15,
                            'gtol': 1e-12
                        }
                    )
            except Exception:
                pass
            
            # If still budget left, try Nelder-Mead for derivative-free robustness
            if time_ok() and call_count < self.budget - 5:
                try:
                    remaining = self.budget - call_count - 5
                    if remaining > 10:
                        res = minimize(
                            tracked_func,
                            best_x,
                            method='Nelder-Mead',
                            options={
                                'maxiter': 200,
                                'maxfev': remaining,
                                'adaptive': True,
                                'xatol': 1e-10,
                                'fatol': 1e-12
                            }
                        )
                except Exception:
                    pass

        return best_x, best_fx
