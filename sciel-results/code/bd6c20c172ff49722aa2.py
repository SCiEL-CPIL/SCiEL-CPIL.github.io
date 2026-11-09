"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Multi-Start Nelder-Mead Optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def __call__(self, func):
        self.start_time = time.time()
        # 300s limit, use 290s for safety margin
        self.deadline = self.start_time + 290.0
        
        rng = np.random.default_rng(42)
        
        # We will perform multiple restarts. 
        # Each restart uses a Nelder-Mead local search.
        # Nelder-Mead in dim=5 typically takes O(100-200) function evaluations to converge well.
        # With budget 500, we can afford roughly 2-4 restarts.
        
        # Strategy: 
        # 1. Start at a few random points.
        # 2. Run Nelder-Mead from each.
        # 3. Keep track of global best.
        
        # To maximize AOCC, we want fast initial progress.
        # Nelder-Mead is very fast in early iterations for smooth functions like Rosenbrock.
        
        # Let's use a fixed number of restarts, but monitor time.
        # If we run out of budget or time, we stop.
        
        num_restarts = 3
        # Allocate budget roughly evenly, but allow the first restart to take more if it converges fast?
        # Actually, Nelder-Mead stops when tol is reached. 
        # We can just let it run until budget is exhausted or tol is met.
        # But we need to count calls globally.
        
        # Let's define a wrapper that counts calls and checks deadline.
        # However, scipy.optimize.minimize doesn't support a callback that stops it easily 
        # without returning a result. 
        # We can pass a `callback` that raises an exception if we hit the limit, 
        # or just check inside the loop if we implement it manually.
        # Using scipy.optimize.minimize is efficient.
        
        # Alternative: Implement a simple Nelder-Mead ourselves to have full control over stopping criteria.
        # Given the constraint of 500 calls total, and the need for speed, 
        # a manual implementation or careful use of scipy with maxiter is good.
        
        # Let's use scipy.optimize.minimize with method='Nelder-Mead'.
        # We need to be careful with the number of function evaluations.
        # maxiter in Nelder-Mead is not exactly function evals, it's iterations.
        # In dim=5, 1 iteration ~ 6 function evals.
        # So 50 iterations ~ 300 evals.
        
        # We will run 3 restarts.
        # Restart 1: 200 budget
        # Restart 2: 150 budget
        # Restart 3: 100 budget
        # Remaining: 50 for a final polish or extra restart?
        
        # Actually, let's just run a loop of restarts.
        # Each restart gets a chunk of the remaining budget.
        
        remaining_budget = self.budget
        
        for r in range(num_restarts):
            if self.calls_made >= self.budget:
                break
            if time.time() > self.deadline:
                break
                
            # Initialize a new simplex around the current best or a random point?
            # If we have a good best_x, we can start near it. 
            # But if the landscape is rotated, starting far away might be better to escape local minima?
            # Rosenbrock has a narrow valley. 
            # Multi-start helps find the valley.
            
            if self.best_x is None:
                x0 = rng.uniform(-4, 4, self.dim)
            else:
                # Start near the best found so far, with some perturbation
                # Or start random? 
                # For Rosenbrock, if we are in the valley, Nelder-Mead converges fast.
                # If we are not, it might get stuck.
                # Let's mix: 50% near best, 50% random.
                if r == 0:
                    x0 = rng.uniform(-4, 4, self.dim)
                else:
                    # Perturb the best point
                    x0 = self.best_x + rng.normal(0, 1.0, self.dim)
                    x0 = np.clip(x0, -5, 5)
            
            # Define a wrapper for the function to count calls and handle clipping
            def wrapped_func(x):
                self.calls_made += 1
                if self.calls_made > self.budget:
                    # This should ideally not happen if we manage budgets, 
                    # but scipy might make one extra call.
                    # Returning a large value will discourage the optimizer.
                    return 1e10
                if time.time() > self.deadline:
                    return 1e10
                
                # Clip x to box
                x_clipped = np.clip(x, -5, 5)
                val = func(x_clipped)
                
                if val < self.best_f:
                    self.best_f = val
                    self.best_x = x_clipped.copy()
                    
                return val
            
            # Run Nelder-Mead
            # maxiter: limit iterations to conserve budget
            # In dim 5, 1 iter ~ 6 calls. 
            # Let's allocate a fraction of the remaining budget.
            # Say, we allow up to 1/3 of remaining budget per restart.
            max_iter_allowed = max(10, (self.budget - self.calls_made) // 3 // 6)
            
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead',
                    options={'maxiter': max_iter_allowed, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                
                # Update best if scipy found something better (it calls wrapped_func so it should be updated)
                # But just in case:
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x
            except Exception:
                # If an exception occurs, we keep the calls made so far
                pass

        # If we have budget left and time, we can do a final fine-tune or more restarts?
        # The loop above handles up to 3 restarts.
        # If we still have budget, we could do a 4th restart with the remaining budget.
        
        if self.calls_made < self.budget and time.time() < self.deadline:
            # One more restart with remaining budget
            x0 = self.best_x + rng.normal(0, 0.5, self.dim)
            x0 = np.clip(x0, -5, 5)
            
            def wrapped_func2(x):
                self.calls_made += 1
                if self.calls_made > self.budget:
                    return 1e10
                if time.time() > self.deadline:
                    return 1e10
                x_clipped = np.clip(x, -5, 5)
                val = func(x_clipped)
                if val < self.best_f:
                    self.best_f = val
                    self.best_x = x_clipped.copy()
                return val
            
            max_iter_allowed = max(5, (self.budget - self.calls_made) // 6)
            try:
                res = scipy.optimize.minimize(
                    wrapped_func2, 
                    x0, 
                    method='Nelder-Mead',
                    options={'maxiter': max_iter_allowed, 'xatol': 1e-9, 'fatol': 1e-9}
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x
            except Exception:
                pass

        return self.best_x, self.best_f

# EVOLVE-BLOCK-END
