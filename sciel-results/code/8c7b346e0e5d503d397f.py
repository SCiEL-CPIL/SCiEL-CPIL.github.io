import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        # (1+1)-ES Parameters
        # Initial step size
        sigma = 2.0
        # 1/5th success rule parameters
        c_sigma = 1.0 / (dim + 1)
        d_sigma = 1.0 / (dim + 1)
        
        # State
        x = np.zeros(dim)
        f_x = func(x)
        
        # Track best
        best_x = x.copy()
        best_f = f_x
        calls_made = 1
        
        # Counters for 1/5th rule
        successes = 0
        total_steps = 0
        
        # Nelder-Mead parameters
        # We will use a simple local polish every few ES steps or when progress stalls
        nm_interval = 20  # Polish every 20 ES steps
        nm_maxiter = 200
        nm_calls = 0
        
        # Pre-compute constants
        # For (1+1) ES, we just need a random direction
        rng = np.random.default_rng(42)
        
        # To make the 1/5th rule effective, we track success over a window
        # But standard 1/5th rule is: if success rate > 1/5, increase sigma, else decrease.
        # A simpler, robust version for short runs:
        # sigma *= exp(c_sigma * (success_rate - 1/5))
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # 1. Generate candidate
            direction = rng.standard_normal(dim)
            # Normalize to unit length to keep sigma meaningful as step size
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-10:
                direction /= norm_dir
            
            x_trial = x + sigma * direction
            # Clip to box
            x_trial = np.clip(x_trial, -5, 5)
            
            # Evaluate
            f_trial = func(x_trial)
            calls_made += 1
            
            # 2. Update (1+1) ES
            if f_trial < f_x:
                x = x_trial
                f_x = f_trial
                successes += 1
                total_steps += 1
                
                if f_x < best_f:
                    best_f = f_x
                    best_x = x.copy()
            else:
                total_steps += 1
            
            # Adapt sigma
            if total_steps > 0:
                success_rate = successes / total_steps
                # 1/5th success rule
                # If success rate > 0.2, increase sigma, else decrease
                # Using exponential update
                sigma *= np.exp(c_sigma * (success_rate - 0.2))
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                # Reset counters every few steps to avoid long-term bias
                # Or just keep them running. Let's keep them running for stability.
            
            # 3. Local Polish with Nelder-Mead
            # Trigger periodically or if we are stuck
            if calls_made % nm_interval == 0 and calls_made < budget - 10:
                # Use the current best point as starting point for NM
                # NM is deterministic and fast for low dim
                # We need to be careful with the budget
                # Wrap func to count calls
                nm_calls_count = [0]
                def nm_func(x_nm):
                    nm_calls_count[0] += 1
                    return func(np.clip(x_nm, -5, 5))
                
                # Initial simplex: current point + small perturbations
                # Nelder-Mead generates its own simplex, but we can provide options
                # To ensure it stays in bounds, we rely on clipping inside nm_func
                # and the fact that NM usually stays close.
                
                # Check if we have enough budget for a reasonable NM run
                # NM in dim=5 might take ~50-100 calls to converge
                if budget - calls_made > 50:
                    try:
                        res = minimize(nm_func, x, method='Nelder-Mead', 
                                       options={'maxiter': nm_maxiter, 'xatol': 1e-6, 'fatol': 1e-8})
                        
                        calls_made += nm_calls_count[0]
                        
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x.copy()
                            # Update ES state to the new best point to continue from there
                            x = res.x.copy()
                            f_x = res.fun
                            # Reset sigma to a smaller value since we are in a local basin
                            # sigma = min(sigma, 0.5)
                            
                    except Exception:
                        pass
            
            # If we are very close to budget, stop NM and just do ES
            if calls_made >= budget:
                break
                
        return best_x, best_f
