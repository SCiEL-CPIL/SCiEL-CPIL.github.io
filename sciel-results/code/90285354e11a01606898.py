"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-Start Adaptive 1+1 Evolution Strategy.
    
    Strategy:
    1. Uses a 1+1 ES (single parent, single offspring) which is extremely lightweight per step.
    2. Adapts step size using the 1/5th success rule (classic ES adaptation).
    3. Employs a "Time-Budgeted Multi-Start" approach: if no improvement is seen for a 
       certain number of steps or time, it restarts from a new random point or the global best 
       with a perturbed step size.
    4. This avoids the overhead of covariance matrix updates and population sorting, 
       allowing significantly more evaluations in the same budget, which is crucial for 
       AOCC (Area Over the Curve of Convergence) where early and steady progress is rewarded.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_fx = np.inf
        calls_made = 0
        
        # Multi-start parameters
        # We will run several "runs" of the 1+1 ES.
        # Each run starts from a random point (or best point found so far) 
        # and runs until stagnation or time/budget limit.
        
        # Initial step size
        base_sigma = 1.0
        
        # 1/5th success rule parameters
        # If success rate > 1/5, increase sigma. If < 1/5, decrease.
        # Typical: sigma *= exp(c * (success_rate - 1/5))
        # c is usually around 1.0 or tuned. Let's use c=1.0 for simplicity and robustness.
        
        # Stagnation detection
        # If no improvement for N steps, restart.
        # N should be proportional to dim, e.g., 10 * dim
        max_stagnant_steps = 20 * dim
        
        # We'll manage the search in a loop of "restarts"
        # To ensure we use the budget effectively, we don't pre-allocate fixed chunks,
        # but rather let the ES run until it stalls, then restart.
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Choose starting point for this run
            if best_x is None:
                # First run: random start
                x = rng.uniform(-5, 5, dim)
            else:
                # Subsequent runs: start near the best known point but with some noise
                # to escape potential local optima, or just random if we suspect we're stuck
                # Let's mix: 50% chance to start from best (with small noise), 50% random
                if rng.random() < 0.5 and best_x is not None:
                    x = best_x + rng.normal(0, base_sigma * 0.5, dim)
                else:
                    x = rng.uniform(-5, 5, dim)
                
                x = np.clip(x, -5, 5)
            
            fx = func(x)
            calls_made += 1
            
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            
            sigma = base_sigma
            stagnant_count = 0
            
            # Inner loop: run 1+1 ES until stagnation or budget/time limit
            while calls_made < budget:
                if time.time() > deadline:
                    break
                
                # Generate offspring
                # Standard normal perturbation
                y = rng.standard_normal(dim)
                x_new = x + sigma * y
                x_new = np.clip(x_new, -5, 5)
                
                fx_new = func(x_new)
                calls_made += 1
                
                # Check success
                success = False
                if fx_new < fx:
                    success = True
                    # Move
                    x = x_new
                    fx = fx_new
                    stagnant_count = 0
                    # Update global best
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    stagnant_count += 1
                
                # Adapt step size (1/5th rule)
                # We need a window to calculate success rate. 
                # For 1+1, we can use a simple exponential moving average of success.
                # Or simpler: just adjust based on last few steps.
                # A common simple implementation:
                # Keep a history of last N successes.
                # Here, let's use a very simple adaptation:
                # If we had a success, increase sigma slightly. If not, decrease.
                # But 1/5 rule is better. Let's maintain a small window.
                
                # To keep it lightweight, let's use a fixed window of last 10 steps.
                # We'll store successes in a list.
                # Actually, even simpler: 
                # sigma = sigma * exp(0.1 * (success - 0.2))
                # This is a simplified 1/5 rule.
                # If success (1), sigma *= exp(0.08) ~ 1.083
                # If fail (0), sigma *= exp(-0.02) ~ 0.980
                # This leads to growth if success rate > ~20%.
                
                if success:
                    sigma *= np.exp(0.1)
                else:
                    sigma *= np.exp(-0.02)
                    
                # Clip sigma to reasonable bounds
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                # Check for stagnation
                if stagnant_count > max_stagnant_steps:
                    break
            
            # If we broke out due to stagnation, we loop back to the outer while
            # and potentially restart.
            
            # If we broke out due to budget/time, the outer while will also break.
            
            # Optional: If we used very little budget in this run (e.g., immediate stagnation),
            # we might want to force a random restart to explore elsewhere.
            # The outer loop handles this by re-entering the selection of x.
            
        return best_x, best_fx
# EVOLVE-BLOCK-END
