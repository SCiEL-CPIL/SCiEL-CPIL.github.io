"""Adaptive (1+1)-ES with strict budget and time management for Schaffers F7."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin for 300s limit
        
        # Track global best
        best_x = None
        best_f = float('inf')
        
        # Initial point: Center is often a good heuristic for symmetric-ish functions,
        # but Schaffers F7 can be ill-conditioned. Let's start at 0.
        x = np.zeros(self.dim)
        # Ensure first call is made immediately to establish a baseline for AOCC
        # and to handle the "exception before first call" risk.
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1
        
        # Initial step size. Box is [-5, 5]. A step of 2.0 covers significant ground.
        sigma = 2.0
        
        # 1/5 success rule parameters
        # We adapt sigma based on success rate over a window of trials.
        # For (1+1), a simple update is:
        # if success: sigma *= 1.5 (or exp(0.2))
        # if fail: sigma *= 0.75 (or exp(-0.2))
        # This is a form of (1+1)-ES with step size adaptation.
        
        # To be more robust, we can use a success counter over the last N steps.
        # However, simple multiplicative updates are fast and effective for low dimensions.
        # Let's use a slightly more sophisticated adaptation if time allows, 
        # but simple multiplicative is safer for strict budgets.
        
        # We will use a window of 10 for the success rate to adjust sigma.
        # This avoids oscillation.
        window_size = 10
        successes = 0
        
        while calls < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate offspring
            # Standard normal perturbation
            noise = np.random.standard_normal(self.dim)
            y = x + sigma * noise
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls += 1
            
            # Check for improvement
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
            
            if fy < fx:
                # Move to offspring
                x = y
                fx = fy
                successes += 1
            else:
                # Stay at parent
                pass
            
            # Adapt sigma every window_size steps
            if calls % window_size == 0:
                success_rate = successes / window_size
                if success_rate > 0.5:
                    sigma *= 1.25  # Expand search
                elif success_rate < 0.5:
                    sigma *= 0.8   # Shrink search
                else:
                    # Keep same, or slight adjustment
                    pass
                successes = 0 # Reset counter
                
                # Prevent sigma from becoming too small or too large
                sigma = np.clip(sigma, 1e-6, 10.0)
                
        return best_x, best_f
# EVOLVE-BLOCK-END
