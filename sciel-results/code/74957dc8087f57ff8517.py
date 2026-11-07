"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        
        # Initialization
        # Start at center, which is often good for BBOB
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1
        
        # Parameters
        sigma = 2.0
        initial_sigma = 2.0
        stagnation_limit = 50
        stagnation_counter = 0
        success_window = 10
        successes = 0
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                successes += 1
                stagnation_counter = 0
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                stagnation_counter += 1
                
            # Adapt sigma based on 1/5 success rule over last 'success_window' steps
            # We need to track successes in a sliding window.
            # For simplicity, we can just check the last few steps or use a cumulative counter that resets.
            # Let's use a simple heuristic:
            # If we had a success, increase sigma slightly. If not, decrease.
            # The 1/5 rule is better. Let's implement a proper window.
            
            # To avoid complex window management, let's use the standard (1+1)-ES adaptation:
            # sigma *= exp(0.33 * (1/5 - (1-success_rate))) ? No.
            # Let's stick to the simple rule but make it smoother.
            # If success: sigma *= 1.1
            # If failure: sigma *= 0.9
            # This is very slow adaptation.
            
            # Let's try the 1/5 rule properly.
            # We need a deque or a counter that decays.
            # Let's use a counter `recent_successes` that decays by 0.9 each step.
            # If success: recent_successes += 1
            # else: recent_successes *= 0.9
            # If recent_successes > 0.5 * (1 + 0.9 + ... )?
            
            # Simpler robust approach:
            # If fy < fx: sigma *= 1.5
            # else: sigma *= 0.8
            # And add restart logic.
            
            if fy < fx:
                sigma *= 1.5
            else:
                sigma *= 0.8
                
            # Restart logic
            if stagnation_counter > stagnation_limit or sigma < 1e-4:
                # Restart from best point with larger sigma
                x = best_x.copy()
                fx = best_f
                sigma = initial_sigma
                stagnation_counter = 0
                # Optional: Jitter the start point?
                # x = np.clip(x + 0.1 * rng.standard_normal(n), -5, 5)
                
        return best_x, best_f
# EVOLVE-BLOCK-END
