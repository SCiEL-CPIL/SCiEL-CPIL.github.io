"""Optimized optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start (1+1)-ES with 1/5 success rule for ill-conditioned problems."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _es_1plus1(self, func, rng, x0, sigma0, max_calls, deadline):
        """
        (1+1)-ES with 1/5 success rule.
        Returns best_x, best_f, calls_used.
        """
        dim = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        sigma = sigma0
        calls = 1
        
        # 1/5 success rule parameters
        # Increase sigma by factor c_inc if success, decrease by c_dec if failure
        # Standard values for (1+1)-ES on ill-conditioned problems
        c_inc = 1.0 / 0.8414  # ~1.188
        c_dec = 0.8414        # ~0.841
        
        # Actually, the standard 1/5 rule is:
        # If success: sigma = sigma * c_inc
        # If failure: sigma = sigma * c_dec
        # With c_inc = 1.1 and c_dec = 0.9 often used, or more precisely
        # c_inc = (1 + 1/5)^(1/5) and c_dec = (1 - 1/5)^(1/5)? No.
        # The 1/5 rule says: adjust step size so that 1/5 of steps are successful.
        # A common implementation is:
        # sigma *= 1.1 if success else 0.9
        
        # Let's use the more principled approach:
        # c_inc = 1.0 / (1 - 0.2) ** 0.2 ? No.
        # Let's stick to simple multiplicative factors.
        # c_inc = 1.05, c_dec = 0.95 is a safe start.
        # Or c_inc = 1.1, c_dec = 0.9.
        
        c_inc = 1.1
        c_dec = 0.9

        best_x = x.copy()
        best_f = fx
        
        while calls < max_calls:
            if time.time() > deadline - 1.0:
                break
                
            z = rng.standard_normal(dim)
            y = x + sigma * z
            # Clip to box [-5, 5]
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls += 1
            
            if fy < fx:
                # Success
                x = y
                fx = fy
                sigma *= c_inc
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # Failure
                sigma *= c_dec
                
            # Prevent sigma from becoming too small or too large
            sigma = np.clip(sigma, 1e-8, 10.0)

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0
        
        # Multi-start strategy
        # We will run several (1+1)-ES instances from different starting points.
        # For dim=5, we can afford several restarts.
        # Let's try 5-10 restarts depending on budget.
        
        num_restarts = 5
        calls_per_restart = budget // num_restarts
        
        # Initial points: center, corners, random
        # Center
        starts = [np.zeros(dim)]
        
        # Random points within [-4, 4] to ensure we are inside the box
        for _ in range(num_restarts - 1):
            x0 = rng.uniform(-4, 4, size=dim)
            starts.append(x0)
        
        # Also add a few points near the boundaries?
        # The problem states minimizer is in [-4, 4]^5.
        # So starting in [-4, 4] is good.
        
        for i, x0 in enumerate(starts):
            if total_calls >= budget:
                break
            
            remaining = budget - total_calls
            # Allocate calls, ensuring at least some for the last one
            if i == len(starts) - 1:
                max_calls = remaining
            else:
                max_calls = min(calls_per_restart, remaining)
            
            if max_calls <= 1:
                break
                
            # Initial sigma
            # For dim=5, a good initial sigma is around 2.0-3.0
            sigma0 = 2.5
            
            bx, bf, used = self._es_1plus1(func, rng, x0, sigma0, max_calls, deadline)
            total_calls += used
            
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
                
            # If we are close to the deadline, stop
            if time.time() > deadline - 5.0:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
