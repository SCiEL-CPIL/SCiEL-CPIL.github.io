import numpy as np
import time
import sys

try:
    from scipy.stats import qmc
    HAS_QMC = True
except ImportError:
    HAS_QMC = False

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0 # Safety margin for 300s limit
        d = self.dim
        budget = self.budget
        
        # 1. Multi-start Initialization
        # We will spend a portion of the budget on diverse initial points
        # to quickly locate the basin of attraction.
        n_init = 50
        if budget <= n_init:
            n_init = max(1, budget // 2)
            
        if HAS_QMC:
            sampler = qmc.Sobol(d, scramble=True)
            points = sampler.random(n_init)
        else:
            # Fallback to random if scipy is missing (unlikely per prompt)
            points = np.random.rand(n_init, d)
            
        # Map [0, 1]^d to [-5, 5]^d
        points = points * 10.0 - 5.0
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Evaluate initial points
        for i in range(n_init):
            if time.time() - start_time > timeout:
                break
            x = points[i]
            try:
                fx = func(x)
            except Exception:
                fx = np.inf
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if best_x is None:
            # Fallback if no successful call
            best_x = np.zeros(d)
            best_f = func(best_x)
            calls_made += 1

        # 2. (1+1)-ES Optimization
        sigma = 1.0
        success_count = 0
        window_size = 15
        successes_in_window = 0
        
        # Simple 1/5 success rule adaptation
        # We check every 'window_size' steps
        
        while calls_made < budget:
            if time.time() - start_time > timeout:
                break
                
            # Generate mutation
            y = best_x + sigma * np.random.standard_normal(d)
            y = np.clip(y, -5.0, 5.0)
            
            try:
                fy = func(y)
            except Exception:
                fy = np.inf
            
            calls_made += 1
            
            if fy < best_f:
                best_x = y
                best_f = fy
                successes_in_window += 1
                # Increase sigma slightly on success
                sigma *= 1.1
            else:
                # Decrease sigma slightly on failure
                sigma *= 0.9
            
            # Adapt sigma periodically
            # Simple adaptation: if success rate is too high/low
            # But simple multiplicative update is often enough for 5D with 450 steps.
            # Let's add a basic check to prevent sigma from becoming too small or large.
            if sigma < 1e-4:
                sigma = 1e-4
            if sigma > 10.0:
                sigma = 10.0

        return best_x, best_f
