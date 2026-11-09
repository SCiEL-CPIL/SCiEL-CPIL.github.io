import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 295  # Safety margin before 300s limit
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        eval_count = 0
        rng = np.random.default_rng(42)
        
        def try_eval(x):
            nonlocal best_f, best_x, eval_count
            if eval_count >= self.budget or time.time() > deadline - 2:
                return None
            try:
                fx = func(x)
            except Exception:
                return None
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial random sampling to get a reasonable starting point
        n_init = min(20, self.budget // 5)
        for i in range(n_init):
            x = rng.uniform(-5, 5, self.dim)
            try_eval(x)
        
        # Phase 2: (1+1)-ES with 1/5th success rule for adaptive step size
        remaining = self.budget - eval_count
        if remaining > 0 and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            
            # Start with a moderate sigma
            sigma = 0.5
            no_improve = 0
            limit = 15  # Reset sigma if no improvement for this many iterations
            success_count = 0
            window_size = 20  # Window for 1/5th success rule
            
            while remaining > 0 and time.time() < deadline - 5:
                # Check 1/5th success rule every window_size iterations
                if success_count >= window_size:
                    # Adjust sigma based on success rate
                    success_rate = success_count / (2 * window_size)  # Approximate
                    if success_rate > 1/5:
                        sigma *= 1.1
                    else:
                        sigma *= 0.9
                    success_count = 0
                
                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = try_eval(y)
                if fy is None:
                    break
                remaining -= 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    success_count += 1
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve > limit:
                        sigma = 0.1
                        no_improve = 0
                    if sigma < 1e-6:
                        break
                        
        # Phase 3: Final high-precision polish with smaller steps
        remaining = self.budget - eval_count
        if remaining > 0 and time.time() < deadline - 2:
            x = best_x.copy()
            fx = best_f
            sigma = 0.01
            rng_local = np.random.default_rng(123)
            
            for _ in range(min(remaining, 100)):
                if time.time() > deadline - 2:
                    break
                y = np.clip(x + sigma * rng_local.standard_normal(self.dim), -5, 5)
                fy = try_eval(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 0.95
                else:
                    sigma *= 0.95
                    if sigma < 1e-8:
                        break
                        
        return best_x, best_f
