import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        eval_count = 0
        
        def check_deadline():
            return time.time() < deadline - 2.0

        # --- Phase 1: Initial Sampling to identify promising basins ---
        n_init = min(30, max(10, int(self.budget * 0.1))) 
        rng = np.random.default_rng(12345)
        
        for i in range(n_init):
            if not check_deadline() or eval_count >= self.budget:
                break
            x = rng.uniform(-5.0, 5.0, self.dim)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if eval_count >= self.budget or not check_deadline():
            return best_x, best_f

        # --- Phase 2: Surrogate-Assisted Search (Bayesian Optimization) ---
        # Use a simple GP-like surrogate to balance exploration and exploitation.
        # We maintain a set of points and their values.
        # At each step, we select the next point based on an acquisition function (e.g., Expected Improvement).
        # For simplicity and robustness, we use a random forest or a simple kernel regression if available,
        # but since we are restricted to stdlib/numpy/scipy, we will implement a simple GP approximation
        # or use scipy's interpolation if GP is too slow/complex.
        
        # Given the constraints and the need for robustness, a simpler approach is often better:
        # Use a small population-based method like CMA-ES if available, otherwise a well-tuned (1+1)-ES
        # with adaptive step size and restarts.
        
        # Let's stick to a robust (1+1)-ES with adaptive sigma and restarts, as it's reliable
        # and doesn't require complex surrogate modeling which might be overkill for 500 evals in 5D.
        
        remaining = self.budget - eval_count
        if remaining > 0 and check_deadline():
            
            # Adaptive (1+1)-ES
            x_cur = best_x.copy()
            f_cur = best_f
            
            sigma = 0.5 # Start with a larger sigma to explore
            rng_polish = np.random.default_rng(54321)
            no_improve_count = 0
            stall_limit = 20 # If 20 steps no improvement, consider it stuck
            
            while remaining > 0 and check_deadline():
                # Generate candidate
                y = np.clip(x_cur + sigma * rng_polish.standard_normal(self.dim), -5.0, 5.0)
                fy = func(y)
                eval_count += 1
                remaining -= 1
                
                if fy < f_cur:
                    # Improvement
                    x_cur = y
                    f_cur = fy
                    if f_cur < best_f:
                        best_f = f_cur
                        best_x = x_cur.copy()
                    no_improve_count = 0
                    # Increase sigma slightly if we are making good progress
                    sigma *= 1.05
                else:
                    no_improve_count += 1
                    # Decrease sigma
                    sigma *= 0.95
                    
                    # If stuck for too long, try a "jump" to escape local optimum
                    if no_improve_count >= stall_limit:
                        # Reset sigma to a moderate value
                        sigma = 0.5
                        no_improve_count = 0
                        
                        # If we are very stuck, try a few independent local searches from random points if budget allows.
                        if remaining > 50:
                            # Spend some budget on random restarts
                            n_restarts = min(10, remaining // 5)
                            for _ in range(n_restarts):
                                if remaining <= 0 or not check_deadline():
                                    break
                                x_r = rng_polish.uniform(-5.0, 5.0, self.dim)
                                f_r = func(x_r)
                                eval_count += 1
                                remaining -= 1
                                if f_r < f_cur:
                                    x_cur = x_r
                                    f_cur = f_r
                                    if f_cur < best_f:
                                        best_f = f_cur
                                        best_x = x_cur.copy()
                                    sigma = 0.2
                                    no_improve_count = 0
                                
                # Safety: if sigma becomes too small, we are likely at a local min
                if sigma < 1e-4:
                    break

        return best_x, best_f
