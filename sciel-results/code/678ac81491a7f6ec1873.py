import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin
        
        calls = [0]
        best_x = None
        best_f = float('inf')

        def wrapped_func(x):
            nonlocal best_x, best_f
            if time.time() > deadline:
                # Return a large value to signal stop? 
                # Or raise an exception?
                # Minimize might catch exceptions.
                # Better to just return a very large number to discourage further search in that direction?
                # Actually, if we exceed time, we should stop the whole process.
                # We can't easily break out of scipy minimize.
                # We will rely on the budget and the fact that 500 calls is fast.
                # 500 calls in 5D is very fast. Time limit is unlikely to be hit unless func is slow.
                # But the prompt says "evaluator KILLS... after 330s".
                # So we must be careful.
                
                if calls[0] >= self.budget:
                    return float('inf')
            
            calls[0] += 1
            val = func(x)
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
                
            return val

        # Phase 1: Latin Hypercube Sampling
        # 20 points
        n_lhs = 20
        if self.budget > n_lhs:
            # Generate LHS
            # For each dimension, divide [-5, 5] into n_lhs bins
            # Pick one random point in each bin
            lhs_points = []
            for i in range(n_lhs):
                point = []
                for d in range(self.dim):
                    # Bin [i/n_lhs, (i+1)/n_lhs] mapped to [-5, 5]
                    low = -5 + (10 * i / n_lhs)
                    high = -5 + (10 * (i + 1) / n_lhs)
                    point.append(np.random.uniform(low, high))
                lhs_points.append(point)
            
            lhs_points = np.array(lhs_points)
            
            # Evaluate all
            for x in lhs_points:
                if calls[0] >= self.budget or time.time() > deadline:
                    break
                wrapped_func(x)
        
        # If we used all budget in LHS, return best
        if calls[0] >= self.budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Powell's Method
        # Start from best_x
        # x0 is best_x
        # Bounds are [-5, 5]
        
        # Powell options
        # maxiter: max number of iterations. 
        # Each iteration involves several function calls.
        # We have ~480 calls left.
        # Powell in 5D might use ~5-10 calls per iteration?
        # So ~50-100 iterations.
        
        try:
            res = minimize(wrapped_func, best_x, method='Powell', 
                           bounds=[(-5, 5)] * self.dim,
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-12})
            
            # Update best if Powell found something better (though wrapped_func tracks it)
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
                
        except Exception as e:
            # If minimize crashes, we keep the best found so far
            pass

        return best_x, best_f
