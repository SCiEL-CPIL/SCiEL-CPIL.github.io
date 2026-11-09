import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = time.time()
        self.time_limit = 320.0  # Safety margin

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()
        
        def eval_func(x):
            if self.calls >= self.budget:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            self.calls += 1
            return func(np.clip(x, -5.0, 5.0))

        n = self.dim
        
        # Multi-start Powell optimizer
        # Generate diverse starting points
        best_x = None
        best_fx = float('inf')
        
        # Starting points: origin, corners, and random points
        start_points = [np.zeros(n)]
        
        # Add some structured points
        for i in range(n):
            p1 = np.zeros(n)
            p1[i] = 2.0
            start_points.append(p1)
            
            p2 = np.zeros(n)
            p2[i] = -2.0
            start_points.append(p2)
        
        # Add random points
        np.random.seed(42)
        for _ in range(5):
            start_points.append(np.random.uniform(-4, 4, size=n))
        
        # Evaluate each starting point with Powell
        for i, x0 in enumerate(start_points):
            if self.calls >= self.budget:
                break
            if time.time() - self.start_time > self.time_limit:
                break
            
            try:
                from scipy.optimize import minimize
                
                # Use Powell's method for derivative-free optimization
                # Limit iterations to stay within budget
                max_iter = min(50, (self.budget - self.calls) // (n + 1))
                if max_iter < 1:
                    max_iter = 1
                    
                res = minimize(eval_func, x0, method='Powell', 
                              options={'maxiter': max_iter, 'ftol': 1e-12, 'xtol': 1e-12})
                
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
                    
            except StopIteration:
                break
            except Exception:
                continue
        
        # If we still have budget, do a final refinement from the best point
        remaining = self.budget - self.calls
        if remaining > 10 and best_x is not None and time.time() - self.start_time < self.time_limit:
            try:
                from scipy.optimize import minimize
                max_iter = min(20, remaining // (n + 1))
                if max_iter >= 1:
                    res = minimize(eval_func, best_x, method='Powell',
                                  options={'maxiter': max_iter, 'ftol': 1e-14, 'xtol': 1e-14})
                    if res.fun < best_fx:
                        best_fx = res.fun
                        best_x = res.x.copy()
            except Exception:
                pass
        
        if best_x is None:
            best_x = np.zeros(n)
        
        return best_x, best_fx
