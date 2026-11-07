import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin from 330s limit
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Generate diverse initial points
        n_initial = 25
        points = []
        values = []
        
        # Center
        x_center = np.zeros(self.dim)
        f_center = func(x_center)
        points.append(x_center)
        values.append(f_center)
        
        # Vertices (8 corners for 5D)
        for i in range(1, min(8, n_initial - 1)):
            x_vert = np.array([5.0 if (i >> j) & 1 else -5.0 for j in range(self.dim)])
            f_vert = func(x_vert)
            points.append(x_vert)
            values.append(f_vert)
        
        # Random points to fill up to n_initial
        for _ in range(len(points), n_initial):
            x_rand = rng.uniform(-5, 5, self.dim)
            f_rand = func(x_rand)
            points.append(x_rand)
            values.append(f_rand)
        
        # Find top 5 best points
        sorted_indices = np.argsort(values)
        top_indices = sorted_indices[:5]
        
        best_x = points[top_indices[0]].copy()
        best_f = values[top_indices[0]]
        
        calls_used = len(points)
        remaining_budget = self.budget - calls_used
        
        # Phase 2: Multi-start (1+1)-ES with adaptive step size
        es_budget_per_start = max(10, remaining_budget // 5)
        
        for idx in top_indices:
            if time.time() > deadline - 2:
                break
            
            x = points[idx].copy()
            fx = values[idx]
            sigma = 0.5  # Smaller initial step size
            
            successes = 0
            total_trials = 0
            es_calls = 0
            
            while es_calls < es_budget_per_start:
                if time.time() > deadline - 1:
                    break
                
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                es_calls += 1
                total_trials += 1
                
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                
                # Adapt step size every 10 trials
                if total_trials % 10 == 0:
                    success_rate = successes / 10.0
                    if success_rate > 0.3:
                        sigma *= 1.3
                    elif success_rate < 0.1:
                        sigma /= 1.3
                    successes = 0
                
                sigma = max(1e-8, min(sigma, 5.0))
        
        return best_x, best_f
