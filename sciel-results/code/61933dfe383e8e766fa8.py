import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        # Pre-compute diverse starting points using Halton sequence
        # We want points in [-5, 5]^dim
        # Let's generate 10 diverse points and use 5 of them for 5 restarts
        # Or just generate them on the fly for each restart if we decide on dynamic restarts.
        # Fixed restarts are safer for AOCC as they guarantee exploration.
        
        num_restarts = 5
        budget_per_restart = self.budget // num_restarts
        
        # Halton sequence generator
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim, offset=1):
            points = []
            for i in range(n):
                point = []
                for d in range(1, dim + 1):
                    # Prime bases
                    base = 2 if d == 1 else 3 if d == 2 else 5 if d == 3 else 7 if d == 4 else 11
                    point.append(halton(i + offset, base))
                points.append(point)
            return np.array(points)

        # Generate 5 diverse points in [0, 1]^5
        halton_points = halton_sequence(num_restarts, self.dim)
        
        # Map to [-5, 5]^5
        start_points = halton_points * 10.0 - 5.0
        
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
                
            x = start_points[r].copy()
            fx = func(x)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            sigma = 1.0
            no_improve_count = 0
            max_no_improve = 20 # Restart if stuck for 20 steps
            
            # Use remaining budget for this restart
            remaining_budget = self.budget - (r * budget_per_restart)
            # But we must not exceed total budget
            # Actually, we allocated budget_per_restart, so let's just run that many steps
            # unless we hit the deadline or stagnation
            
            steps_in_this_restart = 0
            for i in range(budget_per_restart):
                if time.time() > deadline:
                    break
                if steps_in_this_restart >= budget_per_restart:
                    break
                    
                # Generate mutation
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    no_improve_count = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    no_improve_count += 1
                    
                # Clip sigma
                sigma = np.clip(sigma, 1e-4, 10.0)
                
                if no_improve_count > max_no_improve:
                    break
            # End restart loop
            
        return best_x, best_f
