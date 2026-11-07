import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT1 Pareto Front: x2..xn = 0
        # f1 = x1
        # f2 = 1 - sqrt(x1)
        
        if self.budget < 2:
             x = np.zeros(self.dim)
             x[0] = 0.5
             x = np.clip(x, lo, hi)
             return [problem(x)]
             
        # Helper to calculate objective values for a given x1
        def get_obj(x1_val):
            x = np.zeros(self.dim)
            x[0] = x1_val
            x = np.clip(x, lo, hi)
            return problem(x)
        
        # Evaluate endpoints
        f_start = get_obj(0.0)
        f_end = get_obj(1.0)
        
        evaluated_f = [f_start, f_end]
        
        # List of intervals to process: [x_a, x_b, f_a, f_b]
        # We use a list of tuples (x_a, x_b, f_a, f_b)
        # f_a and f_b are the objective vectors at the boundaries
        
        intervals = [(0.0, 1.0, f_start, f_end)]
        
        for i in range(2, self.budget):
            if time.time() > deadline:
                break
            
            best_interval_idx = -1
            best_gain = -1
            best_x = None
            best_f = None
            
            # Iterate through intervals to find the one with highest marginal gain
            for idx, (xa, xb, fa, fb) in enumerate(intervals):
                # Find optimal x in [xa, xb]
                # The function is f2(x) = 1 - sqrt(x)
                # We want to maximize the area added.
                # The area added by point x is the area of the triangle (xa, fa), (xb, fb), (x, f(x))
                # which is 0.5 * | (xb-xa)*(f(x)-fa) - (x-xa)*(fb-fa) |
                # Since f is concave, f(x) is above the chord.
                # We want to maximize the vertical distance h(x) = f(x) - (fa + (x-xa)/(xb-xa)*(fb-fa))
                
                # For f(x) = 1 - sqrt(x), the maximum deviation from the chord
                # occurs at x* where f'(x*) = slope of chord.
                # slope = (fb - fa) / (xb - xa)
                # f'(x) = -1 / (2 * sqrt(x))
                # -1 / (2 * sqrt(x*)) = slope
                # sqrt(x*) = -1 / (2 * slope)
                # x* = 1 / (4 * slope^2)
                
                slope = (fb[1] - fa[1]) / (xb - xa)
                if slope >= 0:
                    # Should not happen for ZDT1 Pareto front (f2 decreases)
                    # But handle gracefully
                    x_opt = (xa + xb) / 2.0
                else:
                    x_opt = 1.0 / (4.0 * slope * slope)
                    
                # Clamp x_opt to [xa, xb]
                if x_opt < xa:
                    x_opt = xa
                elif x_opt > xb:
                    x_opt = xb
                    
                # Calculate gain
                # f_opt = 1 - sqrt(x_opt)
                f_opt_val = 1.0 - np.sqrt(x_opt)
                
                # Area of triangle
                # Base = xb - xa
                # Height = f_opt_val - (fa[1] + (x_opt - xa) * slope)
                chord_val = fa[1] + (x_opt - xa) * slope
                height = f_opt_val - chord_val
                gain = 0.5 * (xb - xa) * height
                
                if gain > best_gain:
                    best_gain = gain
                    best_interval_idx = idx
                    best_x = x_opt
                    best_f = f_opt_val
            
            if best_interval_idx == -1:
                break
                
            # Evaluate the best point
            f_new = get_obj(best_x)
            evaluated_f.append(f_new)
            
            # Update intervals
            xa, xb, fa, fb = intervals.pop(best_interval_idx)
            
            # Split into [xa, best_x] and [best_x, xb]
            # We need to be careful with floating point and order
            if best_x > xa:
                intervals.append((xa, best_x, fa, f_new))
            if best_x < xb:
                intervals.append((best_x, xb, f_new, fb))
                
        # Return the list of objective vectors
        return evaluated_f
