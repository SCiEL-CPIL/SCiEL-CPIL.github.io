import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0

    def __call__(self, func):
        # Track calls
        def safe_func(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            self.calls += 1
            return func(x)

        # Deterministic starting points
        # Covering center and symmetric off-center points
        starts = [
            np.zeros(self.dim),
            np.full(self.dim, 1.0),
            np.full(self.dim, -1.0),
            np.full(self.dim, 2.0),
            np.full(self.dim, -2.0),
            np.full(self.dim, 3.0),
            np.full(self.dim, -3.0),
            np.full(self.dim, 4.0),
            np.full(self.dim, -4.0)
        ]
        
        # Evaluate starts
        start_values = []
        for s in starts:
            f_s = safe_func(s)
            start_values.append(f_s)
        
        # Sort starts by value (best first)
        sorted_indices = np.argsort(start_values)
        
        best_x = starts[sorted_indices[0]].copy()
        best_f = start_values[sorted_indices[0]]
        
        # Multi-Scale Systematic Scan
        # We define a sequence of step sizes.
        # For each step size, we scan each dimension across the box [-5, 5].
        step_sizes = [2.0, 1.0, 0.5, 0.25]
        
        for h in step_sizes:
            if self.calls >= self.budget:
                break
            
            for i in range(self.dim):
                if self.calls >= self.budget:
                    break
                
                # Scan dimension i from -5 to 5 with step h
                scan_points = np.arange(-5.0, 5.0 + h/2, h)
                for val in scan_points:
                    if self.calls >= self.budget:
                        break
                    
                    x_candidate = best_x.copy()
                    x_candidate[i] = val
                    f_val = safe_func(x_candidate)
                    if f_val < best_f:
                        best_f = f_val
                        best_x = x_candidate
        
        # Fine-tuning with remaining budget
        # Use a simple coordinate descent with small steps to polish
        h = 0.1
        min_h = 1e-4
        while self.calls < self.budget and h >= min_h:
            moved = False
            for i in range(self.dim):
                if self.calls >= self.budget:
                    break
                
                # Try positive step
                x_plus = best_x.copy()
                x_plus[i] += h
                if x_plus[i] <= 5.0:
                    f_plus = safe_func(x_plus)
                    if f_plus < best_f:
                        best_x = x_plus
                        best_f = f_plus
                        moved = True
                        continue

                # Try negative step
                x_minus = best_x.copy()
                x_minus[i] -= h
                if x_minus[i] >= -5.0:
                    f_minus = safe_func(x_minus)
                    if f_minus < best_f:
                        best_x = x_minus
                        best_f = f_minus
                        moved = True
            
            if not moved:
                h *= 0.5
            else:
                h *= 0.8

        return best_x, best_f
