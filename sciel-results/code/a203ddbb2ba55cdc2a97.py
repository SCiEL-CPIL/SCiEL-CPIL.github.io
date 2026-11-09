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
        
        # Multi-scale Coordinate Descent on the best start
        # Systematic scan with geometrically shrinking step size
        h = 2.0
        min_h = 1e-6
        
        while self.calls < self.budget and h >= min_h:
            moved = False
            for i in range(self.dim):
                if self.calls >= self.budget:
                    break
                
                # Try positive step
                x_plus = best_x.copy()
                x_plus[i] += h
                if x_plus[i] > 5.0:
                    x_plus[i] = 5.0
                if not np.allclose(x_plus, best_x):
                    f_plus = safe_func(x_plus)
                    if f_plus < best_f:
                        best_x = x_plus
                        best_f = f_plus
                        moved = True
                        continue

                # Try negative step
                x_minus = best_x.copy()
                x_minus[i] -= h
                if x_minus[i] < -5.0:
                    x_minus[i] = -5.0
                if not np.allclose(x_minus, best_x):
                    f_minus = safe_func(x_minus)
                    if f_minus < best_f:
                        best_x = x_minus
                        best_f = f_minus
                        moved = True
            
            if not moved:
                h *= 0.5
            else:
                # Shrink slightly to encourage finer steps even if moving
                h *= 0.9

        return best_x, best_f
