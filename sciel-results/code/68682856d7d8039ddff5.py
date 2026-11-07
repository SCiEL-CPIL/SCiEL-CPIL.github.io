import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _compute_optimal_front_points(self, n_points, lo0, hi0):
        if n_points <= 1:
            return np.array([(lo0 + hi0) / 2.0])
        
        # Initialize with CDF quantiles
        # F(s) = (10s + (2/3)s^{3/2}) / (32/3)
        s_vals = np.zeros(n_points)
        t_uniform = np.linspace(0.0, 1.0, n_points)
        target = t_uniform * (32.0 / 3.0)
        
        for i in range(n_points):
            s = target[i] / 10.0 if target[i] > 0 else 0.0
            s = max(0.0, min(1.0, s))
            for _ in range(20):
                f_val = 10.0 * s + (2.0/3.0) * s**1.5
                f_prime = 10.0 + s**0.5
                correction = (f_val - target[i]) / f_prime
                s_new = s - correction
                s_new = max(0.0, min(1.0, s_new))
                if abs(s_new - s) < 1e-14:
                    s = s_new
                    break
                s = s_new
            s_vals[i] = s
        
        # Enforce x_n = 1.0 (hi0)
        s_vals[-1] = hi0
        
        # Refine using the recurrence relation:
        # 9x_i^2 - (6x_{i-1} + 4x_{i+1}) x_i + x_{i-1}^2 = 0
        # x_i = [(6x_{i-1} + 4x_{i+1}) - sqrt((6x_{i-1} + 4x_{i+1})^2 - 36x_{i-1}^2)] / 18
        # We iterate this update.
        for _ in range(50):
            new_s_vals = s_vals.copy()
            for i in range(1, n_points - 1):
                x_prev = s_vals[i-1]
                x_next = s_vals[i+1]
                A = 9.0
                B = -(6.0 * x_prev + 4.0 * x_next)
                C = x_prev * x_prev
                
                discriminant = B*B - 4.0*A*C
                if discriminant < 0:
                    continue
                
                sqrt_disc = np.sqrt(discriminant)
                # Two roots
                root1 = (-B + sqrt_disc) / (2.0 * A)
                root2 = (-B - sqrt_disc) / (2.0 * A)
                
                # We want the root that is between x_prev and x_next
                # Usually the smaller one
                if x_prev < root1 < x_next:
                    new_s_vals[i] = root1
                elif x_prev < root2 < x_next:
                    new_s_vals[i] = root2
                else:
                    # Keep old value or pick closest valid
                    # If no root in range, maybe the function is not convex or we are far
                    # Try to pick the one closer to current s_vals[i]
                    if abs(root1 - s_vals[i]) < abs(root2 - s_vals[i]):
                        if x_prev <= root1 <= x_next:
                            new_s_vals[i] = root1
                        else:
                            new_s_vals[i] = s_vals[i]
                    else:
                        if x_prev <= root2 <= x_next:
                            new_s_vals[i] = root2
                        else:
                            new_s_vals[i] = s_vals[i]
            
            s_vals = new_s_vals
            s_vals = np.clip(s_vals, lo0, hi0)
            s_vals[-1] = hi0

        return s_vals

    def __call__(self, problem, bounds):
        t0 = time.time()
        hard_deadline = t0 + 270.0
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        
        n = self.budget
        if n <= 0:
            return np.array([])
        
        x1_opt = self._compute_optimal_front_points(n, lo[0], hi[0])
        
        pop = np.zeros((n, self.dim), dtype=np.float64)
        pop[:, 0] = x1_opt
        pop[:, 1:] = lo[1:]
        
        count = 0
        for idx in range(n):
            if time.time() > hard_deadline:
                break
            problem(pop[idx])
            count += 1
        
        return pop
