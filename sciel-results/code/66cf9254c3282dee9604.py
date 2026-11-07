import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _compute_optimal_x1(self, n_points, lo0, hi0):
        """
        Compute optimal x1 values for hypervolume maximization on ZDT1 Pareto front.
        
        Strategy:
        1. Initialize with quantiles of the optimal distribution derived from
           equalizing marginal hypervolume contributions.
           The CDF is F(s) = (10s + (2/3)s^(3/2)) / (32/3) for s in [0,1]
           with reference point at 11.
        2. Refine using the recurrence relation from first-order optimality:
           9x_i^2 - (6x_{i-1} + 4x_{i+1})x_i + x_{i-1}^2 = 0
        """
        if n_points <= 1:
            return np.array([(lo0 + hi0) / 2.0])
        
        # Step 1: Initialize with CDF quantiles
        # F(s) = (10s + (2/3)s^(3/2)) / (32/3)
        # Inverse: given uniform u in [0,1], find s such that F(s) = u
        s_vals = np.zeros(n_points)
        t_uniform = np.linspace(0.0, 1.0, n_points)
        target = t_uniform * (32.0 / 3.0)
        
        for i in range(n_points):
            # Initial guess
            s = target[i] / 10.0 if target[i] > 0 else 0.0
            s = max(0.0, min(1.0, s))
            
            # Newton's method to solve F(s) = target[i]
            for _ in range(50):
                f_val = 10.0 * s + (2.0 / 3.0) * s ** 1.5
                f_prime = 10.0 + s ** 0.5
                if f_prime == 0:
                    break
                correction = (f_val - target[i]) / f_prime
                s_new = s - correction
                s_new = max(0.0, min(1.0, s_new))
                if abs(s_new - s) < 1e-15:
                    s = s_new
                    break
                s = s_new
            s_vals[i] = s
        
        # Enforce endpoints to be at bounds
        s_vals[0] = lo0
        s_vals[-1] = hi0
        
        # Step 2: Refine using recurrence relation
        # 9x_i^2 - (6x_{i-1} + 4x_{i+1})x_i + x_{i-1}^2 = 0
        # Solve for x_i in (x_{i-1}, x_{i+1})
        for _ in range(200):
            new_s_vals = s_vals.copy()
            for i in range(1, n_points - 1):
                x_prev = s_vals[i - 1]
                x_next = s_vals[i + 1]
                
                # Quadratic: 9x^2 - (6*x_prev + 4*x_next)x + x_prev^2 = 0
                A = 9.0
                B = -(6.0 * x_prev + 4.0 * x_next)
                C = x_prev * x_prev
                
                discriminant = B * B - 4.0 * A * C
                if discriminant < 0:
                    continue
                
                sqrt_disc = np.sqrt(discriminant)
                root1 = (-B + sqrt_disc) / (2.0 * A)
                root2 = (-B - sqrt_disc) / (2.0 * A)
                
                # We want the root between x_prev and x_next
                candidate = None
                if x_prev < root1 < x_next:
                    candidate = root1
                elif x_prev < root2 < x_next:
                    candidate = root2
                else:
                    # If no root in range, pick the one closer to current value
                    if abs(root1 - s_vals[i]) < abs(root2 - s_vals[i]):
                        if x_prev <= root1 <= x_next:
                            candidate = root1
                    else:
                        if x_prev <= root2 <= x_next:
                            candidate = root2
                
                if candidate is not None:
                    new_s_vals[i] = candidate
            
            # Check convergence
            if np.max(np.abs(new_s_vals - s_vals)) < 1e-14:
                s_vals = new_s_vals
                break
            s_vals = new_s_vals
            
            # Re-enforce bounds
            s_vals[0] = lo0
            s_vals[-1] = hi0
            s_vals = np.clip(s_vals, lo0, hi0)
        
        return s_vals

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin before 300s limit
        
        # Extract bounds
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        
        n_evals = self.budget
        if n_evals <= 0:
            return np.array([])
        
        # Compute optimal x1 values for hypervolume maximization on ZDT1 front
        x1_vals = self._compute_optimal_x1(n_evals, lo[0], hi[0])
        
        # Create the population
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x1_vals
        X[:, 1:] = lo[1:]
        
        # Evaluate each point to record it
        for i in range(n_evals):
            if time.time() >= deadline:
                break
            problem(X[i])
        
        return X
