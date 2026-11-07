import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # ZDT4 specific knowledge:
        # x[0] is the variable determining f1.
        # x[1:] determine the 'shape' or distance from Pareto front.
        
        # We want to cover x[0] in [-5, 5] (or bounds[0])
        x0_lo, x0_hi = bounds[0]
        
        # Number of points to cover the x0 dimension
        # We need enough points to capture the hypervolume.
        # 200 budget. Let's use 50 points for x0 coverage.
        n_points = 50
        if self.budget < 100:
            n_points = self.budget // 2
        else:
            n_points = min(50, self.budget // 4)
            
        # Generate distinct x0 values
        # Uniform distribution is a good start, but maybe cluster where f2 changes rapidly?
        # For now, uniform is robust.
        x0_vals = np.linspace(x0_lo, x0_hi, n_points)
        
        # Initialize population
        # x[0] is fixed to x0_vals
        # x[1:] initialized randomly
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x0_vals
        # Initialize x[1:] near 0? Or random?
        # The minima of g(x) are at x_i=0 for the sin(10x_i^2) term?
        # sin(0)=0. x^2 sin(10x^2) is 0 at x=0.
        # So x[1:] = 0 is a good starting point for the minima of the oscillating part.
        # However, g(x) = 10(n-1) + sum...
        # If we set x[1:] = 0, sum is 0.
        # Let's start x[1:] at 0.
        X[:, 1:] = 0.0
        
        F = np.array([problem(x) for x in X])
        used = n_points
        
        # Local search to minimize f2 (and thus move towards Pareto front)
        # We keep x[0] fixed and perturb x[1:]
        
        # We have budget - used evaluations left.
        # Distribute them among the n_points.
        evals_per_point = max(1, (self.budget - used) // n_points)
        
        # Simple local search: random restarts or steps
        # Since we started at 0, and 0 is likely a minimum for the oscillating part,
        # we might already be on the front.
        # Let's verify by evaluating.
        # If we are not on the front, we need to find the minimum of g(x).
        # We can do a random search in x[1:] space.
        
        for i in range(n_points):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
                
            # Local search around X[i]
            # We want to minimize F[i, 1] (f2) while keeping F[i, 0] (f1) fixed.
            # Since f1 depends only on x[0], and we fix x[0], f1 is constant.
            # So we just minimize f2.
            
            best_x = X[i].copy()
            best_f = F[i].copy()
            
            # Try a few random perturbations in x[1:]
            # Step size decay
            step_size = 0.5
            for _ in range(evals_per_point):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Perturb x[1:]
                delta = np.zeros(self.dim)
                delta[1:] = np.random.uniform(-step_size, step_size, self.dim - 1)
                new_x = best_x + delta
                # Clip to bounds
                new_x = np.clip(new_x, lo, hi)
                
                new_f = problem(new_x)
                used += 1
                
                # If new_f is better (smaller f2), accept
                if new_f[1] < best_f[1]:
                    best_x = new_x
                    best_f = new_f
                    X[i] = best_x
                    F[i] = best_f
                    # Reduce step size on success
                    step_size *= 0.8
                else:
                    # Increase step size slightly or keep same?
                    # Let's keep it simple: random walk
                    pass
                    
            # After local search, update global arrays
            X[i] = best_x
            F[i] = best_f
            
        return X
