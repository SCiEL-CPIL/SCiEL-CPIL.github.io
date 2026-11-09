import numpy as np
import time


class Optimizer:
    """
    Optimizer for DTLZ7 that exploits the structure of the problem.
    
    DTLZ7 with n_obj=3:
    f1 = x[0]
    f2 = x[1]  
    f3 = (1 + g) * (n_obj - 1 - sum_{i=2}^{M} (1+x[i]) * sin(pi * x[i] / 2))
    where g = sum_{i=2}^{M} (x[i] - 0.5)^2 and M = dim-1.
    
    Key insight: The Pareto front is a 2D surface parameterized by x[0] and x[1].
    However, f3 depends on the tail variables x[2..M]. To minimize g (and thus
    get points close to the Pareto front), we need x[i] = 0.5 for i >= 2.
    
    But wait - if all tail vars are 0.5, then f3 is constant for all (x0, x1).
    This means the "front" in objective space is actually a line segment in the
    f1-f2 plane at a fixed f3 value. The hypervolume is determined by the spread
    in f1 and f2 values.
    
    Strategy: Use a dense grid on x[0] and x[1] with tail variables fixed to 0.5.
    This maximizes the spread in f1 and f2, which maximizes hypervolume.
    """
    
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
    
    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # For DTLZ7, the optimal tail values are 0.5 (minimizes g)
        tail_val = 0.5 * (lo[2:] + hi[2:])  # Midpoint of bounds for tail variables
        
        n = self.budget
        # Use a square-ish grid on x[0] and x[1]
        # We want to cover [lo[0], hi[0]] x [lo[1], hi[1]] densely
        side = int(np.ceil(np.sqrt(n)))
        if side * side > n:
            side -= 1
        
        # Generate grid points
        x0_vals = np.linspace(lo[0], hi[0], side)
        x1_vals = np.linspace(lo[1], hi[1], side)
        
        # Evaluate all grid points
        F = []
        for x0 in x0_vals:
            for x1 in x1_vals:
                if time.time() > deadline:
                    break
                x = np.empty(self.dim)
                x[0] = x0
                x[1] = x1
                x[2:] = tail_val
                F.append(problem(x))
            if time.time() > deadline:
                break
        
        # If we have budget left, add some random points for robustness
        remaining = self.budget - len(F)
        if remaining > 0 and time.time() < deadline:
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * np.random.rand(self.dim)
                F.append(problem(x))
        
        return F
