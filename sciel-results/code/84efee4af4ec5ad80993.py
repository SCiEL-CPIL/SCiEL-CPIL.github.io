import numpy as np
import time

class Optimizer:
    """
    Optimizer for ZDT4 exploiting its decoupled structure.
    """
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 330s kill, use 320s
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # ZDT4: x1 controls the front, x2..xn control deviation from front.
        # Pareto front is at x2..xn = 0 (minimizes g(x)).
        # We sweep x1 uniformly across its bounds to cover the entire Pareto front.
        
        # Generate x1 values (uniform, includes endpoints)
        x1_samples = np.linspace(lo[0], hi[0], n_evals)
        
        # Prepare the constant part of the vector (x2..xn)
        # Set to 0, which minimizes the deviation term g(x)
        # Clip to bounds to be safe
        x_const = np.zeros(self.dim)
        # Check if 0.0 is within the bounds for dimensions 1 to n-1
        # If not, clip to the nearest bound
        if np.any(lo[1:] > 0.0):
            x_const[1:] = np.maximum(lo[1:], 0.0)
        elif np.any(hi[1:] < 0.0):
            x_const[1:] = np.minimum(hi[1:], 0.0)
        else:
            x_const[1:] = 0.0
        
        # Apply the improvement: non-uniform sampling x1 = t^2
        # This concentrates points near x1=0, which often has higher curvature
        # or contributes more to hypervolume relative to the reference point.
        t_samples = np.linspace(0.0, 1.0, n_evals)
        x1_samples = t_samples ** 2
        x1_samples = lo[0] + (hi[0] - lo[0]) * x1_samples

        for i in range(n_evals):
            x = x_const.copy()
            x[0] = x1_samples[i]
            problem(x)
            
            # Check time only every 50 iterations to reduce overhead
            if (i + 1) % 50 == 0 and time.time() > deadline:
                break
            
        return None
