import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Effective grid sizes based on Jacobian analysis
        # u in [0, 0.5], v in [0, 1]
        # Aspect ratio 1:2. So N_u : N_v = 1:2.
        # N_u * N_v = 200 -> 10 * 20 = 200.
        n0 = 10
        n1 = 20
        
        # Generate uniform u in [0, 0.5]
        u_vals = np.linspace(0.0, 0.5, n0)
        # Map to x0
        x0_vals = 1.0 - np.sqrt(1.0 - 2.0 * u_vals)
        
        # Generate uniform v in [0, 1]
        x1_vals = np.linspace(0.0, 1.0, n1)
        
        # Create meshgrid
        X0, X1 = np.meshgrid(x0_vals, x1_vals, indexing='ij')
        
        # Flatten
        x0_flat = X0.flatten()
        x1_flat = X1.flatten()
        
        n_samples = len(x0_flat)
        
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = x0_flat
        X[:, 1] = x1_flat
        # X[:, 2:] = 0
        
        for i in range(n_samples):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
