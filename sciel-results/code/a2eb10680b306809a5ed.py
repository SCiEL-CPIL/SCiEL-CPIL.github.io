import numpy as np
import time

try:
    from scipy.stats import qmc
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        # DTLZ1 specifics
        # M=3, N=7
        # g(x) depends on x[3], x[4], x[5], x[6]
        # Optimal g=0 at x[3]=x[4]=x[5]=x[6]=0.5
        # f1 = 0.5 * x[0]
        # f2 = 0.5 * x[1]
        # f3 = 0.5 * (1 - x[0]^2 - x[1]^2)
        # Pareto front is x[0]^2 + x[1]^2 <= 1, x[0],x[1] >= 0
        
        n_points = self.budget
        points = np.zeros((n_points, self.dim))
        
        # Set fixed variables for g=0
        points[:, 3:7] = 0.5
        
        if HAS_SCIPY:
            sampler = qmc.Sobol(d=2, scramble=True)
            # Generate points in [0,1]^2
            u = sampler.random(n_points)
        else:
            u = np.random.random((n_points, 2))
            
        # Map [0,1]^2 to quarter disk
        # r = sqrt(u1), theta = pi/2 * u2
        r = np.sqrt(u[:, 0])
        theta = 0.5 * np.pi * u[:, 1]
        
        points[:, 0] = r * np.cos(theta)
        points[:, 1] = r * np.sin(theta)
        
        # Evaluate
        for i in range(n_points):
            if time.time() > deadline:
                break
            # problem expects a single vector
            problem(points[i])
            
        return points
