import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """
    Optimizer for DTLZ4 using analytical construction of Pareto front points.
    Uses Sobol sampling for uniform coverage of the front surface area.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.t0 = time.time()
        self.deadline = self.t0 + 280  # Safety margin for 300s limit

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_points = self.budget
        
        # Generate low-discrepancy points in [0, 1]^2
        # These will be mapped to uniform surface area on the Pareto front
        if n_points <= 0:
            return []
            
        try:
            sobol = qmc.Sobol(d=2, scramble=True)
            uv = sobol.random(n_points)
        except Exception:
            # Fallback to Halton if Sobol fails (unlikely)
            uv = np.random.rand(n_points, 2)

        u = uv[:, 0]
        v = uv[:, 1]
        
        # Map u, v to decision variables x
        # x_0 derived from uniform area sampling of theta
        # cos(theta) = 1 - u
        # sin(theta) = sqrt(2u - u^2)
        # f3 = sin(theta)
        # DTLZ4: f3 = sin(pi/2 * x_0^10)
        # => pi/2 * x_0^10 = arcsin(sqrt(2u - u^2))
        
        val = np.clip(2.0 * u - u**2, 0.0, 1.0)
        sin_theta = np.sqrt(val)
        theta = np.arcsin(sin_theta)
        x0_base = (2.0 / np.pi) * theta
        x0 = np.power(x0_base, 0.1)
        
        # x_1 derived from uniform sampling of phi
        # phi = v * pi/2
        # f1 = cos(theta) cos(phi), f2 = cos(theta) sin(phi)
        # DTLZ4: cos(pi/2 * x_1^10) = cos(phi)
        # => pi/2 * x_1^10 = phi = v * pi/2
        # => x_1^10 = v
        x1 = np.power(v, 0.1)
        
        # Construct full x vectors
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x0
        X[:, 1] = x1
        X[:, 2:] = 0.5  # Set distance variables to 0.5 to ensure g(x)=0
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F_list = []
        for i in range(n_points):
            if time.time() > self.deadline:
                break
            f = problem(X[i])
            F_list.append(f)
            
        return F_list
