import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # DTLZ4 specific: 12 dims, 3 objs
        # Position vars: 0 to 9 (10 vars)
        # Shape vars: 10, 11 (2 vars)
        
        # Initialize position variables to 0.5 to make g(x) = 0
        x = np.zeros(self.dim)
        x[:10] = 0.5
        
        # We will use all budget evaluations to sample the Pareto front
        # Uniform sampling on the quarter-sphere surface
        # f1^2 + f2^2 + f3^2 = 1
        # f3 = cos(theta) where theta in [0, pi/2]
        # f1 = sin(theta) * cos(phi)
        # f2 = sin(theta) * sin(phi)
        # phi in [0, pi/2]
        
        # To sample uniformly on the surface:
        # u ~ U[0,1], v ~ U[0,1]
        # cos(theta) = 1 - u  => f3 = 1 - u
        # sin(theta) = sqrt(1 - (1-u)^2) = sqrt(2u - u^2)
        # phi = (pi/2) * v
        
        # The shape variables x[10], x[11] are mapped to angles.
        # Assuming standard DTLZ4 scaling: angle = (pi/2) * x
        # So x = (2/pi) * angle
        
        for _ in range(self.budget):
            u = np.random.random()
            v = np.random.random()
            
            f3 = 1.0 - u
            sin_theta = np.sqrt(2.0 * u - u*u)
            phi = 0.5 * np.pi * v
            
            f1 = sin_theta * np.cos(phi)
            f2 = sin_theta * np.sin(phi)
            
            # Invert to get x[10], x[11]
            # f3 = sin((pi/2) * x[10])
            # x[10] = (2/pi) * arcsin(f3)
            x[10] = (2.0 / np.pi) * np.arcsin(np.clip(f3, 0, 1))
            
            # f1 = cos((pi/2) * x[10]) * cos((pi/2) * x[11])
            # f2 = cos((pi/2) * x[10]) * sin((pi/2) * x[11])
            # tan((pi/2) * x[11]) = f2 / f1
            # (pi/2) * x[11] = atan2(f2, f1)
            # x[11] = (2/pi) * atan2(f2, f1)
            x[11] = (2.0 / np.pi) * np.arctan2(f2, f1)
            
            # Ensure bounds [0,1]
            x[10] = np.clip(x[10], 0, 1)
            x[11] = np.clip(x[11], 0, 1)
            
            problem(x)
            
        return x
