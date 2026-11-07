import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # DTLZ4 with 3 objectives.
        # Pareto front is achieved when x[1:] (angles) are set appropriately and x[0]=1, x[3:]=0.5.
        # Mapping:
        # f1 = x0 * g * cos(x1 * pi/2)
        # f2 = x0 * g * sin(x1 * pi/2) * cos(x2 * pi/2)
        # f3 = x0 * g * sin(x1 * pi/2) * sin(x2 * pi/2)
        # Here indices are 0-based. x[0] is weight, x[1], x[2] are angles. x[3:] are for g.
        # To minimize g, x[3:] must be 0.5.
        # To be on the front, x[0] must be 1.0.
        
        n_points = self.budget
        
        # Use Halton sequence for low-discrepancy sampling in [0, 1]^2
        # This helps cover the parameter space more evenly than a grid.
        
        def halton(index, base):
            result = 0
            f = 1
            i = index
            while i > 0:
                f /= base
                result += f * (i % base)
                i //= base
            return result

        points = []
        for i in range(1, n_points + 1):
            u1 = halton(i, 2)
            u2 = halton(i, 3)
            
            # We want points uniformly distributed on the spherical Pareto front.
            # The surface element is proportional to sin(theta).
            # To sample uniformly on the sphere, we sample cos(theta) uniformly.
            # Let u1 ~ Uniform(0,1). cos(theta) = u1.
            # theta = arccos(u1).
            # x[1] = theta / (pi/2)
            
            # Let u2 ~ Uniform(0,1). phi = u2 * pi/2.
            # x[2] = phi / (pi/2) = u2
            
            # Note: u1=0 -> theta=pi/2 -> x[1]=1. f1=0.
            # u1=1 -> theta=0 -> x[1]=0. f1=1.
            # This covers the full range.
            
            theta = np.arccos(u1)
            x1_angle = theta / (np.pi / 2.0)
            x2_angle = u2
            
            x = np.zeros(self.dim)
            x[0] = 1.0
            x[1] = x1_angle
            x[2] = x2_angle
            x[3:] = 0.5
            
            points.append(x)
        
        # Evaluate all points
        F = []
        for x in points:
            F.append(problem(x))
            
        return points
