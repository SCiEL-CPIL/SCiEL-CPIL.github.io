import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Generate 200 points on the Pareto front of DTLZ2
        # DTLZ2 with M=3, N=12
        # k = N - M + 1 = 10
        # x[0:10] determine g. Set to 0.5 to make g=0.
        # x[10] and x[11] determine angles.
        
        n_points = self.budget
        points = np.zeros((n_points, self.dim))
        
        # Set g variables to 0.5
        points[:, :10] = 0.5
        
        # Generate random angles
        # u for cos(theta), v for phi
        u = np.random.rand(n_points)
        v = np.random.rand(n_points)
        
        # theta = arccos(u)
        theta = np.arccos(u)
        # phi = v * pi / 2
        phi = v * (np.pi / 2)
        
        # Map to x[10], x[11]
        # x[11] corresponds to theta in f3 = sin(x11*pi/2)
        # Wait, standard DTLZ2:
        # f3 = (1+g) * sin(x_{N-1} * pi / 2)
        # So x_{N-1} = x[11] is the polar angle variable.
        # x[11] = 2 * theta / pi
        points[:, 11] = 2 * theta / np.pi
        
        # f1 = (1+g) * cos(x_{N-1} * pi / 2) * cos(x_{N-2} * pi / 2)
        # f2 = (1+g) * cos(x_{N-1} * pi / 2) * sin(x_{N-2} * pi / 2)
        # So x_{N-2} = x[10] is the azimuthal angle variable.
        # x[10] = 2 * phi / pi
        points[:, 10] = 2 * phi / np.pi
        
        # Evaluate all points
        for i in range(n_points):
            problem(points[i])
            
        return points
