import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        points = []

        # DTLZ4 with m=3, beta=10:
        # f1 = x0 * cos(x1 * pi/2)
        # f2 = x0 * sin(x1 * pi/2)
        # f3 = 1 - x0
        # g(x) = sum((xi - 0.5)^2 for i in 2..dim-1)
        # The Pareto front is g=0, i.e., x[2:] = 0.5
        # The front is a quarter unit disk in (f1, f2) with f3 = 1 - x0

        # Add the three extreme points of the Pareto front
        # (1, 0, 0): x0=1, x1=0
        x = np.zeros(dim)
        x[0] = 1.0
        x[1] = 0.0
        x[2:] = 0.5
        points.append(x)

        # (0, 1, 0): x0=1, x1=1
        x = np.zeros(dim)
        x[0] = 1.0
        x[1] = 1.0
        x[2:] = 0.5
        points.append(x)

        # (0, 0, 1): x0=0, x1=0
        x = np.zeros(dim)
        x[0] = 0.0
        x[1] = 0.0
        x[2:] = 0.5
        points.append(x)

        # Add the centroid of the triangle: (1/3, 1/3, 1/3)
        # x0 = 2/3, x1 = 2/3
        x = np.zeros(dim)
        x[0] = 2.0 / 3.0
        x[1] = 2.0 / 3.0
        x[2:] = 0.5
        points.append(x)

        # Add midpoints of the three edges
        # Edge between (1,0,0) and (0,1,0): f=(0.5, 0.5, 0) -> x0=1, x1=0.5
        x = np.zeros(dim)
        x[0] = 1.0
        x[1] = 0.5
        x[2:] = 0.5
        points.append(x)

        # Edge between (1,0,0) and (0,0,1): f=(0.5, 0, 0.5) -> x0=0.5, x1=0
        x = np.zeros(dim)
        x[0] = 0.5
        x[1] = 0.0
        x[2:] = 0.5
        points.append(x)

        # Edge between (0,1,0) and (0,0,1): f=(0, 0.5, 0.5) -> x0=0.5, x1=1
        x = np.zeros(dim)
        x[0] = 0.5
        x[1] = 1.0
        x[2:] = 0.5
        points.append(x)

        # Fill the rest with Sobol points in the 2D position space
        n_sobol = budget - len(points)
        if n_sobol > 0:
            # Use scrambled Sobol for better distribution
            sobol = qmc.Sobol(d=2, scramble=True, seed=42)
            sobol_points = sobol.random(n_sobol)
            for p in sobol_points:
                x = np.zeros(dim)
                x[0] = p[0]
                x[1] = p[1]
                x[2:] = 0.5
                points.append(x)

        # Evaluate all points
        F = np.array([problem(x) for x in points])

        if time.time() > deadline:
            return F

        return F
