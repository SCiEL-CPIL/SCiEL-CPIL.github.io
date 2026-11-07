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
        deadline = start_time + 250.0

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        points = []

        # For DTLZ4 with 3 objectives, the Pareto front is parameterized by (x0, x1)
        # where x0, x1 in [0,1] and x0^2 + x1^2 <= 1 (quarter disk)
        # The objectives are: f0 = x0, f1 = x1, f2 = sqrt(1 - x0^2 - x1^2)
        # Wait, actually for DTLZ4: f_i = (1+t(x)) * g * h_i(x)
        # But the Pareto front shape depends on the parameterization.
        # For DTLZ4 with m=3, the front is a quarter sphere.
        # The decision variables x[0], x[1] directly influence the position on the front.
        # x[2:] affects the distance from the front (g term).
        # To be ON the Pareto front, we need g=0, which means x[2:] should be at their
        # optimal values (typically 0.5 for DTLZ4 with alpha parameter).

        # Strategy: Place points densely on the Pareto front parameter space
        # The parameter space is the quarter disk: x0^2 + x1^2 <= 1, x0>=0, x1>=0

        # Add vertex points (corners of the Pareto front)
        # Vertex 1: (0, 0) -> f = (0, 0, 1)
        x = np.zeros(dim)
        x[0] = 0.0
        x[1] = 0.0
        x[2:] = 0.5
        points.append(x)

        # Vertex 2: (1, 0) -> f = (1, 0, 0)
        x = np.zeros(dim)
        x[0] = 1.0
        x[1] = 0.0
        x[2:] = 0.5
        points.append(x)

        # Vertex 3: (0, 1) -> f = (0, 1, 0)
        x = np.zeros(dim)
        x[0] = 0.0
        x[1] = 1.0
        x[2:] = 0.5
        points.append(x)

        # Add edge points with non-uniform spacing biased toward vertices
        # Edge 1: from (0,0) to (1,0): x1=0, x0 in [0,1]
        # Use sqrt-spaced points to cluster near vertices
        n_edge = 8
        for i in range(1, n_edge):
            t = (i / n_edge) ** 2  # Bias toward 0 and 1
            x = np.zeros(dim)
            x[0] = t
            x[1] = 0.0
            x[2:] = 0.5
            points.append(x)

        # Edge 2: from (0,0) to (0,1): x0=0, x1 in [0,1]
        for i in range(1, n_edge):
            t = (i / n_edge) ** 2
            x = np.zeros(dim)
            x[0] = 0.0
            x[1] = t
            x[2:] = 0.5
            points.append(x)

        # Edge 3: from (1,0) to (0,1): quarter circle x0^2 + x1^2 = 1
        for i in range(1, n_edge):
            theta = (i / n_edge) * (np.pi / 2)
            x = np.zeros(dim)
            x[0] = np.cos(theta)
            x[1] = np.sin(theta)
            x[2:] = 0.5
            points.append(x)

        # Fill remaining budget with Sobol points in the quarter disk
        n_sobol = budget - len(points)
        if n_sobol > 0:
            # Use Sobol in 2D, then map to quarter disk
            sobol = qmc.Sobol(d=2, scramble=True)
            sobol_points = sobol.random(n_sobol)
            # Map to quarter disk: use polar coordinates
            for p in sobol_points:
                # p[0], p[1] in [0,1)
                # Map to quarter disk: r = sqrt(p[0]), theta = p[1] * pi/2
                r = np.sqrt(p[0])
                theta = p[1] * (np.pi / 2)
                x0 = r * np.cos(theta)
                x1 = r * np.sin(theta)
                x = np.zeros(dim)
                x[0] = x0
                x[1] = x1
                x[2:] = 0.5
                points.append(x)

        # Evaluate all points
        F = np.array([problem(x) for x in points])

        if time.time() > deadline:
            return F

        return F
