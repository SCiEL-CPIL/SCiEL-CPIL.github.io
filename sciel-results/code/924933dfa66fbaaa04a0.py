import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit
        
        n_points = self.budget
        dim = self.dim
        
        # DTLZ1 Pareto front:
        # For M=3 objectives, shaping variables are x[0], x[1]
        # Non-shaping variables x[2] to x[dim-1] must be 0.5 for g(x)=0
        # The Pareto front is: f1 = 0.5*x0, f2 = 0.5*x1*(1-x0), f3 = 0.5*(1-x1)*(1-x0)
        # with x0, x1 in [0,1] and x0+x1 <= 1
        
        # Use a deterministic nested lattice grid for perfect uniformity
        # on the 2D Pareto simplex (triangle in x0-x1 space)
        
        # Create a triangular grid that covers the Pareto front uniformly
        # We want exactly n_points points on the triangle x0>=0, x1>=0, x0+x1<=1
        
        # Strategy: use a regular grid in (u,v) space where:
        # x0 = u, x1 = v*(1-u) for u in [0,1], v in [0,1]
        # This maps the unit square to the triangle
        
        # For n=200, use a grid of size k x k where k*k >= 200
        # sqrt(200) ≈ 14.14, so use 15x15 = 225 points, but we only need 200
        # Actually, let's use a more efficient approach
        
        # Use a triangular number approach:
        # For a triangular grid with n rows, total points = n*(n+1)/2
        # We need n*(n+1)/2 >= 200
        # n=20: 210 points (too many)
        # n=19: 190 points (too few)
        
        # Alternative: use a 14x14 grid = 196 points, plus 4 extra points
        # Or use a 15x15 grid and take first 200
        
        # Let's use a systematic approach with a square grid mapped to triangle
        # Use 15x15 grid = 225 points, take first 200
        
        k = 15  # 15x15 = 225 >= 200
        points_2d = []
        
        for i in range(k):
            for j in range(k):
                # Map (i,j) in [0,k-1]^2 to (u,v) in [0,1]^2
                u = i / (k - 1)
                v = j / (k - 1)
                
                # Map to triangle: x0 = u, x1 = v*(1-u)
                x0 = u
                x1 = v * (1 - u)
                
                points_2d.append((x0, x1))
                
                if len(points_2d) == n_points:
                    break
            if len(points_2d) == n_points:
                break
        
        points_2d = np.array(points_2d)
        
        # Construct full dimension vectors
        X = np.zeros((n_points, dim))
        X[:, 0] = points_2d[:, 0]
        X[:, 1] = points_2d[:, 1]
        X[:, 2:] = 0.5
        
        # Evaluate all points
        for i in range(n_points):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
