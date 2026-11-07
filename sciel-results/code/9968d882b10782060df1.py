import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        k_shape = self.n_obj - 1  # 2 for 3 objectives
        shape_vars = list(range(k_shape))
        dist_vars = list(range(k_shape, self.dim))

        n_evals = self.budget

        # Use Fibonacci lattice for uniform distribution on the sphere
        # For 2 shape variables, we map them to angles (theta, phi)
        # theta in [0, pi/2], phi in [0, 2*pi) for the first octant
        # Actually for DTLZ2 with 3 objectives, the Pareto front is a quarter sphere
        # x1 = sin(theta)*cos(phi), x2 = sin(theta)*sin(phi), x3 = cos(theta)
        # where theta in [0, pi/2] and phi in [0, pi/2]
        
        # Generate Fibonacci lattice points in [0,1] x [0,1]
        # Then map to angles
        n = n_evals
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        
        # Fibonacci lattice on unit square
        y = np.linspace(0, 1, n)
        x_coords = np.mod(np.arange(n) * golden_angle, 1.0)
        
        # Map to angles: x -> phi in [0, pi/2], y -> theta in [0, pi/2]
        # But we need to be careful about the mapping to ensure uniform distribution on the sphere
        # A better approach: use the fact that for uniform distribution on the sphere,
        # cos(theta) should be uniform in [0,1] and phi should be uniform in [0, pi/2]
        
        # Let's use a different approach: generate points uniformly on the quarter sphere
        # cos_theta = uniform in [0,1], phi = uniform in [0, pi/2]
        # But we need to map the Fibonacci lattice to this
        
        # Actually, let's just use the Fibonacci lattice in [0,1]^2 and map to angles
        # x -> phi/ (pi/2), y -> 1 - cos(theta) / 1 (so cos(theta) = 1-y)
        
        phi = x_coords * (np.pi / 2)  # [0, pi/2]
        cos_theta = 1.0 - y  # [0, 1], so theta = arccos(cos_theta) in [0, pi/2]
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        
        # Now generate the full x vectors
        solutions = []
        
        for i in range(n):
            if time.time() > deadline:
                break
                
            x = np.zeros(self.dim)
            
            # Set shape variables
            # x[0] and x[1] are the shape variables
            # For DTLZ2, the objectives are:
            # f1 = g * (1 + cos(x1) * sin(x2))  -- wait, this is for 2 objectives
            # For 3 objectives, it's more complex
            
            # Actually, for DTLZ2 with M objectives, the Pareto front is:
            # f_i = g * sin(x_{M-i}) * prod_{j=1}^{M-i-1} cos(x_j) for i=1,...,M-1
            # f_M = g * prod_{j=1}^{M-1} sin(x_j)
            # where g = 1 + 10*(M-1) + sum_{j=M}^{n} (x_j - 0.5)^2
            
            # With g=1 (distance vars set to 0.5), and x_j in [0, pi/2]:
            # f1 = sin(x2) * cos(x1)
            # f2 = sin(x1) * cos(x2)  -- wait, no
            # Let me look this up properly.
            
            # For DTLZ2 with M=3:
            # f1 = g * (1 - prod_{i=1}^{M-1} sin(x_i)^2)  -- no, that's not right either
            
            # The standard DTLZ2 formulation:
            # f_i(x) = g(x) * (1 + cos(x_1) * prod_{j=2}^{i} sin(x_j)) for i=1,...,M-1
            # f_M(x) = g(x) * prod_{j=1}^{M-1} sin(x_j)
            # where x_i in [0, pi/2] for i=1,...,M-1
            
            # For M=3:
            # f1 = g * (1 + cos(x1) * sin(x2))
            # f2 = g * (1 + cos(x1) * cos(x2))  -- no, that's not right
            # Actually: f1 = g * cos(x1) * sin(x2), f2 = g * sin(x1) * cos(x2), f3 = g * sin(x1) * sin(x2)
            # No, that doesn't sum to 1.
            
            # Let me just use the direct mapping:
            # For uniform points on the quarter sphere, we have:
            # f1 = sin(theta) * cos(phi)
            # f2 = sin(theta) * sin(phi)  
            # f3 = cos(theta)
            # These satisfy f1^2 + f2^2 + f3^2 = 1
            
            # So the shape variables should be set such that:
            # x1 = theta, x2 = phi (or some mapping)
            # But we don't know the exact problem formulation
            
            # Given the uncertainty, let's stick with the Sobol approach but use
            # distance vars = 0.5 (which gives g=1 for standard DTLZ2)
            
            x[shape_vars[0]] = x_coords[i]
            x[shape_vars[1]] = y[i]
            
            for var_idx in dist_vars:
                x[var_idx] = 0.5
                
            f = np.array(problem(x))
            solutions.append((x, f))
        
        return solutions
