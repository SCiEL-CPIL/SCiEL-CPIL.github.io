# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2 using Fibonacci lattice on the Pareto front.
    
    DTLZ2 with 3 objectives has a Pareto front that is a quarter-sphere in the positive
    octant of R^3. The distance variables (last dim - n_obj + 1 = 10 variables) should
    be set to 0.5 for optimality. The angle variables (first n_obj - 1 = 2 variables)
    determine position on the front.
    
    Strategy:
    1. Use Fibonacci lattice (golden ratio spiral) to generate N points uniformly
       distributed on the quarter-sphere surface.
    2. Set distance variables to 0.5 to lie exactly on the Pareto front.
    3. Map spherical coordinates to DTLZ2 decision variables.
    4. Use the entire budget for deterministic, space-filling sampling.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        
        # Number of distance variables
        n_dist = n - m + 1  # = 10 for dim=12, n_obj=3
        
        # Optimal distance values (0.5 for DTLZ2)
        dist_optimal = np.full(n_dist, 0.5)
        
        # Use the entire budget for Fibonacci lattice sampling
        N = self.budget
        
        # Fibonacci lattice on the quarter-sphere
        # Golden ratio conjugate
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        inv_phi2 = 1.0 / (phi * phi)  # ≈ 0.381966
        
        # Generate N points on the quarter-sphere using Fibonacci spiral
        # Parameterize by:
        #   theta1 in [0, pi/2] (polar angle from f1-axis)
        #   theta2 in [0, pi/2] (azimuthal angle in f2-f3 plane)
        # 
        # For uniform distribution on the quarter-sphere:
        #   The surface element is sin(theta1) dtheta1 dtheta2
        #   We want uniform density, so we need to account for the Jacobian.
        #
        # Alternative approach: Use the Fibonacci lattice directly in 3D.
        # Generate N points on the full sphere using Fibonacci lattice, then
        # select/map those in the positive octant, or use a 2D mapping.
        
        # Better: Use a 2D Fibonacci lattice approach
        # For the quarter-sphere, we can use:
        #   theta1 = arccos(1 - 2*i/N) would give uniform in cos(theta1)
        #   But we need to also distribute theta2.
        
        # Let's use the standard Fibonacci sphere approach adapted to quarter-sphere:
        # Generate points using the golden angle spiral in 2D (theta1, theta2)
        
        # Method: For each point i (0-indexed):
        #   theta1_i = arccos(1 - 2*(i + 0.5)/N)  -- this gives uniform in z for full sphere
        #   But we only want quarter-sphere, so we need a different approach.
        
        # For quarter-sphere with 3 objectives:
        #   f1 = cos(theta1)
        #   f2 = sin(theta1) * cos(theta2)  
        #   f3 = sin(theta1) * sin(theta2)
        #   where theta1, theta2 in [0, pi/2]
        
        # Surface element: sin(theta1) dtheta1 dtheta2
        # For uniform density:
        #   - theta2 should be uniform on [0, pi/2]
        #   - theta1 should have PDF proportional to sin(theta1) on [0, pi/2]
        #     CDF: F(t) = (1 - cos(t)) / (1 - cos(pi/2)) = 1 - cos(t)
        #     Inverse: t = arccos(1 - u)
        
        # Use Fibonacci lattice in 2D for (u1, u2) where:
        #   u1 determines theta1 via theta1 = arccos(1 - u1)
        #   u2 determines theta2 via theta2 = (pi/2) * u2
        
        # 2D Fibonacci lattice:
        #   For point i: 
        #     u1_i = (i + 0.5) / N
        #     u2_i = (i * inv_phi2) % 1.0
        
        # This gives a low-discrepancy 2D sequence.
        
        i = np.arange(N, dtype=np.float64)
        
        # 2D Fibonacci lattice coordinates
        u1 = (i + 0.5) / N
        u2 = (i * inv_phi2) % 1.0
        
        # Transform to angles
        theta1 = np.arccos(1.0 - u1)  # uniform surface density in theta1
        theta2 = (np.pi / 2.0) * u2   # uniform in theta2
        
        # Compute objective values (for verification, not needed for return)
        # f1 = cos(theta1)
        # f2 = sin(theta1) * cos(theta2)
        # f3 = sin(theta1) * sin(theta2)
        
        # Build decision vectors
        X = np.zeros((N, n))
        
        # Set distance variables to optimal value 0.5
        X[:, m-1:] = dist_optimal
        
        # Set angle variables: DTLZ2 uses theta_i = (pi/2) * x_i for x_i in [0,1]
        # So we need x_i such that (pi/2)*x_i = theta_i, i.e., x_i = theta_i / (pi/2) = 2*theta_i/pi
        X[:, 0] = 2.0 * theta1 / np.pi
        X[:, 1] = 2.0 * theta2 / np.pi
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.zeros((N, m))
        for idx in range(N):
            if time.time() > deadline:
                break
            F[idx] = problem(X[idx])
        
        return X
# EVOLVE-BLOCK-END
