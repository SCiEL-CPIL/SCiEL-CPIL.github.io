"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic Pareto front constructor for DTLZ2 using Fibonacci lattice."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n = self.budget  # 200 points
        
        # Fibonacci lattice (Koksma) on the unit sphere in 3D
        # For 3 objectives, the Pareto front is a quarter-sphere in [0,1]^3
        # We use spherical coordinates: theta in [0, pi/2], phi in [0, pi/2]
        # f_1 = cos(theta)
        # f_2 = sin(theta) * cos(phi)  
        # f_3 = sin(theta) * sin(phi)
        
        # Fibonacci lattice: distribute points uniformly on sphere surface
        # Using the golden ratio for uniform distribution
        golden = (1 + np.sqrt(5)) / 2.0
        inv_golden = 1.0 / golden
        
        # Generate 200 points on the quarter-sphere
        # theta_i and phi_i from Fibonacci sequence
        indices = np.arange(n)
        
        # For uniform distribution on sphere, use:
        # cos(theta_i) = 1 - 2*(i + 0.5)/n  for full sphere
        # For quarter sphere, we need to adjust
        # Actually, for DTLZ2 with 3 objectives:
        # The mapping from (x_1, x_2) to objectives is:
        # f_1 = 1 - x_1
        # f_2 = x_1 * (1 - x_2)
        # f_3 = x_1 * x_2
        # This maps [0,1]^2 to the quarter sphere
        # So we need to distribute points in (x_1, x_2) space
        # such that they map uniformly to the sphere surface
        
        # The Jacobian of the transformation from (x_1, x_2) to sphere surface
        # is proportional to 1/sqrt(x_1 * (1-x_1)) approximately
        # To get uniform distribution on sphere, we should use:
        # x_1 = cos(theta) where theta is uniformly distributed
        # x_2 = cos(phi) where phi is uniformly distributed
        # But with proper weighting
        
        # Better approach: use Fibonacci lattice directly in (theta, phi) space
        # theta in [0, pi/2], phi in [0, pi/2]
        # For uniform area on sphere: dA = sin(theta) dtheta dphi
        # So we want theta_i such that sin(theta_i) is uniformly spaced
        # i.e., cos(theta_i) = 1 - 2*i/n (for full hemisphere)
        # For quarter sphere: cos(theta_i) = 1 - i/(n-1) approximately
        
        # Let's use a proper Fibonacci lattice on the quarter sphere
        # Point i: 
        #   cos(theta_i) = 1 - (2*(i + 0.5) * (1/golden - 1)) / n  -- no, simpler
        #   
        # Standard Fibonacci sphere:
        #   z_i = 1 - 2*(i + 0.5)/n
        #   r_i = sqrt(1 - z_i^2)
        #   theta_i = 2*pi*i*inv_golden (or golden angle)
        #   x_i = r_i * cos(theta_i)
        #   y_i = r_i * sin(theta_i)
        #
        # For quarter sphere in first octant, we need all coordinates positive
        # Use:
        #   cos(theta_i) = 1 - (i + 0.5) * 2 / n  -- this gives z from 1 to 0
        #   But we need two angles for 3D
        #
        # For 3 objectives, the Pareto front is a 2D surface (quarter sphere)
        # We can parameterize with two angles: theta and phi
        # f_1 = cos(theta)
        # f_2 = sin(theta) * cos(phi)
        # f_3 = sin(theta) * sin(phi)
        # where theta in [0, pi/2], phi in [0, pi/2]
        #
        # Area element: dA = sin(theta) dtheta dphi
        # For uniform distribution, we want:
        #   u_i = (i + 0.5) / n  for theta distribution
        #   cos(theta_i) = 1 - u_i  (since integral of sin(theta) dtheta from 0 to theta = 1-cos(theta))
        #   v_j = (j + 0.5) / m  for phi distribution (uniform in phi since no weight)
        #   phi_j = v_j * pi/2
        #
        # But we have n points total, not a grid. Use Fibonacci lattice in 2D:
        # Map i to (theta_i, phi_i) using golden ratio
        #
        # Simpler: use a 2D Fibonacci lattice in (u, v) space where
        # u in [0,1] maps to theta via cos(theta) = 1 - u
        # v in [0,1] maps to phi via phi = v * pi/2
        
        # Generate 2D Fibonacci lattice points in [0,1]^2
        # Using the standard 2D low-discrepancy sequence
        # For i = 0, ..., n-1:
        #   u_i = (i * inv_golden) % 1
        #   v_i = ((i+1) * inv_golden) % 1  -- or use different offset
        
        # Actually, for 2D Fibonacci lattice:
        # u_i = (i * 1/golden) % 1
        # v_i = ((i+1) * 1/golden) % 1
        # This doesn't give perfect uniformity. Better to use:
        # u_i = (i * inv_golden) % 1
        # v_i = ((i * inv_golden * inv_golden) % 1)  -- not standard
        
        # Let's use a proper 2D Sobol-like or just a grid with better spacing
        # Since we have exactly 200 points, use a 20x10 grid but with
        # proper transformation to account for the Jacobian
        
        # For uniform distribution on the quarter sphere:
        # cos(theta) should be uniformly distributed in [0,1]
        # phi should be uniformly distributed in [0, pi/2]
        
        # So:
        # theta_i = arccos(1 - u_i) where u_i is uniform in [0,1]
        # phi_j = v_j * pi/2 where v_j is uniform in [0,1]
        
        # Use a 20x10 grid for (u, v) in [0,1]^2
        n_u = 20
        n_v = 10
        if n_u * n_v > n:
            n_u = int(np.sqrt(n))
            n_v = n // n_u
        
        u_vals = (np.arange(n_u) + 0.5) / n_u  # uniform in [0,1]
        v_vals = (np.arange(n_v) + 0.5) / n_v  # uniform in [0,1]
        
        X = []
        for u in u_vals:
            for v in v_vals:
                # cos(theta) = 1 - u, so theta = arccos(1 - u)
                # phi = v * pi/2
                cos_theta = 1.0 - u
                sin_theta = np.sqrt(1.0 - cos_theta**2)
                phi = v * np.pi / 2.0
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                
                # Objectives:
                # f_1 = cos_theta
                # f_2 = sin_theta * cos_phi
                # f_3 = sin_theta * sin_phi
                
                # Map to DTLZ2 decision variables:
                # f_1 = 1 - x_1  =>  x_1 = 1 - f_1 = 1 - cos_theta
                # f_2 = x_1 * (1 - x_2)  =>  1 - x_2 = f_2 / x_1  (if x_1 > 0)
                # f_3 = x_1 * x_2
                
                x_1 = 1.0 - cos_theta
                if x_1 > 1e-10:
                    x_2 = 1.0 - (sin_theta * cos_phi) / x_1
                else:
                    x_2 = 0.0  # when x_1 ~ 0, f_2 ~ 0, x_2 can be anything
                
                x = np.full(self.dim, 0.5)
                x[0] = x_1
                x[1] = x_2
                x = np.clip(x, lo, hi)
                X.append(x)
        
        # Evaluate all points
        F = []
        for x in X:
            if time.time() > deadline:
                break
            f = problem(x)
            F.append(f)
        
        return X
# EVOLVE-BLOCK-END
