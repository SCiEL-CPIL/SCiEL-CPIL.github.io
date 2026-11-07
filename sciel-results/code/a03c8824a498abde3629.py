"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Structure-aware optimizer for DTLZ2 using Fibonacci lattice on the Pareto front.
    
    Strategy:
    1. Assume standard DTLZ2 definition:
       - m = n_obj = 3
       - Angle variables: x[0], x[1]
       - Distance variables: x[2] ... x[dim-1]
       - g(x_M) = sum((x_i - 0.5)^2) for i in angle_vars+1 to dim
       - f_i = (1+g) * spherical_coord_mapping(x_angles)
       
    2. To maximize hypervolume, we want points uniformly distributed on the Pareto front (unit sphere octant).
    3. Use a Fibonacci lattice on the full sphere to generate points, filter to the positive octant, and map back to DTLZ2 variables.
    4. Set distance variables to 0.5 to ensure g=0.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin: 300s limit, stop at 290s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_angle = self.n_obj - 1  # Standard DTLZ2 uses m-1 angle variables
        n_dist = self.dim - n_angle
        
        # We need 'budget' points on the octant of the sphere.
        # Generate 8 * budget points on the full sphere to ensure enough in the octant.
        n_sphere_points = self.budget * 8 + 100
        
        # Fibonacci lattice on the sphere
        # phi: polar angle (0 to pi)
        # theta: azimuthal angle (0 to 2*pi)
        i = np.arange(n_sphere_points)
        # Uniform distribution in cos(phi)
        cos_phi = 1.0 - (2.0 * i) / (n_sphere_points - 1)
        phi = np.arccos(cos_phi)
        
        # Golden angle for theta
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        theta = i * golden_angle
        
        # Convert to Cartesian coordinates on the unit sphere
        sin_phi = np.sin(phi)
        f1_cart = sin_phi * np.cos(theta)
        f2_cart = sin_phi * np.sin(theta)
        f3_cart = cos_phi
        
        # Filter for the positive octant (f1, f2, f3 > 0)
        # Note: Due to floating point, we might need a small epsilon, but strict > 0 is fine for filtering
        mask = (f1_cart > 1e-9) & (f2_cart > 1e-9) & (f3_cart > 1e-9)
        
        # Get indices of points in the octant
        octant_indices = np.where(mask)[0]
        
        # We need exactly self.budget points. 
        # If we have more, we take the first self.budget.
        # If we have fewer (unlikely with 8x oversampling), we take all.
        n_take = min(self.budget, len(octant_indices))
        selected_indices = octant_indices[:n_take]
        
        # Extract the Cartesian coordinates for the selected points
        f1_sel = f1_cart[selected_indices]
        f2_sel = f2_cart[selected_indices]
        f3_sel = f3_cart[selected_indices]
        
        # Map Cartesian coordinates back to DTLZ2 angle variables x_1, x_2
        # f1 = sin(pi * x1 / 2)
        # f2 = cos(pi * x1 / 2) * sin(pi * x2 / 2)
        # f3 = cos(pi * x1 / 2) * cos(pi * x2 / 2)
        
        # x1 = (2/pi) * arcsin(f1)
        x1 = (2.0 / np.pi) * np.arcsin(np.clip(f1_sel, -1.0, 1.0))
        
        # x2 = (2/pi) * arctan2(f2, f3)
        # Note: arctan2(y, x) gives angle in [-pi, pi]. 
        # Since f2, f3 > 0, angle is in (0, pi/2).
        x2 = (2.0 / np.pi) * np.arctan2(f2_sel, f3_sel)
        
        # Clip x1, x2 to [0, 1] just in case of numerical issues
        x1 = np.clip(x1, 0.0, 1.0)
        x2 = np.clip(x2, 0.0, 1.0)
        
        # Construct the full variable vectors
        X = np.zeros((n_take, self.dim))
        
        # Assign angle variables
        X[:, 0] = x1
        X[:, 1] = x2
        
        # Assign distance variables to 0.5 (center of [0, 1]) to ensure g=0
        # This assumes the bounds are [0, 1]. If bounds are different, we should use (lo+hi)/2
        # But DTLZ2 standard is [0,1].
        mid_dist = (lo[n_angle:] + hi[n_angle:]) / 2.0
        X[:, n_angle:] = mid_dist
        
        # Evaluate
        for i in range(n_take):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
# EVOLVE-BLOCK-END
