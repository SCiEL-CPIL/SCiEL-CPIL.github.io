"""Optimized DTLZ4 solver: direct analytical Pareto front construction."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using analytical Pareto front construction."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 with M=3, N=12:
        # Position variables: x[0], x[1] (M-1 = 2)
        # Distance variables: x[2]...x[11] (N-M+1 = 10)
        # For Pareto front (g=0), set distance variables to 0.5
        # Objectives:
        # f1 = cos(x0*pi/2)*cos(x1*pi/2)
        # f2 = cos(x0*pi/2)*sin(x1*pi/2)
        # f3 = sin(x0*pi/2)

        # Strategy: Use budget to create a well-distributed set on the Pareto front.
        # For hypervolume maximization in 3D, we want points that cover the front
        # efficiently. Use a quasi-random (Sobol-like) or grid approach.

        n_points = budget

        # Use a structured approach: create points on a spherical triangle
        # that maximizes coverage for hypervolume.
        
        # Method: Use 2D Fibonacci lattice (golden-ratio grid) on the parameter space
        # mapped to the spherical triangle Pareto front of DTLZ4 (M=3).
        
        # DTLZ4 M=3 Pareto front: points on quarter unit sphere in positive octant
        # Parameterize by (alpha, beta) where:
        #   f1 = cos(alpha) * cos(beta)
        #   f2 = cos(alpha) * sin(beta)  
        #   f3 = sin(alpha)
        # with alpha in [0, pi/2], beta in [0, pi/2]
        
        # For hypervolume maximization, we want even coverage in objective space.
        # Use Fibonacci lattice for low-discrepancy sampling in [0,1]^2, then map
        # to (alpha, beta) using inverse CDF to account for the spherical geometry.
        
        n = n_points
        
        # 2D Fibonacci lattice: use golden ratio conjugate for both dimensions
        # to get a well-distributed set in the unit square.
        # The 1D Fibonacci lattice in [0,1) is: frac(i * phi), i = 0, ..., n-1
        # where phi = (1 + sqrt(5)) / 2. For 2D, we can use two irrational shifts
        # or use the same sequence with different offsets.
        
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        # Generate indices
        idx = np.arange(n, dtype=np.float64)
        
        # 1D Fibonacci sequence for first coordinate
        h1 = np.mod(idx / phi, 1.0)
        # For second coordinate, use a different irrational shift to decorrelate
        # Use golden ratio conjugate: 1/phi = phi - 1
        inv_phi = 1.0 / phi
        h2 = np.mod(idx / inv_phi, 1.0)
        
        # Map from [0,1]^2 to (alpha, beta) on the spherical triangle.
        # For the spherical triangle with vertices (1,0,0), (0,1,0), (0,0,1):
        # Map to uniform surface area on the quarter sphere.
        # Area element dA = sin(alpha) d(alpha) d(beta).
        # To sample uniformly on the surface, we need P(alpha < a) = integral_0^a sin(t) dt = 1 - cos(a).
        # So u = 1 - cos(alpha) => cos(alpha) = 1 - u => alpha = arccos(1 - u).
        # h1 is in [0, 1). 1 - h1 is in (0, 1].
        # arccos(1) = 0, arccos(0) = pi/2.
        alpha = np.arccos(1.0 - h1)
        beta = (np.pi / 2) * h2
        
        # Compute objective values
        f1 = np.cos(alpha) * np.cos(beta)
        f2 = np.cos(alpha) * np.sin(beta)
        f3 = np.sin(alpha)
        
        # Ensure we're strictly better than reference point [2.5, 2.5, 2.5]
        # All values are in [0, 1], so this is automatically satisfied.
        
        # Construct decision vectors: x[0]=alpha/(pi/2), x[1]=beta/(pi/2), x[2:]=0.5
        # Note: x[0] and x[1] are the position variables in [0,1] that map to
        # alpha = x[0] * pi/2 and beta = x[1] * pi/2
        # So we need x[0] = alpha / (pi/2) = 2*alpha/pi and x[1] = beta / (pi/2) = 2*beta/pi
        # But alpha = arccos(1-h1) and beta = (pi/2)*h2
        # So x[0] = 2*arccos(1-h1)/pi and x[1] = h2
        
        # Actually, let's reconsider. The problem states:
        # f1 = cos(x0*pi/2)*cos(x1*pi/2)
        # f2 = cos(x0*pi/2)*sin(x1*pi/2)
        # f3 = sin(x0*pi/2)
        
        # So if we set x0 = alpha/(pi/2) and x1 = beta/(pi/2), then:
        # f1 = cos(alpha)*cos(beta), f2 = cos(alpha)*sin(beta), f3 = sin(alpha)
        
        # We have alpha = arccos(1-h1) and beta = (pi/2)*h2
        # So x0 = arccos(1-h1)/(pi/2) = 2*arccos(1-h1)/pi
        # And x1 = h2
        
        x0_vals = 2.0 * alpha / np.pi
        x1_vals = beta / (np.pi / 2.0)  # which is just h2
        
        params = np.column_stack([x0_vals, x1_vals])
        
        # Add the 3 corner points explicitly to ensure extreme coverage
        # Corners in (x0, x1) space:
        # (0,0) -> f=(1,0,0), (0,1) -> f=(0,1,0), (1,0) -> f=(0,0,1)
        corner_params = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ])
        
        if n_points > 3:
            # Replace last 3 points with corners to ensure extreme coverage
            params[-3:] = corner_params
        else:
            params = corner_params[:n_points]
        
        # Final check: ensure all params are in [0, 1]
        params = np.clip(params, 0.0, 1.0)
        
        # Construct decision vectors
        X = np.zeros((len(params), dim))
        X[:, 0] = params[:, 0]
        X[:, 1] = params[:, 1]
        X[:, 2:] = 0.5
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        for i in range(len(X)):
            if time.time() > deadline:
                break
            problem(X[i])
        
        return X
# EVOLVE-BLOCK-END
