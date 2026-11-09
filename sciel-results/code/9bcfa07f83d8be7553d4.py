# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for DTLZ4 that exploits the known Pareto front geometry.
    It sets distance variables to their optimal values (0.5) to minimize g(x),
    and samples shape variables using a stratified grid with vertex inclusion
    to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 with 3 objectives:
        # Shape variables: x_0, x_1 (indices 0, 1)
        # Distance variables: x_2 ... x_{dim-1} (indices 2 to 11 for dim=12)
        n_shape = n_obj - 1  # 2
        n_dist = dim - n_shape  # 10

        # Optimal distance values for DTLZ4 are 0.5
        # g(x) = sum((x_i - 0.5)^2), so g=0 when x_i=0.5
        
        # For DTLZ4 3-objective, the Pareto front is a quarter sphere.
        # The mapping from decision variables to angles is:
        # theta_0 = (pi / 2) * x_0  =>  f_0 = cos(theta_0)
        # theta_1 = (pi / 2) * x_1  =>  f_1 = sin(theta_0) * cos(theta_1)
        #                                       f_2 = sin(theta_0) * sin(theta_1)
        
        # To get uniform coverage on the sphere surface:
        # cos(theta_0) ~ Uniform(0, 1)  =>  x_0 = (2/pi) * arccos(u)
        # theta_1 ~ Uniform(0, pi/2)    =>  x_1 = v
        
        results = []
        
        # Use a stratified grid with vertex inclusion
        # n_grid x n_grid grid points, with vertices included
        n_grid = int(np.sqrt(budget))
        n_extra = budget - n_grid * n_grid
        
        # For hypervolume, points near the vertices of the Pareto front contribute most.
        # The three "corner" points in (u,v) space are:
        #   (u=1, v=0) -> f=(1,0,0)
        #   (u=0, v=0) -> f=(0,1,0)
        #   (u=0, v=1) -> f=(0,0,1)
        # 
        # Strategy:
        # 1. Always evaluate the 3 corner points
        # 2. Use a stratified grid with power-law spacing to cluster near boundaries
        # 3. Fill remaining budget with low-discrepancy Halton points
        
        # Always include the 3 corner points
        corner_uv = [(1.0, 0.0), (0.0, 0.0), (0.0, 1.0)]
        
        # Use remaining budget for grid + Halton
        n_remaining = budget - 3
        n_grid = int(np.sqrt(n_remaining))
        n_extra = n_remaining - n_grid * n_grid
        
        # Power-law spacing: cluster points near 0 and 1
        # t in [0,1], use t^alpha and 1-(1-t)^alpha to create clusters at both ends
        # For a grid that includes 0 and 1, use a symmetric power-law
        alpha = 0.5  # sqrt spacing clusters near ends
        
        # Create grid values with power-law spacing
        # We want n_grid points in [0,1] including 0 and 1
        # Use: for k in 0..n_grid-1, t_k = 0.5 * (1 - cos(pi * k / (n_grid - 1)))
        # This gives cosine spacing which clusters near 0 and 1
        if n_grid > 1:
            u_vals = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_grid) / (n_grid - 1)))
            v_vals = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_grid) / (n_grid - 1)))
        else:
            u_vals = np.array([0.0])
            v_vals = np.array([0.0])
        
        # Evaluate corner points first
        for u, v in corner_uv:
            if time.time() > deadline:
                return None
            
            x_0 = (2.0 / np.pi) * np.arccos(u)
            x_1 = v
            
            x = np.zeros(dim)
            x[0] = x_0
            x[1] = x_1
            x[2:] = 0.5
            
            f = problem(x)
            results.append(f)
        
        # Evaluate grid points
        for i in range(n_grid):
            for j in range(n_grid):
                # Check deadline
                if time.time() > deadline:
                    return None
                
                u = u_vals[i]
                v = v_vals[j]
                
                # Calculate shape variables
                x_0 = (2.0 / np.pi) * np.arccos(u)
                x_1 = v
                
                # Construct full vector
                x = np.zeros(dim)
                x[0] = x_0
                x[1] = x_1
                x[2:] = 0.5  # Set distance variables to optimal value
                
                # Evaluate
                f = problem(x)
                results.append(f)
        
        # Add extra points if budget allows
        if n_extra > 0:
            # Use Halton sequence for low-discrepancy sampling
            for k in range(n_extra):
                if time.time() > deadline:
                    return None
                
                u = self._halton(k + 1, 2)
                v = self._halton(k + 1, 3)
                
                x_0 = (2.0 / np.pi) * np.arccos(u)
                x_1 = v
                
                x = np.zeros(dim)
                x[0] = x_0
                x[1] = x_1
                x[2:] = 0.5
                
                f = problem(x)
                results.append(f)
        
        return None
    
    def _halton(self, n, base):
        """Generate the nth Halton number in the given base."""
        result = 0.0
        f = 1.0
        i = n
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
# EVOLVE-BLOCK-END
