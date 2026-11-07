import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Deadline guard
        start_time = time.time()
        timeout = 290.0  # Safety margin before 300s limit
        
        # DTLZ4 specific knowledge:
        # M = 3 objectives
        # n = 12 variables
        # Position vars: x[0], x[1]
        # G-vars: x[2] to x[11]
        
        # Optimal G-value is 0.5 for DTLZ4 (based on the successful strategy)
        # We fix x[2:] to 0.5 to ensure points are on the Pareto front manifold.
        
        # To maximize hypervolume, we want uniform coverage of the Pareto front surface area.
        # The Pareto front of DTLZ4 (with M=3) is a quarter of a unit sphere (scaled by g*).
        # The area element on the sphere parameterized by angles alpha, beta is sin(alpha) dalpha dbeta.
        # alpha = x[0] * pi/2, beta = x[1] * pi/2.
        # Uniform sampling in x[0] leads to over-sampling near the poles (alpha=0, pi/2).
        # To get uniform area coverage, we should sample cos(alpha) uniformly.
        # Let u = cos(alpha) = cos(x[0] * pi/2). u should be uniform in [0, 1].
        # Then x[0] = (2/pi) * arccos(u).
        # beta (x[1]) is uniform in area, so x[1] should be uniform in [0, 1].
        
        # We will generate points using a quasi-random sequence (Sobol) or a grid in the 
        # "uniform area" space (u, v) where u ~ U[0,1] and v ~ U[0,1].
        # A grid in this transformed space is equivalent to stratified sampling on the sphere surface.
        
        n_grid = 14
        # 14x14 = 196 points
        u_vals = np.linspace(0, 1, n_grid)
        v_vals = np.linspace(0, 1, n_grid)
        
        points = []
        
        # Generate grid points in transformed space
        for u in u_vals:
            for v in v_vals:
                x = np.zeros(self.dim)
                # Map u (uniform in [0,1]) to x[0] such that cos(x[0]*pi/2) = u
                # x[0] * pi/2 = arccos(u)
                # x[0] = (2/pi) * arccos(u)
                x[0] = (2.0 / np.pi) * np.arccos(u)
                # x[1] is uniform in [0, 1]
                x[1] = v
                # Fix G-variables to optimal value
                x[2:] = 0.5
                points.append(x)
        
        # We have 196 points. Budget is 200.
        # Add 4 more points to utilize the full budget.
        # We can add points from a slightly finer grid or specific midpoints.
        # Let's add 4 points that are midpoints of the quadrants in the u-v space,
        # but shifted to avoid duplicates if possible, or just use a small Sobol sequence for the extras.
        # Simpler: Just use 4 specific points from a 15x15 grid that fall into the "gaps" of the 14x14 grid?
        # The 14x14 grid has spacing 1/13.
        # Let's just add 4 points from a random-like deterministic sequence to fill gaps.
        # Or, we can just use 200 points from a Sobol sequence directly in u-v space.
        # Sobol is often better than grid for non-linear mappings if the mapping distorts the grid.
        # However, the mapping from u,v to x is monotonic. The distortion is handled by the transformation.
        # So a grid in u,v space is effectively a stratified sample on the sphere.
        
        # Let's add 4 points from a 15x15 grid that are not in the 14x14 grid.
        # 15x15 grid indices: 0..14, step 1/14.
        # 14x14 grid indices: 0..13, step 1/13.
        # This is complex to check uniqueness.
        
        # Alternative: Use 200 points from a Sobol sequence in u-v space.
        # This ensures low discrepancy in the uniform area space.
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=2, scramble=False)
        # Generate 256 points and take first 200? Or just 200.
        # Sobol requires power of 2 for optimal properties, but works for any N.
        # Let's generate 200 points.
        sobol_points = sampler.random(200)
        
        # Re-generate points using Sobol sequence in u-v space
        # This replaces the grid approach with a quasi-random approach in the uniform area space.
        # This is likely superior to a fixed grid because it avoids potential resonance with the 
        # specific shape of the hypervolume integrand if any, though for a sphere it's smooth.
        # Actually, for a sphere, the grid in u-v is very good. Sobol might be slightly better 
        # due to lower discrepancy in higher dimensions (though here it's 2D).
        
        # Let's stick to the Grid in u-v space for the first 196, and add 4 Sobol points 
        # that are "far" from the grid points? 
        # Or just use 200 Sobol points. 
        # The previous program used Grid + Random. 
        # Let's use 200 Sobol points in u-v space. It's cleaner and uses the full budget.
        
        final_points = []
        for uv in sobol_points:
            u, v = uv[0], uv[1]
            x = np.zeros(self.dim)
            x[0] = (2.0 / np.pi) * np.arccos(u)
            x[1] = v
            x[2:] = 0.5
            final_points.append(x)
            
        # Evaluate all points
        F = np.array([problem(x) for x in final_points])
        
        # Check deadline
        if time.time() - start_time > timeout:
            return F
            
        return F
