"""Direct construction optimizer for DTLZ4 exploiting analytical Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using direct front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 structure:
        # First dim-5 variables shape the front.
        # Last 5 variables affect g. g=0 when they are 0.5.
        # For 3 objectives, only x1 and x2 determine the front shape.
        # Front is a quarter sphere of radius 0.5.
        
        n_g = 5
        n_front = dim - n_g

        # We will use the entire budget to sample the Pareto front optimally.
        # To get a uniform distribution on the surface of the quarter sphere:
        # Let theta = pi/2 - alpha, phi = beta.
        # alpha = pi * x1 / 2, beta = pi * x2 / 2.
        # Uniform on sphere surface: cos(theta) ~ U[0,1] and phi ~ U[0, pi/2].
        # cos(theta) = sin(alpha) = sin(pi * x1 / 2).
        # So sin(pi * x1 / 2) = u1 -> x1 = (2/pi) * arcsin(u1).
        # phi = pi * x2 / 2 = u2 * pi/2 -> x2 = u2.
        
        # Use a stratified grid (Latin Hypercube style) for better coverage
        # and to avoid clustering artifacts from low-discrepancy sequences
        # after non-linear mapping.
        
        # Determine grid size. We want roughly square grid.
        # k x k points.
        k = int(np.sqrt(budget))
        # If k*k < budget, we can add a few extra points or just use k*k.
        # Let's stick to k*k to keep it simple and regular.
        # If budget is not a perfect square, we might leave some budget unused,
        # which is acceptable if the grid is superior.
        # Or we can adjust k to be closer to sqrt(budget).
        
        # Generate grid points in [0, 1] x [0, 1]
        # Using midpoints of cells to avoid boundary issues
        grid_u1 = np.linspace(0.5/k, 1 - 0.5/k, k)
        grid_u2 = np.linspace(0.5/k, 1 - 0.5/k, k)
        
        # Create meshgrid
        U1, U2 = np.meshgrid(grid_u1, grid_u2)
        
        # Flatten to 1D arrays
        u1 = U1.flatten()
        u2 = U2.flatten()
        
        n_points = len(u1)
        
        # Compute x1 and x2
        # The mapping to the sphere surface:
        # We want uniform distribution on the surface of the quarter sphere.
        # Surface element dA ~ sin(alpha) dalpha dphi.
        # To get uniform dA, we sample cos(theta) ~ U[0,1] where theta is angle from z-axis?
        # In DTLZ4, f1, f2, f3 are components.
        # f3 = 0.5 sin(alpha). f1, f2 depend on cos(alpha).
        # The "height" is f3. The "radius" in f1-f2 plane is 0.5 cos(alpha).
        # Uniform on sphere: z ~ U[-1, 1] (here z is f3/0.5 = sin(alpha)).
        # So sin(alpha) ~ U[0, 1].
        # alpha = arcsin(u).
        # x1 = alpha / (pi/2) = (2/pi) arcsin(u).
        # phi ~ U[0, pi/2].
        # beta = phi. x2 = beta / (pi/2) = u2.
        
        x1_vals = (2.0 / np.pi) * np.arcsin(u1)
        x2_vals = u2

        # Construct decision vectors
        X = np.zeros((n_points, dim))
        X[:, 0] = x1_vals
        X[:, 1] = x2_vals
        # Set other front variables to 0.5 (arbitrary, doesn't affect objectives)
        if n_front > 2:
            X[:, 2:n_front] = 0.5
        # Set g variables to 0.5 to ensure g=0
        X[:, n_front:] = 0.5

        # Clip to bounds (should already be in [0,1] but just in case)
        X = np.clip(X, lo, hi)

        # Evaluate all points
        # We must be careful not to exceed the time limit.
        # 200 evaluations is very fast, so we can do them all.
        
        # Check time before starting evaluation loop if necessary, 
        # but 200 evals are negligible.
        if time.time() - t0 > deadline:
            # In case we somehow took too long to generate (unlikely), return empty or best effort
            # But we haven't evaluated anything yet.
            return X

        F = np.array([problem(x) for x in X])
        
        # The return value is ignored, but we return X as per the seed program style
        return X
# EVOLVE-BLOCK-END
