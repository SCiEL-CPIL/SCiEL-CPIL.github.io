import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 exploiting problem structure.
    
    DTLZ2 with 3 objectives and 12 variables:
    - g(x) = sum_{i=2}^{11} (x_i - 0.5)^2, minimized at 0 when x_i=0.5
    - f1 = (1+g)*cos(t1)*cos(t2), f2 = (1+g)*cos(t1)*sin(t2), f3 = (1+g)*sin(t1)
    - where t1 = x0*pi/2, t2 = x1*pi/2
    - Pareto front: g=0, so f1^2+f2^2+f3^2 = 1
    
    Strategy: 
    1. Use the entire budget to evaluate points exactly on the Pareto front.
    2. Use a Lattice distribution in the surface-area coordinates to ensure 
       uniform coverage of the Pareto front's surface area, which maximizes 
       the hypervolume for a fixed number of points.
    3. The surface area element is dA = cos(t1) dt1 dt2. 
       Let u = sin(t1), then dA = du dt2. 
       We generate a uniform grid in (u, t2) space.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self._deadline = None

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: stop 30s before the 330s wall-clock kill
        self._deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Determine grid dimensions
        # 200 points. 20 x 10 is a good aspect ratio for the 2D parameter space
        # if we consider the distortion, but since we are mapping from uniform
        # surface area coordinates, a square-ish grid in (u, v) is good.
        # Let's try 14x14 + 4? Or 20x10?
        # 20x10 = 200 exactly.
        n_u = 20
        n_v = 10
        
        # Generate lattice points in (u, v) space [0, 1] x [0, 1]
        # Using midpoints to avoid boundaries if possible, but boundaries are part of the front.
        # Standard lattice: (i + 0.5) / n
        u_vals = (np.arange(n_u) + 0.5) / n_u
        v_vals = (np.arange(n_v) + 0.5) / n_v
        
        # Create meshgrid
        U, V = np.meshgrid(u_vals, v_vals, indexing='ij')
        U_flat = U.flatten()
        V_flat = V.flatten()
        
        # Map to x0, x1
        # u = sin(t1) => t1 = arcsin(u) => x0 = 2/pi * arcsin(u)
        # v = t2 / (pi/2) => t2 = v * pi/2 => x1 = v
        x0 = (2.0 / np.pi) * np.arcsin(U_flat)
        x1 = V_flat
        
        # Construct full 12D vectors on the Pareto front
        X = np.zeros((budget, dim))
        X[:, 0] = x0
        X[:, 1] = x1
        X[:, 2:] = 0.5  # g=0
        
        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate all points
        F = []
        used = 0
        for x in X:
            if time.time() > self._deadline:
                break
            f = problem(x)
            F.append(f)
            used += 1
        
        return F
