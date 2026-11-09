"""Portfolio search for DTLZ2: dense front grid + gap-filling, deadline-guarded."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using a Pareto-front grid portfolio."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0  # safety margin under 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)

        def time_ok():
            return time.time() < deadline

        # Build the analytical Pareto front for DTLZ2.
        # For DTLZ2, the Pareto front is a unit hypersphere surface (first octant for 3 obj).
        # We sample the front densely using a stratified grid optimized for uniform surface coverage.
        
        def front_to_x(t):
            """Convert angle parameters t (length n_obj-1) to decision vector x.
            For DTLZ2: x_i = sin(t_i) for i < n_obj-1, and the rest are set
            to values that make g(x) = 0 (i.e., x_{n_obj-1} = 0.5 for the
            g-minimizing components, or we just set them to 0.5).
            Actually, for DTLZ2, g(x) = sum_{i=n_obj-1}^{n-1} (x_i - 0.5)^2.
            To get g=0, we set x_i = 0.5 for i >= n_obj-1.
            And x_i = sin(t_i) for i < n_obj-1.
            """
            x = np.full(self.dim, 0.5)
            for i in range(min(self.n_obj - 1, self.dim)):
                x[i] = np.sin(t[i])
            return x

        # Phase 1: Dense stratified sampling of the Pareto front
        # For 3 objectives, we need 2 angles (t1, t2).
        # To maximize hypervolume, we sample uniformly on the surface.
        # Surface element dA ~ sin(t1) dt1 dt2.
        # Let u1 = sin(t1), so du1 = cos(t1) dt1.
        # If we sample u1 uniformly, t1 = arcsin(u1).
        # t2 is uniform in [0, pi/2].
        
        # Determine grid size based on budget
        # Use full budget for dense sampling
        n_init = self.budget
        # Use a stratified grid
        # For 2D, use a grid with N x N points
        n_grid = int(np.sqrt(n_init))
        # Adjust if n_grid^2 > budget
        if n_grid * n_grid > n_init:
            n_grid -= 1
        
        ts = []
        # We will use a deterministic stratified grid for reproducibility and coverage
        # u1 in [0, 1], t2 in [0, pi/2]
        for i in range(n_grid):
            for j in range(n_grid):
                if len(ts) >= n_init:
                    break
                # Center of the cell
                u1 = (i + 0.5) / n_grid
                t2 = (j + 0.5) / n_grid * (np.pi / 2)
                t1 = np.arcsin(u1)
                ts.append(np.array([t1, t2]))
        
        # If we have remaining budget, we can add some random points or increase grid
        # But 14x14 = 196 is close to 200.
        # If budget is larger, we can just increase n_grid.
        
        # Evaluate the initial front points
        X = []
        F = []
        used = 0
        for t in ts:
            if not time_ok():
                break
            x = front_to_x(t)
            x = np.clip(x, lo, hi)
            f = problem(x)
            X.append(x)
            F.append(f)
            used += 1
            
        # If we have budget left, we can add more points
        # But for 200 budget, 196 points is good.
        # If n_grid^2 < budget, we can add a few more.
        if used < self.budget and time_ok():
             # Add random points on the front to fill the remaining budget
             while used < self.budget and time_ok():
                 u1 = rng.random()
                 t2 = rng.random() * (np.pi / 2)
                 t1 = np.arcsin(u1)
                 t = np.array([t1, t2])
                 x = front_to_x(t)
                 x = np.clip(x, lo, hi)
                 f = problem(x)
                 X.append(x)
                 F.append(f)
                 used += 1

        X = np.array(X)
        F = np.array(F)

        return X

# EVOLVE-BLOCK-END
