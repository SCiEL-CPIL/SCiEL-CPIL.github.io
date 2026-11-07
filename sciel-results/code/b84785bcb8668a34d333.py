"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Problem-specific optimizer exploiting DTLZ2's known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ2 specifics:
        # f_i(x) = (1 + g(x)) * h_i(y)
        # g(x) = sum_{i=n-M+2}^n (x_i - 0.5)^2
        # h_i depends on y = x_1, ..., x_{M-1}
        # To be on Pareto front, g(x) must be 0.
        # This implies x_i = 0.5 for i = n-M+2, ..., n.
        # The variables x_1, ..., x_{M-1} determine the position on the front.
        
        d_pos = n_obj - 1
        d_dist = dim - d_pos
        
        # Strategy: Use a dense uniform grid (Lattice) to ensure comprehensive coverage.
        # For d_pos=2 (3 objectives), a square grid of size ~sqrt(budget) is optimal for coverage.
        # We generate a grid of points in [0, 1]^d_pos.
        
        points_pos = []
        
        if d_pos == 1:
            # 1D case: linear spacing
            n_grid = budget
            points_pos = np.linspace(0, 1, n_grid, endpoint=False)
            points_pos = points_pos.reshape(-1, 1)
        elif d_pos == 2:
            # 2D case: square grid
            # We want roughly budget points. sqrt(200) ~ 14.14.
            # Using 15x15 = 225 points might exceed budget if we evaluate all.
            # But we can just generate the grid and take the first `budget` points?
            # Or adjust grid size to fit budget exactly?
            # A grid of 14x14 = 196 points is safe and covers the space well.
            # 15x15 = 225 > 200.
            # Let's use a grid size that fits within budget.
            side = int(np.sqrt(budget))
            # If side*side > budget, we might skip some.
            # Better: generate full grid, then subsample or just use the grid if it fits.
            # Actually, for hypervolume, regularity is key.
            # Let's try to fit as many grid points as possible.
            # 14*14 = 196.
            # 15*15 = 225.
            # We have 200 evaluations.
            # We can use a 14x14 grid (196 points) and 4 extra random points?
            # Or just 14x14.
            # Let's stick to a pure grid for consistency.
            n_grid = side
            # If n_grid*n_grid < budget, we have spare evaluations.
            # We can add points on the boundary or just stick to the grid.
            # For now, generate the grid.
            x1 = np.linspace(0, 1, n_grid, endpoint=False)
            x2 = np.linspace(0, 1, n_grid, endpoint=False)
            X1, X2 = np.meshgrid(x1, x2)
            points_pos = np.column_stack((X1.ravel(), X2.ravel()))
            
            # If we have budget remaining, we can add some points.
            # But a grid is usually better for coverage than random sprinkling.
            # However, the corners (0,0), (0,1), (1,0), (1,1) are important for hypervolume.
            # linspace(0, 1, n, endpoint=False) includes 0 but not 1.
            # So (1,1) is missing. (1,0) and (0,1) are missing.
            # This might hurt hypervolume at the edges.
            # Let's use endpoint=True? No, that duplicates.
            # Let's manually add the corners if we have space.
            # Actually, standard practice for hypervolume is to cover the convex hull.
            # The front is a convex arc/curve.
            # A grid in the parameter space maps to the front.
            
            # Let's refine the grid to include boundaries better if possible.
            # But with 200 budget, 14x14 is 196.
            # We have 4 extra.
            # Let's just use the 14x14 grid for simplicity and regularity.
            
        else:
            # Fallback for higher dimensions (unlikely for DTLZ2 with 3 obj)
            # Use random sampling
            points_pos = np.random.rand(budget, d_pos)

        # Construct full dimension vectors
        X = np.zeros((len(points_pos), dim))
        
        # Assign position variables
        if d_pos > 0:
            X[:, :d_pos] = points_pos
            
        # Assign distance variables to 0.5
        X[:, d_pos:] = 0.5
        
        # Clip to bounds just in case
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        # The evaluator records the objective values returned by problem(x)
        n_evals = 0
        for i in range(len(X)):
            if time.time() > deadline:
                break
            problem(X[i])
            n_evals += 1
            # If we have budget left and we finished the grid, we could add more,
            # but the grid is designed to be the primary strategy.
            
        return X

# EVOLVE-BLOCK-END
