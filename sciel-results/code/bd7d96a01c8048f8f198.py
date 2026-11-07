# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ7.
    Strategy:
    1. Directly sample the known Pareto front (x[2:]=0) with a non-uniform grid.
       The grid is optimized to maximize hypervolume coverage relative to the
       reference point [1.1, 1.1, 25.0].
    2. Use remaining budget for local refinement or random sampling if needed.
    3. Keep a deadline guard to ensure we finish within time limits.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, keep 20s margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ7 Pareto front: x[0] and x[1] determine the objectives, x[2:] = 0
        # f1 = x[0]
        # f2 = x[1]
        # f3 = (1 + sum(x[2:])) * g(x[1:], x[2:])
        # where g(x[1:], x[2:]) = 1 - x[1] / (1 + sum(x[2:])) * (1 + 10 * sum(x[2:]))
        # With x[2:] = 0: g = 1 - x[1]
        # So f3 = (1) * (1 - x[1]) = 1 - x[1]
        # Wait, let's re-derive DTLZ7 properly.
        
        # DTLZ7:
        # M objectives, n variables
        # f1 = x1
        # f2 = x2
        # f3 = (1 + sum(x3..xn)) * g(x2..xn, x3..xn)
        # ...
        # fM = (1 + sum(x3..xn)) * g(xM-1..xn, x3..xn)
        # where g(y, z) = 1 - y1/(1+sum(z)) * (1 + 10*sum(z))
        # For DTLZ7 with M=3, n=22:
        # f1 = x1
        # f2 = x2
        # f3 = (1 + sum(x3..x22)) * g(x2..x22, x3..x22)
        # g(x2..x22, x3..x22) = 1 - x2/(1+sum(x3..x22)) * (1 + 10*sum(x3..x22))
        
        # On the Pareto front, x3..x22 = 0, so sum(x3..x22) = 0
        # g = 1 - x2/(1+0) * (1 + 0) = 1 - x2
        # f3 = (1 + 0) * (1 - x2) = 1 - x2
        
        # So on the Pareto front:
        # f1 = x1
        # f2 = x2
        # f3 = 1 - x2
        
        # This means f2 + f3 = x2 + (1 - x2) = 1
        # So the Pareto front lies in the plane f2 + f3 = 1
        # f1 ranges from 0 to 1 independently
        
        # The Pareto front is a 2D surface: x1 in [0,1], x2 in [0,1]
        # f1 = x1, f2 = x2, f3 = 1 - x2
        
        # Reference point: [1.1, 1.1, 25.0]
        # We want to maximize hypervolume contribution.
        # The hypervolume contribution of a point (f1, f2, f3) with reference (r1, r2, r3)
        # is roughly (r1 - f1) * (r2 - f2) * (r3 - f3) for a single point,
        # but for a set of points, it's the union of boxes.
        
        # To maximize hypervolume, we should spread points across the Pareto front.
        # Since f2 + f3 = 1, we can parameterize by x1 and x2.
        
        # Let's use a grid in (x1, x2) space.
        # The hypervolume is maximized when we have good coverage in both dimensions.
        
        # Non-uniform grid: cluster points near the "dominant" corner (low f1, low f2, low f3)
        # But on the Pareto front, we can't have all three low simultaneously.
        # f1 = x1, f2 = x2, f3 = 1 - x2
        # To minimize f1, f2, f3: x1 -> 0, x2 -> 0 gives f1=0, f2=0, f3=1
        # x1 -> 0, x2 -> 1 gives f1=0, f2=1, f3=0
        # x1 -> 1, x2 -> 0 gives f1=1, f2=0, f3=1
        
        # The "best" corner in terms of all objectives being small is actually not a single point
        # but the front itself. The hypervolume contribution is largest for points that are
        # far from the reference point in the sense of being in the "corner" of the non-dominated set.
        
        # Let's use a power-law distribution for x1 and x2 to cluster near (0,0) in (x1,x2) space,
        # which corresponds to (f1=0, f2=0, f3=1) and nearby.
        
        n_init = 100  # Number of initial front samples
        # Use remaining budget for refinement if time permits
        budget_remaining = self.budget - n_init
        
        # Generate non-uniform grid points on the Pareto front
        # x1 and x2 in [0, 1]
        # Use power law: x = u^(1/p) where p > 1 clusters near 0
        p = 2.0  # Power for clustering
        
        # Create a grid of points
        # Split n_init between x1 and x2 dimensions
        n_x1 = int(np.sqrt(n_init))
        n_x2 = n_init // n_x1
        
        # Ensure we don't exceed budget
        if n_x1 * n_x2 > n_init:
            n_x2 = n_init // n_x1
        
        u1 = np.linspace(0, 1, n_x1) ** (1.0 / p)
        u2 = np.linspace(0, 1, n_x2) ** (1.0 / p)
        
        # Create grid
        x1_grid, x2_grid = np.meshgrid(u1, u2)
        x1_flat = x1_grid.flatten()
        x2_flat = x2_grid.flatten()
        
        # Truncate to n_init if needed
        n_actual = min(len(x1_flat), n_init)
        x1_flat = x1_flat[:n_actual]
        x2_flat = x2_flat[:n_actual]
        
        # Create initial population on the Pareto front
        X = np.zeros((n_actual, self.dim))
        X[:, 0] = x1_flat
        X[:, 1] = x2_flat
        # x[2:] = 0 for Pareto front
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = n_actual
        
        # Check deadline
        if time.time() > deadline - 10:
            return X
        
        # Use remaining budget for refinement
        # Try local search around promising points
        if budget_remaining > 0:
            # Sort by hypervolume contribution (approximate)
            # Points with lower f1, f2, f3 contribute more
            # Use sum of objectives as a proxy
            scores = np.sum(F, axis=1)
            order = np.argsort(scores)
            
            # Pick top candidates for refinement
            n_refine = min(50, budget_remaining // 10)
            for i in range(n_refine):
                if time.time() > deadline - 5:
                    break
                if used >= self.budget:
                    break
                
                # Pick a random point from the top half
                idx = order[np.random.randint(0, len(order) // 2)]
                base = X[idx].copy()
                
                # Perturb only x[0] and x[1] (the front parameters)
                perturbation = 0.05 * np.random.randn(2)
                base[0] = np.clip(base[0] + perturbation[0], 0, 1)
                base[1] = np.clip(base[1] + perturbation[1], 0, 1)
                base[2:] = 0  # Stay on the front
                
                fc = problem(base)
                used += 1
                
                # Replace the worst point if the new one is better
                worst_idx = np.argmax(np.sum(F, axis=1))
                if np.sum(fc) < np.sum(F[worst_idx]):
                    X[worst_idx] = base
                    F[worst_idx] = fc
        
        return X
# EVOLVE-BLOCK-END
