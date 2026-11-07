"""Optimized multi-objective optimizer for RE21 with hypervolume-guided adaptive DE."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hypervolume-guided adaptive differential evolution for 2-objective RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        # Reference point for hypervolume (only points strictly better in all objectives count)
        ref = np.array([3000.0, 0.06])

        def hypervolume_2d(points):
            """Exact hypervolume for 2D minimization with given reference point.
            Only points strictly better than ref in both objectives contribute."""
            # Filter points strictly better than ref in both objectives
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            # Sort by first objective (ascending)
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            hv = 0.0
            prev_y = ref[1]
            for i in range(len(pts)):
                x_i, y_i = pts[i]
                # Contribution: width in x * height in y
                # Width is from x_i to next point's x (or ref[0] if last)
                if i == len(pts) - 1:
                    next_x = ref[0]
                else:
                    next_x = pts[i + 1, 0]
                # Only if y_i < prev_y (which should be true since we're going in order of x)
                # Actually, we need the min y so far. Let's use a different approach.
                pass
            
            # Better approach: sort by x, then track minimum y seen so far
            order = np.argsort(pts[:, 0])
            pts_sorted = pts[order]
            hv = 0.0
            min_y = ref[1]
            for i in range(len(pts_sorted)):
                x_i, y_i = pts_sorted[i]
                if y_i < min_y:
                    # The rectangle from x_i to next_x (or ref[0]) with height (min_y - y_i)
                    # But we need to be careful: the contribution is (next_x - x_i) * (min_y - y_i)
                    # where next_x is the x of the next point or ref[0]
                    next_x = pts_sorted[i + 1, 0] if i + 1 < len(pts_sorted) else ref[0]
                    hv += (next_x - x_i) * (min_y - y_i)
                    min_y = y_i
            return hv

        # Deterministic grid initialization to saturate the asymmetric space
        # The first objective dominates hypervolume contribution, so spread more in x
        # Use a grid that covers the bounds well
        n_grid = 16  # 4x4 grid
        xs = np.linspace(lo[0], hi[0], 4)
        ys = np.linspace(lo[1], hi[1], 4)
        # Also add points for other dimensions
        # For 4D, we'll create a structured grid
        # Let's use a Sobol-like or just structured random with deterministic seed
        rng = np.random.RandomState(42)
        
        # Create initial population: mix of grid points and random
        # Grid in first two dimensions, random in last two
        grid_points = []
        for x in xs:
            for y in ys:
                for z in [lo[2], (lo[2]+hi[2])/2, hi[2]]:
                    for w in [lo[3], (lo[3]+hi[3])/2, hi[3]]:
                        grid_points.append([x, y, z, w])
        
        # Take up to budget//3 for initialization
        n_init = min(60, budget // 3)
        if len(grid_points) > n_init:
            # Randomly select from grid points
            idx = rng.choice(len(grid_points), size=n_init, replace=False)
            X = np.array([grid_points[i] for i in idx])
        else:
            X = np.array(grid_points)
            # Pad with random if needed
            while len(X) < n_init:
                X = np.vstack([X, lo + (hi - lo) * rng.random(dim)])
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = len(X)
        
        # Adaptive differential evolution loop
        # Population size for DE
        pop_size = min(40, budget // 5)
        # Ensure we have at least pop_size points
        while len(X) < pop_size:
            new_x = lo + (hi - lo) * rng.random(dim)
            X = np.vstack([X, new_x])
            F = np.vstack([F, [problem(new_x)]])
            used += 1
        
        # Keep only the best pop_size points based on non-domination
        def non_dominated_sort(F):
            """Return indices of non-dominated points."""
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]
        
        # Main loop
        gen = 0
        while used < budget:
            gen += 1
            # Dynamically shrinking mutation factor
            # Start with 0.8, shrink to 0.2
            progress = min(1.0, used / budget)
            F_mut = 0.8 - 0.6 * progress
            
            # Get non-dominated points as parents
            nd_idx = non_dominated_sort(F)
            if len(nd_idx) < 2:
                # If fewer than 2 non-dominated, use all points
                nd_idx = np.arange(len(F))
            
            # Generate children using differential evolution
            n_children = min(pop_size, budget - used)
            for _ in range(n_children):
                if used >= budget:
                    break
                # Pick 3 distinct parents from non-dominated set
                if len(nd_idx) >= 3:
                    p1_idx, p2_idx, p3_idx = rng.choice(len(nd_idx), size=3, replace=False)
                    p1, p2, p3 = X[nd_idx[p1_idx]], X[nd_idx[p2_idx]], X[nd_idx[p3_idx]]
                    mutant = p1 + F_mut * (p2 - p3)
                elif len(nd_idx) == 2:
                    p1, p2 = X[nd_idx[0]], X[nd_idx[1]]
                    mutant = p1 + F_mut * (p2 - p1) * rng.random()
                else:
                    mutant = X[nd_idx[0]].copy()
                
                # Add some random perturbation
                mutant = mutant + 0.05 * (hi - lo) * rng.standard_normal(dim)
                mutant = np.clip(mutant, lo, hi)
                
                f_mut = problem(mutant)
                used += 1
                
                # Replace the worst point in the current population
                # Use hypervolume contribution as a proxy for quality
                # Simply replace the point with the largest sum of objectives
                worst_idx = np.argmax(np.sum(F, axis=1))
                X[worst_idx] = mutant
                F[worst_idx] = f_mut
            
            # Periodically re-sort and keep best points
            if gen % 5 == 0:
                # Keep only the best pop_size points
                if len(X) > pop_size:
                    # Sort by hypervolume contribution (approximate by sum)
                    order = np.argsort(np.sum(F, axis=1))
                    X = X[order[:pop_size]]
                    F = F[order[:pop_size]]
        
        return X
# EVOLVE-BLOCK-END
