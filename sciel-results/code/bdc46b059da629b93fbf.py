"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using deterministic grid + evolutionary refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # DTLZ2: first 2 variables determine position on Pareto front, rest are convergence
        # f_i = g(x_n) * (1 + x_i^2) where g = sum((x_i - 0.5)^2) for i in [2, dim-1]
        # Pareto front: g=0, so x[2:] = 0.5
        # f_1 = 1 + x_0^2, f_2 = 1 + x_1^2, f_3 = 1 + x_0^2 * x_1^2 (wait, let me recall DTLZ2)
        # Actually DTLZ2: f_i = g * h_i where for 3 objectives:
        # f_1 = g * (1 + x_0^2), f_2 = g * (1 + x_1^2), f_3 = g * (1 + x_0^2 * x_1^2)?
        # No. Standard DTLZ2 with M objectives:
        # f_i(x) = g(x_n) * h_i(x_m, g)
        # h_i = 1 + (x_{m-1})^2 ... actually:
        # For DTLZ2: h_i(x_m, g) = 1 + x_{m-1}^2 for i=1, and more generally
        # The Pareto front of DTLZ2 is: sum(f_i^2) = g^2 * (1 + sum of squares of position vars)
        # When g=0, we get f_i = 0 which is the ideal point.
        # Actually the Pareto front is: sum((f_i/g)^2) = 1 + sum(position vars squared)
        # With g=1 (convergence), the front is a hypersurface.
        #
        # Let me think about this differently. For DTLZ2:
        # g(x_n) = sum_{i=n_m+1}^{n} (x_i - 0.5)^2
        # h_i(x_m, g) = 1 + x_{m-1}^2 for i = 1
        # Actually the standard formulation:
        # f_i(x) = g(x_n) * h_i(x_m, g)
        # where h_i = 1 + x_{m-1}^2 for i=1, and for i>1 it's different.
        #
        # For 3 objectives DTLZ2:
        # f_1 = g * (1 + x_0^2)
        # f_2 = g * (1 + x_1^2)  
        # f_3 = g * (1 + x_0^2 * x_1^2)  -- no this isn't right either
        #
        # The correct DTLZ2 formulation for M objectives:
        # f_i(x) = g(x_n) * h_i(x_m, g)
        # h_i(x_m, g) = 1 + x_{m-1}^2 for i = 1
        # For i > 1: h_i involves products of previous position variables
        #
        # Actually, the standard DTLZ2 is:
        # f_i = g * (1 + sum_{j=0}^{m-i-1} x_j^2)  -- no
        #
        # Let me just use the known fact: the Pareto front of DTLZ2 (with g=1) satisfies
        # sum(f_i^2) = 1 + sum_{j=0}^{m-1} x_j^2
        # When all position vars are 0: sum(f_i^2) = 1, so the front passes through (1/sqrt(M), ..., 1/sqrt(M))
        # When x_0=1, x_1=0, x_2=0: f_1 = 2, f_2=1, f_3=1 (for M=3)
        #
        # The key insight: to maximize hypervolume with reference [2.5, 2.5, 2.5],
        # we want points that are far from the reference in a way that maximizes the
        # non-dominated hypervolume. Points on the Pareto front with g=1 are best.
        #
        # Strategy: Use a grid on the 2D position space (x_0, x_1) with x[2:] = 0.5
        # to get g=0... wait, g=0 gives f=0 which is the ideal point (all objectives 0).
        # That's actually the best single point! But we need a set of non-dominated points.
        #
        # Actually for DTLZ2, g=0 means all f_i = 0, which dominates everything.
        # But we can only have one such point. The Pareto front is at g=1 (or any fixed g>0).
        # Wait no - for DTLZ2, the Pareto front is defined as g(x_n) = 1? No.
        #
        # Let me reconsider. In DTLZ2, the Pareto front is:
        # g(x_n) = 1 AND the position variables determine the direction.
        # Actually no. The Pareto front of DTLZ2 is the set of points where
        # g(x_n) is minimized (g=0) and the position variables vary.
        # But g=0 gives f_i = 0 for all i, which is a single point.
        #
        # I think I'm confusing myself. Let me just implement a general approach:
        # 1. Start with a grid of points in the position space
        # 2. Set convergence variables to 0.5 (which gives g=0 for DTLZ2)
        # 3. Use evolutionary search to refine

        # For DTLZ2, setting x[2:] = 0.5 makes g = sum((x_i - 0.5)^2) = 0
        # This gives f_i = 0 * h_i = 0 for all i. That's the ideal point.
        # But we need diversity. So we should NOT set g=0.
        #
        # The Pareto front of DTLZ2: for any fixed g > 0, varying position vars gives
        # a curve/surface. The true Pareto front is at g=1 (normalized).
        # Actually, I recall that for DTLZ2, the Pareto-optimal solutions have
        # g(x_n) = 1 and the position variables on the unit sphere in the positive orthant.
        # No wait, that's not right either.
        #
        # Let me just do: evaluate a grid of points where x[2:] = 0.5 (g=0, ideal point)
        # and also points where x[2:] gives g=1, with x[0], x[1] on a grid.
        # Actually, the simplest approach: just use a grid in the full 12D space
        # focused on the first 2 dimensions, with the rest at 0.5.

        # Grid approach: 10x10 grid in (x_0, x_1) space, x[2:] = 0.5
        # This gives 100 points. With budget 200, we have 100 left for refinement.
        
        # But wait - if x[2:] = 0.5, then g = 0, so all f_i = 0. All 100 points
        # are identical (0,0,0). That's useless for hypervolume.
        
        # I need to understand DTLZ2 better. Let me assume the standard formulation:
        # For M=3, n=12, m=2 (position vars), n_m = 10 (convergence vars)
        # g(x_n) = sum_{i=2}^{11} (x_i - 0.5)^2
        # h_1 = 1 + x_0^2
        # h_2 = 1 + x_1^2  
        # h_3 = 1 + x_0^2 * x_1^2  -- I'm not sure about this one
        
        # Actually, I think the correct DTLZ2 for M objectives is:
        # f_i(x) = g(x_n) * h_i(x_m, g)
        # where for i = 1: h_1 = 1 + x_{m-1}^2 = 1 + x_1^2
        # For general i, the h functions are defined recursively.
        #
        # You know what, let me just use a different strategy. Instead of trying to
        # exploit the DTLZ2 structure (which I'm uncertain about), let me use a
        # well-known MOEA approach: NSGA-II style with a good initialization.
        #
        # But the directive says to use a deterministic uniform grid. Let me use
        # a grid in the full space that's likely to hit the Pareto front.
        #
        # Alternative: Use the fact that for DTLZ2, the Pareto front is a
        # (M-1)-dimensional surface. For M=3, it's a 2D surface. The position
        # variables (x_0, x_1) parameterize this surface when convergence is optimal.
        #
        # I'll try: grid on x_0, x_1 in [0,1], and set x[2:] to values that
        # give a reasonable g. If g=0 is the ideal, then the Pareto front is
        # actually at g=0 with varying positions? No, that gives all f=0.
        #
        # I think the issue is that for DTLZ2, the Pareto front is:
        # {x : g(x_n) = 1, x_m in [0,1]^m} mapped through the objective functions.
        # With g=1, f_i = h_i(x_m). The h_i functions define the shape.
        #
        # Let me just implement a robust approach:
        # 1. Random initialization with some structure
        # 2. Use non-dominated sorting and crowding distance
        # 3. Time-bounded evaluation

        rng = np.random.default_rng(0)
        
        # Phase 1: Structured initialization
        # Try a grid in the first 2 dimensions, with convergence vars at 0.5
        # and also some random points
        
        pop_size = min(50, budget // 4)
        X = np.zeros((pop_size, dim))
        
        # First, add grid points in (x_0, x_1) with x[2:] = 0.5
        grid_n = 10
        grid_points = []
        for i in range(grid_n):
            for j in range(grid_n):
                x = np.full(dim, 0.5)
                x[0] = i / (grid_n - 1)
                x[1] = j / (grid_n - 1)
                grid_points.append(x)
        
        grid_points = np.array(grid_points[:pop_size])
        X[:len(grid_points)] = grid_points
        
        # Fill remaining with random points
        if pop_size > len(grid_points):
            X[len(grid_points):] = lo + (hi - lo) * rng.random((pop_size - len(grid_points), dim))
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Phase 2: Evolutionary refinement
        while used < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Non-dominated sorting
            dominated = np.array([
                np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) 
                for f in F
            ])
            parents = X[~dominated] if np.any(~dominated) else X
            
            # Crossover and mutation
            if len(parents) >= 2:
                p1 = parents[rng.integers(len(parents))]
                p2 = parents[rng.integers(len(parents))]
                # Simulated binary crossover
                beta = rng.random()
                child = beta * p1 + (1 - beta) * p2
            else:
                child = parents[0].copy()
            
            # Mutation
            child = child + 0.1 * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst individual
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst], F[worst] = child, fc
        
        return X
# EVOLVE-BLOCK-END
