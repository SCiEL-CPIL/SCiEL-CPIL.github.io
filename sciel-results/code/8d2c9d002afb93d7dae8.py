"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT1-aware optimizer: exploit known Pareto front structure via stratified sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 310.0  # Safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        # ZDT1: f1 = x1, g = 1 + 9*mean(x2..xn), h = 1 - sqrt(f1/g)*sin(10*pi*f1)
        # Pareto front: f1 in [0,1], f2 = 1 - sqrt(f1)
        # Strategy: 
        # 1. Use stratified sampling of f1 values to cover the front
        # 2. For each f1 target, optimize the remaining variables to minimize g
        #    (since g depends on x2..xn, and h is minimized when g is maximized... wait)
        # Actually for ZDT1: to get f2 = 1 - sqrt(f1/g)*sin(10*pi*f1), we need g as large as possible
        # but the Pareto front is f2 = 1 - sqrt(f1) which assumes g = 1 + 9*(n-1)/n... 
        # Wait, the standard ZDT1 Pareto front is f1 in [0,1], f2 = 1 - sqrt(f1).
        # This is achieved when g is at its maximum? No, let's reconsider.
        # f2 = g * h = g * (1 - sqrt(f1/g) * sin(10*pi*f1))
        # For the Pareto front, we want f2 = 1 - sqrt(f1). This happens when g is such that
        # the expression simplifies. Actually, the standard result is that the Pareto front
        # is f1 in [0,1], f2 = 1 - sqrt(f1), achieved when x1 = f1 and x2..xn are all 0 or 1
        # such that g is maximized? No, g = 1 + 9*mean(x2..xn), so max g = 10 when all x2..xn=1.
        # But then f2 = 10 * (1 - sqrt(f1/10)*sin(10*pi*f1)). That's not 1 - sqrt(f1).
        # I think I'm confusing myself. Let me just use a direct approach:
        # 
        # Better approach: Since we know the Pareto front of ZDT1 is f1 in [0,1], f2 = 1 - sqrt(f1),
        # we can:
        # 1. Sample many f1 values uniformly in [0,1]
        # 2. For each f1, set x1 = f1
        # 3. Optimize x2..xn to make the point as close to the Pareto front as possible
        #    i.e., minimize f2 given f1, which means we need to find x2..xn that minimize
        #    g * (1 - sqrt(f1/g)*sin(10*pi*f1))
        #
        # But actually, for a simpler and effective approach:
        # - The key insight for ZDT1 is that f1 = x1, and the rest of the variables affect f2
        #   through g. To get good hypervolume, we want to cover the entire Pareto front.
        # - A simple but effective strategy: sample x1 uniformly, and for each, try to set
        #   x2..xn to values that give a good f2. Since we don't know the exact problem,
        #   we should use a search-based approach.
        #
        # Let me use a different approach: NSGA-II with proper non-dominated sorting.
        # But with only 200 evaluations, we need to be efficient.
        #
        # Actually, let me try a simpler approach first:
        # 1. Evaluate a grid of points with x1 spread across [0,1] and x2..xn at various values
        # 2. Use the remaining budget to refine the best regions
        #
        # With 200 evaluations, let's do:
        # - 50 initial evaluations: x1 = linspace(0, 1, 50), x2..xn = 0.5
        # - Then for each of the 50 points, use 3 evaluations to perturb x2..xn
        #   to try to improve f2
        #
        # Actually, let me just do a smart random search with knowledge of ZDT1 structure.

        # Phase 1: Stratified sampling of x1
        n_strata = 40
        x1_values = np.linspace(0.0, 1.0, n_strata)
        
        points = []
        objectives = []
        used = 0

        # For each x1 value, evaluate with x2..xn = 0.5
        for x1 in x1_values:
            if used >= budget:
                break
            if time.time() > deadline - 5:
                break
            x = np.full(dim, 0.5)
            x[0] = x1
            x = np.clip(x, lo, hi)
            f = problem(x)
            points.append(x)
            objectives.append(f)
            used += 1

        # Phase 2: For remaining budget, do local search around the stratified points
        # Perturb x2..xn to try to reduce f2
        remaining = budget - used
        n_per_point = max(1, remaining // n_strata)
        
        for i in range(n_strata):
            if used >= budget:
                break
            if time.time() > deadline - 5:
                break
            base_x = points[i].copy()
            base_f = objectives[i]
            
            for j in range(n_per_point):
                if used >= budget:
                    break
                if time.time() > deadline - 5:
                    break
                # Perturb x2..xn
                x = base_x.copy()
                x[1:] = base_x[1:] + 0.3 * rng.standard_normal(dim - 1)
                x = np.clip(x, lo, hi)
                f = problem(x)
                points.append(x)
                objectives.append(f)
                used += 1

        # Phase 3: Use any remaining budget for random exploration
        while used < budget:
            if time.time() > deadline - 5:
                break
            x = lo + (hi - lo) * rng.random(dim)
            f = problem(x)
            points.append(x)
            objectives.append(f)
            used += 1

        return points
# EVOLVE-BLOCK-END
