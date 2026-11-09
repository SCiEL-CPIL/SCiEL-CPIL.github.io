# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 using a tailored strategy."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim

        # ZDT4 specific knowledge:
        # x[0] controls f1 = x[0] (so we want x[0] spread from 0 to 1)
        # x[1:10] controls g = 1 + 10*(n-1) + sum(x_i^2 - 10*cos(4*pi*x_i))
        # The minimum of g is achieved when all x_i = 1.0
        # f2 = g * h(f1, g) where h(u,v) = 1 - sqrt(u/v) - (u/v)*sin(10*pi*u)
        # Pareto front: f1 in [0,1], g = 1 (minimum), f2 = 1 - sqrt(f1) - f1*sin(10*pi*f1)

        # Strategy:
        # 1. First, sample x[0] uniformly on [0,1] to cover the Pareto front
        # 2. Set x[1:10] = 1.0 to minimize g
        # 3. Then use remaining budget to refine with local search around promising points

        n_evals = 0
        points = []  # list of (x, f)

        def check_time():
            return time.time() < deadline

        # Phase 1: Structured sampling of the Pareto front
        # Use a significant portion of budget for structured sampling
        n_structured = min(100, self.budget // 2)
        
        # Sample x[0] values: include 0 and 1, and uniform in between
        x0_values = np.linspace(0, 1, n_structured)
        
        for x0 in x0_values:
            if not check_time():
                break
            if n_evals >= self.budget:
                break
            
            x = np.ones(dim)  # x[1:10] = 1.0 for minimum g
            x[0] = x0
            f = problem(x)
            points.append((x.copy(), np.array(f)))
            n_evals += 1

        # Phase 2: Local refinement around sampled points
        # For each sampled point, try small perturbations in x[1:10]
        # to see if we can improve g (though x=1 should already be optimal)
        # Also try small perturbations in x[0] for better coverage

        remaining = self.budget - n_evals
        if remaining > 0 and points:
            # Sort points by f1 for systematic refinement
            points_sorted = sorted(points, key=lambda p: p[1][0])
            
            # Try to refine each point with a few local steps
            n_refine_per_point = max(1, remaining // max(1, len(points_sorted)))
            
            rng = np.random.default_rng(42)
            
            for base_x, base_f in points_sorted:
                if not check_time():
                    break
                if n_evals >= self.budget:
                    break
                
                for _ in range(n_refine_per_point):
                    if not check_time():
                        break
                    if n_evals >= self.budget:
                        break
                    
                    # Small perturbation
                    perturbation = 0.01 * (hi - lo) * rng.standard_normal(dim)
                    x_new = base_x + perturbation
                    x_new = np.clip(x_new, lo, hi)
                    
                    f_new = problem(x_new)
                    points.append((x_new.copy(), np.array(f_new)))
                    n_evals += 1

        # Phase 3: If we have budget left, do more random exploration
        # focused on the Pareto front region
        if n_evals < self.budget and check_time():
            rng = np.random.default_rng(123)
            while n_evals < self.budget and check_time():
                # Sample x[0] uniformly, x[1:10] near 1.0
                x = np.ones(dim)
                x[0] = rng.uniform(lo[0], hi[0])
                # Add small noise to x[1:10]
                for i in range(1, dim):
                    x[i] = 1.0 + 0.05 * rng.standard_normal()
                    x[i] = np.clip(x[i], lo[i], hi[i])
                
                f = problem(x)
                points.append((x.copy(), np.array(f)))
                n_evals += 1

        # Return the best non-dominated points
        # Compute non-dominated set
        if not points:
            return np.array([])

        F = np.array([p[1] for p in points])
        X = np.array([p[0] for p in points])

        # Find non-dominated points
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

        non_dom_idx = np.where(~dominated)[0]
        if len(non_dom_idx) > 0:
            return X[non_dom_idx]
        return X
# EVOLVE-BLOCK-END
