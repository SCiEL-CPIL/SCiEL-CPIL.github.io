"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hypervolume-driven multi-objective optimizer using local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([3000.0, 0.06])
        
        # Store all evaluated points and their objectives
        archive = []
        used = 0
        
        def evaluate(x):
            nonlocal used
            if used >= self.budget:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            if np.all(f < ref):
                archive.append((x, f))
            used += 1
            return f

        # 1. Initial sampling to get a spread of points
        # Use a fixed number of initial points, e.g., 20
        n_init = min(20, self.budget // 5)
        rng = np.random.default_rng(42)
        
        for _ in range(n_init):
            x = lo + (hi - lo) * rng.random(self.dim)
            evaluate(x)
            
        # Helper to compute non-dominated front
        def get_non_dominated(points):
            if not points:
                return []
            nd = []
            for i, p in enumerate(points):
                dominated = False
                for j, q in enumerate(points):
                    if i == j:
                        continue
                    if np.all(q <= p) and np.any(q < p):
                        dominated = True
                        break
                if not dominated:
                    nd.append(p)
            return nd

        # Helper to compute hypervolume contribution of a point in 2D
        def hv_contribution(point, front, ref):
            if not front:
                return 0.0
            # Filter front to points better than ref
            front = [f for f in front if np.all(f < ref)]
            if not front:
                return 0.0
            
            # Sort by first objective
            front_sorted = sorted(front, key=lambda x: x[0])
            
            # Find position of point in sorted order
            # If point is not in front, we need to calculate area it would add
            # This is complex, so we approximate by checking if it dominates existing points
            # or is dominated by them.
            
            # For a 2D front, the hypervolume is the sum of rectangles.
            # A simple approximation for "contribution" is the area of the box 
            # defined by the point and the reference, minus the area already covered.
            # However, a better heuristic for selection is:
            # How much does this point extend the front?
            
            # Let's use a simpler metric: 
            # If point is non-dominated, its contribution is roughly 
            # the area of the rectangle it defines with the "next" points.
            
            # To keep it fast, we'll just return a dummy value or use a simpler metric
            # like distance to the worst point in the front.
            # Actually, for selection, we can just pick the point that is 
            # most "spread out" or far from the centroid of the current front.
            
            # Let's use a simpler strategy: 
            # Pick the point that maximizes the minimum distance to other points in the front (max-min diversity)
            # OR pick the point that is non-dominated and has the best sum of objectives.
            
            # For now, let's just return 0.0 and rely on non-domination for selection in the main loop.
            # We will refine the selection logic in the main loop.
            return 0.0

        # 2. Local Search Phase
        # We will perform local searches from the most promising initial points.
        # Promising points are those that are non-dominated.
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Get current non-dominated set
            current_front = get_non_dominated([f for x, f in archive])
            
            if not current_front:
                # If no points are better than ref, do random search
                x = lo + (hi - lo) * rng.random(self.dim)
                evaluate(x)
                continue
            
            # Select a starting point for local search
            # We want to explore different parts of the front.
            # Let's pick a random non-dominated point from the archive.
            nd_indices = [i for i, (x, f) in enumerate(archive) if f in current_front]
            if not nd_indices:
                # Fallback: pick random
                idx = rng.integers(len(archive))
            else:
                idx = nd_indices[rng.integers(len(nd_indices))]
            
            x0 = archive[idx][0]
            f0 = archive[idx][1]
            
            # Define a scalarized objective for local search
            # We can use a weighted sum with random weights to explore different parts
            w = rng.random(self.n_obj)
            w = w / np.sum(w)
            
            def scalarized_obj(x):
                f = evaluate(x)
                if f is None:
                    return 1e10
                return np.dot(w, f)
            
            # Perform Nelder-Mead local search
            # Limit the number of iterations to save budget
            max_iter = 10
            try:
                res = minimize(scalarized_obj, x0, method='Nelder-Mead', 
                               options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4})
            except Exception:
                pass
            
            # If the local search found a better point, it's already in the archive
            # because evaluate() adds it.
            
            # If we still have budget, we can do a few more random perturbations
            # around the best found point in this local search.
            if res.success and used < self.budget:
                x_best = res.x
                f_best = res.fun
                # f_best is the scalarized value, not the vector. We need to re-evaluate or store it.
                # But evaluate() already stored it. Let's find it in the archive.
                # Actually, it's easier to just do a few random steps around x_best.
                for _ in range(3):
                    if used >= self.budget or time.time() > deadline:
                        break
                    x_pert = x_best + 0.1 * (hi - lo) * rng.standard_normal(self.dim)
                    evaluate(x_pert)

        return archive
# EVOLVE-BLOCK-END
