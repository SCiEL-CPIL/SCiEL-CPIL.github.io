"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search with deadline guard and refined selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        
        # Initialize population
        pop_size = min(20, self.budget // 4)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Reference point for hypervolume contribution (used for selection)
        # Based on task: reference point is [3.0, 5.0, 7.0]
        ref_point = np.array([3.0, 5.0, 7.0])
        
        def hypervolume_contribution(points):
            """Compute hypervolume contribution of each point in the set."""
            n = len(points)
            if n == 0:
                return np.array([])
            if n == 1:
                vol = np.prod(ref_point - points[0])
                return np.array([max(0, vol)])
            
            # For small n, we can use a simple O(n^2) or O(n^3) method
            # Here we use a simple approach: for each point, compute volume of region it uniquely dominates
            # This is an approximation for selection purposes only
            
            contribs = np.zeros(n)
            for i in range(n):
                # Check if point i is non-dominated
                dominated = False
                for j in range(n):
                    if i != j:
                        if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                            dominated = True
                            break
                if dominated:
                    continue
                
                # Approximate contribution: volume of box from point i to ref_point
                # minus overlap with other non-dominated points (simplified)
                box_vol = np.prod(ref_point - points[i])
                if box_vol <= 0:
                    continue
                
                # Simple approximation: assume uniform distribution
                # Better: compute exact contribution for small n
                # For now, use a heuristic: inverse of rank
                contribs[i] = box_vol / (n * np.log(n + 1))
            
            # Normalize to sum to 1 for selection
            total = contribs.sum()
            if total > 0:
                contribs /= total
            else:
                contribs = np.ones(n) / n
            
            return contribs

        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Identify non-dominated points
            dominated = np.array([
                np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) 
                for f in F
            ])
            non_dom_idx = np.where(~dominated)[0]
            
            if len(non_dom_idx) == 0:
                non_dom_idx = np.arange(len(F))
            
            # Select parent using hypervolume contribution
            non_dom_F = F[non_dom_idx]
            contribs = hypervolume_contribution(non_dom_F)
            parent_idx = non_dom_idx[rng.choice(len(non_dom_idx), p=contribs)]
            
            # Mutate
            # Adaptive mutation strength
            mutation_scale = 0.1 * (hi - lo)
            child = X[parent_idx] + mutation_scale * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst point (by sum of objectives)
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst], F[worst] = child, fc
            
            # Final polish: if we have spare budget, do local refinement
            if used >= self.budget - 10 and time.time() < deadline - 10:
                # Do a few local steps
                for _ in range(min(5, self.budget - used)):
                    if time.time() > deadline:
                        break
                    # Pick best point and do small local step
                    best = int(np.argmin(np.sum(F, axis=1)))
                    small_step = 0.01 * (hi - lo)
                    child = X[best] + small_step * rng.standard_normal(self.dim)
                    child = np.clip(child, lo, hi)
                    fc = problem(child)
                    used += 1
                    # Replace if better
                    if np.sum(fc) < np.sum(F[best]):
                        X[best], F[best] = child, fc
        
        return X
# EVOLVE-BLOCK-END
