"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Constrained random search within a shrinking hypersphere for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget

        # Center of the search region (origin for WFG9)
        center = np.zeros(dim, dtype=np.float64)
        
        # Initial radius: large enough to cover the bounds, then shrinks
        # WFG9 Pareto front is typically near the origin in the decision space
        initial_radius = max(np.max(hi - lo) * 0.5, 1.0)
        
        rng = np.random.default_rng(0)
        
        # Store all evaluated points to compute hypervolume later if needed, 
        # but we just need to return a representative set or the best ones.
        # The task says "return value of __call__ is ignored", but we should 
        # return something sensible. We'll return the non-dominated set.
        
        X = []
        F = []
        used = 0
        
        # We will use a simple archive of non-dominated points
        # Since budget is small (200), we can keep all and filter at the end
        all_X = []
        all_F = []

        # Phase 1: Shrinking sphere sampling
        # Radius shrinks from initial_radius to a small value
        min_radius = 0.1
        
        for i in range(budget):
            if time.time() > deadline:
                break
            
            # Linear shrinkage
            t = i / max(1, budget - 1)
            radius = initial_radius * (1.0 - t) + min_radius * t
            
            # Sample direction
            direction = rng.standard_normal(dim)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction /= norm
            else:
                direction[0] = 1.0
            
            # Sample point on the sphere surface (or within? "within a hyper-sphere")
            # Let's sample within the sphere for better coverage, using uniform distribution in ball
            # r = radius * (rng.random() ** (1/dim)) for uniform volume distribution
            # Or just on the surface for WFG9 front which is often near a specific manifold
            # Let's try sampling within the ball
            r = radius * (rng.random() ** (1.0 / dim))
            
            x = center + r * direction
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            # Evaluate
            f = problem(x)
            used += 1
            
            all_X.append(x)
            all_F.append(f)
            
            # Optional: Local refinement if we find a good point?
            # The strategy says "bypassing deceptive intermediate regions entirely without relying on evolutionary operators or local search."
            # So we stick to pure random search in the shrinking sphere.
            
        # If we have time and budget left, or if the loop finished, 
        # we might want to do a few more samples near the very center if the front is there.
        
        # Convert to arrays
        if not all_X:
            # Fallback if no evaluations could be done (should not happen)
            return np.array([np.clip(center, lo, hi)])
        
        X_arr = np.array(all_X)
        F_arr = np.array(all_F)
        
        # Find non-dominated points to return a clean set (though return is ignored)
        # Simple O(N^2) dominance check for small N (200)
        n = len(F_arr)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if not dominated[j]:
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break
        
        idx = np.where(~dominated)[0]
        if len(idx) == 0:
            idx = [0]
            
        return X_arr[idx]
# EVOLVE-BLOCK-END
