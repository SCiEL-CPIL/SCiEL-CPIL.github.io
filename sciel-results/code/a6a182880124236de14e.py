"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Coarse-to-fine lattice global search with Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # Safety margin before 330s kill

        best_x = None
        best_f = float('inf')
        calls = 0

        def safe_func(x):
            nonlocal calls
            if calls >= budget:
                return float('inf')
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Coarse-to-fine lattice search
        # We use a deterministic grid that becomes progressively finer.
        # For dim=5, a full grid of 3^5=243 points is affordable.
        # Then 5^5=3125 is too many for a single pass, but we can do
        # a subset or a coarser refinement.
        
        # Strategy:
        # Level 0: 3 points per dimension -> 3^5 = 243 points
        # Level 1: 5 points per dimension around the best region from Level 0
        #          But we can't just pick the best single point; we need to explore.
        #          Instead, let's do a global Halton-like or stratified sampling
        #          for the first ~300 calls, then refine.
        
        # Actually, the directive suggests a "dense Halton sequence or coarse-to-fine lattice".
        # Let's implement a coarse-to-fine lattice.
        
        # Level 0: Grid of 3x3x3x3x3 = 243 points
        # This covers the whole space [-5, 5]^5 with step size 5.0
        # Points: -5, 0, 5
        
        level0_points = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            p = np.array([i, j, k, l, m], dtype=float)
                            p = p * 5.0 - 5.0  # Map {0,1,2} to {-5,0,5}
                            level0_points.append(p)
        
        # Shuffle to avoid bias towards early indices (though Halton would be better)
        # But since the landscape is unknown, a deterministic order is fine.
        # Let's use a simple shuffle with a fixed seed for reproducibility.
        rng = np.random.default_rng(0)
        rng.shuffle(level0_points)
        
        # Evaluate Level 0
        for p in level0_points:
            if time.time() > deadline - 10.0 or calls >= budget:
                break
            safe_func(p)
        
        # If we used all budget, return
        if calls >= budget:
            return best_x, best_f
        
        # Level 1: Refine around the best point found in Level 0
        # Create a local grid of 5x5x5x5x5 = 3125 points? No, that's too many.
        # Let's do a 3x3x3x3x3 grid around the best point with a smaller step.
        # The step in Level 0 was 5.0. Let's use a step of 2.5 for Level 1.
        # Center is best_x. Range: best_x +/- 5.0 (clipped to [-5,5])
        
        # Actually, a better approach for multimodal functions like Rastrigin:
        # Don't just refine the single best. Rastrigin has many local minima.
        # The global minimum is shifted. We need to explore more broadly.
        
        # Let's use a Halton sequence for the remaining budget, starting from a
        # deterministic set of points that cover the space well.
        
        # Generate Halton points for the remaining budget
        remaining_budget = budget - calls
        # Use a mix: some Halton, some local refinement
        
        # Halton sequence generator
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        bases = [2, 3, 5, 7, 11][:dim]
        
        # Generate Halton points
        # We start from index 1 to avoid the origin (0,0,...,0) which is already covered
        # Actually, let's just generate a large batch and use them.
        
        # To be efficient, let's generate points on the fly.
        # We'll use the next available Halton indices.
        
        # How many Halton points? Let's use 70% of remaining budget for global exploration.
        # And 30% for local refinement around the best few points.
        
        num_halton = int(remaining_budget * 0.7)
        num_local = remaining_budget - num_halton
        
        # Evaluate Halton points
        halton_index = 1
        # We already evaluated some points in Level 0. Halton points are different.
        # Let's just evaluate num_halton new Halton points.
        
        for _ in range(num_halton):
            if time.time() > deadline - 5.0 or calls >= budget:
                break
            p = np.array([halton(halton_index, base) for base in bases])
            p = p * 10.0 - 5.0  # Map [0,1] to [-5,5]
            halton_index += 1
            safe_func(p)
        
        if calls >= budget:
            return best_x, best_f
        
        # Local refinement: Use Nelder-Mead starting from the best point
        # and also from the second and third best points to avoid getting stuck
        # in a local minimum.
        
        # We need to track the top K points.
        # For simplicity, let's just use the best point for now, but maybe
        # restart NM from a few of the best Halton points.
        
        # Let's collect the top 3 best points from all evaluations so far.
        # We need to store them. Let's modify safe_func to store history.
        # But we already have a list of points evaluated. Let's just re-evaluate
        # the best few? No, that wastes calls.
        
        # Alternative: Just run NM from the current best_x.
        # If it gets stuck, we can try another point. But we don't have the
        # second best stored easily.
        
        # Let's just run NM from best_x.
        
        if time.time() < deadline - 2.0 and calls < budget:
            nm_calls = 0
            def nm_func(x):
                nonlocal nm_calls, calls, best_f, best_x
                if calls >= budget:
                    return best_f
                nm_calls += 1
                x_clipped = np.clip(x, -5.0, 5.0)
                fx = func(x_clipped)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x_clipped.copy()
                return fx
            
            try:
                res = minimize(nm_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-14})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
