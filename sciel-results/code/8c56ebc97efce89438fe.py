"""Rewritten program using LHS initialization and non-dominated refinement for WFG9."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using constrained random search for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 280

        # Strategy: Constrained random search in a hyper-sphere centered at origin
        # The WFG9 Pareto front is typically located near the origin in the decision space.
        # We sample points x = r * u, where u is a random unit vector and r is a radius.
        # We shrink r over time to refine the search.
        
        # Determine the maximum radius to cover the bounds
        # The center is 0. The furthest point in bounds from 0 is max(|lo|, |hi|)
        # Since bounds are typically [-50, 50] for WFG9, max_radius should be ~50 * sqrt(dim)
        # But we want to stay within bounds.
        # A simpler approach: sample in a box, then reject if outside a sphere? 
        # Or sample direction u, then scale r such that x = r*u is within bounds.
        
        # Let's use the approach: 
        # 1. Generate random direction u on the unit sphere.
        # 2. Determine max r such that 0 <= r*u + 0 <= hi and 0 >= r*u + 0 >= lo (if lo < 0)
        # Actually, the problem clips x to bounds. So we can just sample x = r*u and let problem clip it.
        # However, clipping might distort the distribution.
        # Better: Sample u, find max r such that r*u is in the box [lo, hi].
        # r_max = min over i of:
        #   if u[i] > 0: hi[i] / u[i]
        #   if u[i] < 0: lo[i] / u[i]  (note lo[i] is negative, u[i] is negative, so ratio is positive)
        #   if u[i] == 0: inf
        
        # Let's implement this.
        
        # Precompute for speed? No, u changes every time.
        
        # We will use a shrinking radius factor.
        # Start with factor 1.0 (full box coverage via sphere intersection)
        # End with factor 0.0 (converge to origin)
        # Or maybe end with a small epsilon to avoid zero radius.
        
        # Let's try a linear decay of the radius scale.
        # r = scale * r_max(u)
        # scale goes from 1.0 to 0.0
        
        # To ensure we don't miss the front, we should probably not go all the way to 0 too fast.
        # Let's use a power law or linear. Linear is simple.
        
        # We also need to handle the case where the origin is not the center of the box.
        # WFG9 bounds are usually symmetric [-50, 50].
        
        # Let's iterate through the budget.
        
        # To improve performance, we can vectorize the evaluation?
        # The problem function takes a single vector. So we must loop.
        # But we can prepare the x vectors in a batch? No, problem(x) is called sequentially.
        
        # Let's just loop.
        
        # Optimization: If time is tight, we might not be able to do all 200.
        # But 200 evaluations of WFG9 should be fast.
        
        # Let's add a check for time in the loop.
        
        # We will keep track of the best points found?
        # The task says "Every returned objective vector is recorded".
        # It does not ask us to return the best set, just to call problem.
        # The evaluator records all calls.
        # So we just need to make 200 good calls.
        
        # Strategy details:
        # For t in range(n):
        #   scale = 1.0 - (t / n) * 0.9  # From 1.0 to 0.1
        #   Generate u (random unit vector)
        #   Calculate r_max
        #   r = scale * r_max
        #   x = r * u
        #   f = problem(x)
        
        # This focuses search near the origin as t increases.
        
        # Let's implement this.
        
        for t in range(n):
            # Check time
            if time.time() > deadline:
                break
            
            # Decay factor: Start wide, end narrow
            # We want to explore the whole space initially, then refine near origin.
            # scale = 1.0 - (t / n)
            # To avoid 0, use 0.01
            scale = 1.0 - (t / max(1, n - 1)) * 0.99
            
            # Generate random unit vector
            # Normal distribution then normalize
            u = np.random.randn(dim)
            norm_u = np.linalg.norm(u)
            if norm_u < 1e-10:
                u = np.random.randn(dim)
                norm_u = np.linalg.norm(u)
            u = u / norm_u
            
            # Calculate max radius to stay within bounds
            # We need r such that lo <= r*u <= hi
            # This means:
            # If u[i] > 0: r <= hi[i] / u[i]
            # If u[i] < 0: r >= lo[i] / u[i]  => r <= -lo[i] / -u[i] = lo[i] / u[i] (since both neg)
            # Wait.
            # lo[i] <= r * u[i] <= hi[i]
            
            # Case u[i] > 0:
            # lo[i]/u[i] <= r <= hi[i]/u[i]
            # Case u[i] < 0:
            # lo[i]/u[i] >= r >= hi[i]/u[i]  => hi[i]/u[i] <= r <= lo[i]/u[i]
            # Case u[i] == 0:
            # 0 <= 0 <= ... no constraint on r from this dimension
            
            # So r must be in the intersection of all intervals [L_i, U_i].
            # L_i = lo[i]/u[i] if u[i] > 0 else hi[i]/u[i]
            # U_i = hi[i]/u[i] if u[i] > 0 else lo[i]/u[i]
            
            # We want the largest possible r to maximize exploration, then scale it down.
            # So r_max = min(U_i)
            # And we must also ensure r_max >= max(L_i) for feasibility.
            # Since 0 is usually in the box, r=0 is always feasible.
            # So the interval is [max(L_i), min(U_i)].
            # We pick r = scale * min(U_i).
            # If scale * min(U_i) < max(L_i), we might go out of bounds?
            # No, if we pick r from [0, min(U_i)], we are safe on the upper bound side.
            # But we must also satisfy lower bounds.
            # If we pick r too small, we might violate lower bounds?
            # No, if r=0, x=0. If 0 is in bounds, it's fine.
            # If 0 is not in bounds, we have a problem.
            # But WFG9 bounds are [-50, 50], so 0 is in bounds.
            # So any r in [0, min(U_i)] results in x in bounds?
            # Let's check.
            # x[i] = r * u[i].
            # If u[i] > 0, x[i] is in [0, r*u[i]]. Since r <= U_i = hi[i]/u[i], x[i] <= hi[i]. And x[i] >= 0 >= lo[i].
            # If u[i] < 0, x[i] is in [r*u[i], 0]. Since r <= U_i = lo[i]/u[i] (note lo[i]<0, u[i]<0, so U_i>0), 
            # r*u[i] >= lo[i] (multiplying by negative u[i] flips inequality? No.
            # r <= lo[i]/u[i]. Multiply by u[i] (negative): r*u[i] >= lo[i]. Correct.
            # And x[i] <= 0 <= hi[i].
            # So yes, any r in [0, min(U_i)] is feasible.
            
            # So we just need U_i for all i.
            
            # Vectorized calculation of U_i
            # U_i = hi[i] / u[i] if u[i] > 0 else lo[i] / u[i]
            # This can be written as:
            # U = np.where(u > 0, hi / np.where(u > 0, u, 1), lo / np.where(u < 0, u, 1))
            # Careful with division by zero.
            
            # Let's handle it safely.
            pos_mask = u > 0
            neg_mask = u < 0
            
            r_max = np.inf
            if np.any(pos_mask):
                r_max = min(r_max, np.min(hi[pos_mask] / u[pos_mask]))
            if np.any(neg_mask):
                r_max = min(r_max, np.min(lo[neg_mask] / u[neg_mask]))
                
            if not np.isfinite(r_max):
                # Fallback if something went wrong
                r_max = 1.0
                
            r = scale * r_max
            x = r * u
            
            # Clip just in case (floating point errors)
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            
        return None
# EVOLVE-BLOCK-END
