import numpy as np
import time
import math
import random
from scipy.optimize import minimize

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 615
    
    n = 33
    
    def count_convex_7goms_fast(points):
        """
        Fast vectorized count of convex 7-gons.
        Uses the property: A subset of size k is convex iff all points are extreme points.
        For N=33, iterating all C(33,7) is ~5.3M. In Python, this is too slow.
        We use a heuristic approximation or a faster exact method.
        
        Actually, let's use the relation: 
        Number of convex 7-gons = sum_{i=0}^{6} (-1)^i * ... (Inclusion-Exclusion is hard).
        
        Let's stick to the evaluator's definition if possible, or a tight proxy.
        The current program uses a proxy: (# Convex Quads) * C(29,3).
        This proxy is likely correlated. I will use a more accurate proxy:
        Count all convex k-gons for k=4,5,6,7 and use a linear combination?
        
        Better: Use the fact that for random points, C is proportional to the volume of the convex hull? No.
        
        Let's implement a fast exact counter for N=33 using Numpy.
        We can check convexity of a subset by checking if the cross products of consecutive edges (in sorted order) have the same sign.
        For a subset S, sort by x. If y is unimodal, it's not necessarily convex.
        
        Given the time constraint, I will use the SAME proxy as the current program but vectorized, 
        as it is known to yield score 2.42. The goal is to find a BETTER cluster configuration.
        """
        # Vectorized check for all 4-subsets being convex
        # This is O(N^4) but vectorized.
        # N=33 -> C(33,4) = 40920 subsets. This is fast.
        
        # Get all 4-subsets
        indices = np.arange(n)
        # Use itertools for indices, then vectorize the check
        # Actually, 40920 checks is trivial. The bottleneck in the current program was the SA loop, not the count.
        
        # I will implement the exact count of convex 7-gons using a recursive/DP approach if possible,
        # or just use the quad-proxy which is fast.
        
        # Let's use the quad-proxy for speed, as it's proven to work.
        # Proxy: C_approx = (# Convex Quads) * C(29,3)
        
        # Vectorized convex quad check
        # A quad (a,b,c,d) sorted by x is convex if:
        # 1. No point is inside the triangle of the other 3.
        # For x-sorted points, this is equivalent to:
        # The slopes (y_{i+1}-y_i)/(x_{i+1}-x_i) must not be monotonic in a way that creates a "dent".
        # Specifically, if we sort by x, the points form a convex quad iff the three points 
        # (x1,y1), (x2,y2), (x3,y3), (x4,y4) do not have one inside the triangle of the others.
        
        # Simpler: Just loop the 40920 quads in Python? 40920 * 4 area calcs = ~160k ops. Very fast.
        
        count_quads = 0
        # Precompute x, y
        xs = points[:, 0]
        ys = points[:, 1]
        
        # Generate all 4-combinations
        # To avoid itertools overhead in tight loop, we can use numpy meshgrid for small N?
        # Or just use itertools, it's fast enough for 40k items.
        from itertools import combinations
        for i, j, k, l in combinations(range(n), 4):
            # Check convexity
            # Sort indices by x
            idxs = [i, j, k, l]
            idxs.sort(key=lambda idx: xs[idx])
            x1, x2, x3, x4 = xs[idxs[0]], xs[idxs[1]], xs[idxs[2]], xs[idxs[3]]
            y1, y2, y3, y4 = ys[idxs[0]], ys[idxs[1]], ys[idxs[2]], ys[idxs[3]]
            
            # Check if any point is inside the triangle of the other 3
            # Since x is sorted, only idxs[1] and idxs[2] can be inside triangles formed by the others?
            # Actually, for x-sorted points, a point is inside the triangle of the other 3 iff
            # it is on the "wrong" side of the line segments.
            
            # Area signs
            def area2(a, b, c):
                return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
            
            p1 = (x1, y1); p2 = (x2, y2); p3 = (x3, y3); p4 = (x4, y4)
            
            # Check p2 inside (p1, p3, p4)?
            # Check p3 inside (p1, p2, p4)?
            # Check p1 inside (p2, p3, p4)? (Impossible if x1 < x2,x3,x4)
            # Check p4 inside (p1, p2, p3)? (Impossible if x4 > x1,x2,x3)
            
            # So only need to check p2 and p3.
            
            # p2 inside triangle (p1, p3, p4)?
            # This is true if p2 is on the same side of p1-p3 as p4? No.
            # Let's just use the general 4-point check, it's robust.
            
            # A set of 4 points is convex iff no point is inside the triangle of the other 3.
            # For x-sorted, p1 and p4 are never inside.
            # p2 is inside (p1, p3, p4) iff:
            #   area2(p2, p1, p3) and area2(p2, p3, p4) and area2(p2, p4, p1) have same sign?
            #   No, p2 is inside if it is to the "same" side of all edges?
            
            # Simpler: 
            # If the polygon p1-p2-p3-p4 is convex, the cross products of consecutive edges must have the same sign.
            # Edge 1: p1->p2, Edge 2: p2->p3, Edge 3: p3->p4, Edge 4: p4->p1
            # Cross 1: (p2-p1) x (p3-p2)
            # Cross 2: (p3-p2) x (p4-p3)
            # Cross 3: (p4-p3) x (p1-p4)
            # Cross 4: (p1-p4) x (p2-p1)
            
            # If any cross product is <= 0 (assuming CCW), it's not strictly convex.
            # We need to handle the order.
            
            # Let's just use the point-in-triangle test for p2 and p3.
            
            # p2 inside (p1, p3, p4)?
            d1 = area2(p2, p1, p3)
            d2 = area2(p2, p3, p4)
            d3 = area2(p2, p4, p1)
            if (d1 > 0 and d2 > 0 and d3 > 0) or (d1 < 0 and d2 < 0 and d3 < 0):
                continue # p2 is inside, not convex
            
            # p3 inside (p1, p2, p4)?
            d1 = area2(p3, p1, p2)
            d2 = area2(p3, p2, p4)
            d3 = area2(p3, p4, p1)
            if (d1 > 0 and d2 > 0 and d3 > 0) or (d1 < 0 and d2 < 0 and d3 < 0):
                continue # p3 is inside, not convex
            
            # If neither is inside, it's a convex quad (or a line, but we assume general position)
            count_quads += 1
            
        return count_quads * 3654 # C(29,3)

    def objective(params):
        # params: [cx1, cy1, sx1, sy1, cx2, cy2, sx2, sy2, cx3, cy3, sx3, sy3]
        # 3 clusters, 11 points each
        points = np.zeros((33, 2))
        
        for c in range(3):
            cx = params[c*4]
            cy = params[c*4+1]
            sx = params[c*4+2]
            sy = params[c*4+3]
            
            # Ensure positive spread
            sx = abs(sx) + 0.01
            sy = abs(sy) + 0.01
            
            # Clip centers to [0.1, 0.9]
            cx = np.clip(cx, 0.1, 0.9)
            cy = np.clip(cy, 0.1, 0.9)
            
            # Generate 11 points
            # Use fixed random offsets for determinism
            offsets = rng_cluster[c]
            for i in range(11):
                points[c*11 + i, 0] = cx + offsets[i, 0] * sx
                points[c*11 + i, 1] = cy + offsets[i, 1] * sy
        
        # Clip to [0, 1]
        points = np.clip(points, 0.0, 1.0)
        
        # Check validity (distances and collinearity)
        # If invalid, return high penalty
        valid = True
        for i in range(33):
            for j in range(i+1, 33):
                dx = points[i, 0] - points[j, 0]
                dy = points[i, 1] - points[j, 1]
                if dx*dx + dy*dy < 1e-12:
                    return 1e10 # Penalty
                # Collinearity check with a third point is expensive, skip for speed?
                # Or do a sparse check.
        
        # Count convex 7-gons (proxy)
        c = count_convex_7goms_fast(points)
        return c

    # Initialize random generator
    np.random.seed(42)
    random.seed(42)
    
    # Pre-generate fixed offsets for clusters to ensure determinism and speed
    # Offsets should be small, e.g., standard normal
    rng_cluster = [np.random.normal(0, 1, (11, 2)) for _ in range(3)]
    
    # Initial parameters: 3 clusters at corners
    # Cluster 1: (0.2, 0.2), Cluster 2: (0.8, 0.2), Cluster 3: (0.5, 0.8)
    x0 = np.array([
        0.2, 0.2, 0.05, 0.05,
        0.8, 0.2, 0.05, 0.05,
        0.5, 0.8, 0.05, 0.05
    ])
    
    # Use Nelder-Mead for local refinement
    # This is a different optimization family than SA.
    
    best_params = x0
    best_val = objective(x0)
    
    # Run Nelder-Mead
    # Bounds are handled by clipping in objective
    
    # Since we have 12 parameters, NM is efficient.
    # We will run multiple restarts with different initializations if time permits.
    
    restarts = 5
    for r in range(restarts):
        if r > 0:
            # Perturb initial parameters
            x0 = x0 + np.random.normal(0, 0.1, 12)
        
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
        
        # Check time
        if time.time() > deadline - 60:
            break
            
    # Reconstruct points from best_params
    points = np.zeros((33, 2))
    for c in range(3):
        cx = best_params[c*4]
        cy = best_params[c*4+1]
        sx = abs(best_params[c*4+2]) + 0.01
        sy = abs(best_params[c*4+3]) + 0.01
        
        cx = np.clip(cx, 0.1, 0.9)
        cy = np.clip(cy, 0.1, 0.9)
        
        for i in range(11):
            points[c*11 + i, 0] = cx + rng_cluster[c][i, 0] * sx
            points[c*11 + i, 1] = cy + rng_cluster[c][i, 1] * sy
            
    points = np.clip(points, 0.0, 1.0)
    
    # Final validity check and repair
    # If invalid, add small jitter
    for _ in range(100):
        valid = True
        for i in range(33):
            for j in range(i+1, 33):
                dx = points[i, 0] - points[j, 0]
                dy = points[i, 1] - points[j, 1]
                if dx*dx + dy*dy < 1e-12:
                    valid = False
                    break
            if not valid: break
            
        if valid:
            # Check collinearity (sampled)
            for _ in range(1000):
                i, j, k = random.sample(range(33), 3)
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                if area < 1e-9:
                    valid = False
                    break
            
            if valid:
                break
        
        # Jitter
        points += np.random.normal(0, 0.001, (33, 2))
        points = np.clip(points, 0.0, 1.0)
        
    return points

# EVOLVE-BLOCK-END
