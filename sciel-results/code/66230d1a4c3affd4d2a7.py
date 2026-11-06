"""Number-theoretic construction: Grids of indices mapped to lattice points."""
import time
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 50.0 # Safety margin

    def evaluate_fast(points):
        """
        Vectorized evaluation of max multiplicity of positive doubled areas.
        points: list of 30 (x, y) tuples.
        """
        pts = np.array(points, dtype=np.int64)
        n = len(pts)
        
        # Generate all combinations of indices (i, j, k) with i < j < k
        # Using itertools is faster for setup, but numpy is faster for computation
        # For n=30, 4060 triples is small.
        
        # We can do this without itertools to avoid overhead, or just use it.
        # Let's use a simple loop to build arrays, then vectorize the det calculation.
        # Or even better:
        # i, j, k ranges
        # Since n is small, we can just compute all pairwise differences? No, that's O(N^2) memory, fine.
        
        # Method:
        # 1. Create indices for all triples
        # 2. Compute vectors
        # 3. Compute dets
        # 4. Count
        
        # Efficient way:
        # For each i, consider pairs (j, k) with j, k > i.
        # This avoids generating redundant permutations.
        
        counts = Counter()
        for i in range(n):
            pi = pts[i]
            # All points j > i
            rest = pts[i+1:]
            n_rest = len(rest)
            if n_rest < 2:
                continue
            
            # Compute vectors from pi to all points in rest
            # v_j = rest[j] - pi
            # We need det(v_j, v_k) for all j < k in rest
            # This is equivalent to det(rest[j] - pi, rest[k] - pi)
            
            # Let's compute the cross products for all pairs in rest relative to pi
            # This is O(N^2) per i, so O(N^3) total.
            # For N=30, 30 * (29*28/2) ~ 12000 ops. Very fast.
            
            # Vectorized for the inner loop:
            # We can compute all cross products of rest[j] with rest[k]?
            # No, they must share the same base point pi.
            
            # Let's just loop j, and vectorize k.
            for j in range(n_rest):
                pj = rest[j]
                v1 = pj - pi
                # v2 for all k > j
                # rest[j+1:]
                if j + 1 >= n_rest:
                    continue
                v2s = rest[j+1:] - pi
                
                # Cross product: v1[0]*v2[1] - v1[1]*v2[0]
                dets = np.abs(v1[0] * v2s[:, 1] - v1[1] * v2s[:, 0])
                
                # Filter positive
                dets = dets[dets > 0]
                if len(dets) > 0:
                    # Update counter
                    vals, cnts = np.unique(dets, return_counts=True)
                    for v, c in zip(vals, cnts):
                        counts[int(v)] += c
        
        if not counts:
            return 0
        return max(counts.values())

    def make_grid(a, b, ox=0, oy=0, scale=1):
        """Create a x b grid of points scaled by 'scale' and offset by (ox, oy)."""
        pts = []
        for i in range(a):
            for j in range(b):
                x = ox + i * scale
                y = oy + j * scale
                pts.append((x, y))
        return pts

    def valid(points):
        if len(points) != 30:
            return False
        if len(set(points)) != 30:
            return False
        for px, py in points:
            if not (-30 <= px <= 30 and -30 <= py <= 30):
                return False
        return True

    # Candidate Configurations
    # We want to maximize the frequency of a specific area.
    # Regular grids tend to have high frequency of small areas (like 1, 2, etc. in index space).
    # We will test standard grids and a few variations.
    
    candidates = []
    
    # 5x6 Grid
    candidates.append(make_grid(5, 6, ox=-10, oy=-10))
    # 6x5 Grid
    candidates.append(make_grid(6, 5, ox=-10, oy=-10))
    # 3x10 Grid
    candidates.append(make_grid(3, 10, ox=-10, oy=-10))
    # 10x3 Grid
    candidates.append(make_grid(10, 3, ox=-10, oy=-10))
    # 2x15 Grid
    candidates.append(make_grid(2, 15, ox=-10, oy=-10))
    # 15x2 Grid
    candidates.append(make_grid(15, 2, ox=-10, oy=-10))
    
    # Try a "hexagonal" packing approximation?
    # Points (i, j) -> (i + j//2, j) ?
    # Let's stick to simple lattices first as per directive "linear combination of two primitive lattice vectors".
    # The grids above are exactly that: u=(1,0), v=(0,1).
    
    # Let's also try rotated/scaled versions to see if bounds allow larger scales?
    # No, scale doesn't change multiplicity, only area value. Multiplicity is invariant under affine transformations that preserve the lattice structure (unimodular).
    # Scaling by integer k multiplies all areas by k^2. Multiplicity stays the same.
    # So the index structure is what matters.
    
    best_points = None
    best_score = 0
    
    for pts in candidates:
        if not valid(pts):
            continue
        score = evaluate_fast(pts)
        if score > best_score:
            best_score = score
            best_points = pts
            
    if best_points is None:
        # Fallback
        best_points = make_grid(5, 6, ox=0, oy=0)
        best_score = evaluate_fast(best_points)
        
    # Optional: Local Search
    # Given the timeout issue, we will be very conservative.
    # Only if we have plenty of time.
    
    if time.time() < deadline - 10:
        current = [list(p) for p in best_points]
        current_score = best_score
        
        # Simple hill climbing with strict time limit
        max_iters = 50
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            # Try moving each point by 1 unit in 4 directions
            for idx in range(30):
                if time.time() > deadline - 5:
                    break
                ox, oy = current[idx]
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nx, ny = ox + dx, oy + dy
                    if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                        continue
                    new_pts = [tuple(p) for p in current]
                    new_pts[idx] = (nx, ny)
                    if len(set(new_pts)) != 30:
                        continue
                    score = evaluate_fast(new_pts)
                    if score > current_score:
                        current[idx] = [nx, ny]
                        current_score = score
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
                
        if current_score > best_score:
            best_points = [tuple(p) for p in current]
            best_score = current_score

    return best_points
# EVOLVE-BLOCK-END
