import numpy as np
import time
import random
from itertools import product

def _score_vectorized(points):
    """
    Calculate the maximum multiplicity of a positive doubled area.
    Uses vectorized NumPy operations for speed.
    """
    points = np.array(points, dtype=np.int64)
    n = len(points)
    if n < 3:
        return 0
    
    # Compute all pairwise determinants (doubled areas)
    # For each pair (i, j), compute det(points[j] - points[i], points[k] - points[i]) for all k > j
    # Actually, simpler: compute all pairwise vectors and use broadcasting?
    # N=30 is small. N^3 = 27000 operations.
    # Let's do it with a loop over i, but vectorize j, k.
    
    max_count = 0
    # We can use a dictionary for counts
    # But creating a dict in a tight loop is slow.
    # Let's collect all areas in a list and use np.unique?
    # Or better, just accumulate counts for the specific areas we care about?
    # No, we want the max multiplicity of ANY area.
    
    # Efficient approach:
    # Generate all triples (i, j, k) with i < j < k.
    # There are C(30, 3) = 4060 triples.
    # This is small enough to do in a simple loop if the inner op is fast,
    # but let's try to vectorize the area calculation for all triples at once.
    
    # Indices for combinations of 3
    # i, j, k
    idx = np.indices((n, n, n))
    i_idx, j_idx, k_idx = idx
    
    # Mask for i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    
    # Extract points
    pi = points[i_idx[mask]]
    pj = points[j_idx[mask]]
    pk = points[k_idx[mask]]
    
    # Compute doubled area: | (xj - xi)(yk - yi) - (yj - yi)(xk - xi) |
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    
    areas = np.abs(dx1 * dy2 - dy1 * dx2)
    
    # Exclude zero areas
    areas = areas[areas > 0]
    
    if len(areas) == 0:
        return 0
        
    # Count frequencies
    # np.unique with return_counts is fast for small arrays
    unique_areas, counts = np.unique(areas, return_counts=True)
    
    return int(np.max(counts))

def _generate_lattice_points(n1, n2, u, v, offset=(0,0)):
    """
    Generate points for a lattice defined by basis u, v and dimensions n1, n2.
    Returns None if points are not distinct or out of bounds.
    """
    pts = []
    seen = set()
    for i in range(n1):
        for j in range(n2):
            x = offset[0] + i * u[0] + j * v[0]
            y = offset[1] + i * u[1] + j * v[1]
            if not (-30 <= x <= 30 and -30 <= y <= 30):
                return None
            p = (x, y)
            if p in seen:
                return None
            seen.add(p)
            pts.append(p)
    
    if len(pts) == 30:
        return pts
    return None

def _center_lattice(n1, n2, u, v):
    """
    Find the optimal offset to center the lattice in [-30, 30].
    Returns the offset tuple or None if it doesn't fit.
    """
    # Calculate bounding box of the lattice relative to (0,0)
    xs = [i * u[0] + j * v[0] for i in range(n1) for j in range(n2)]
    ys = [i * u[1] + j * v[1] for i in range(n1) for j in range(n2)]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    width = max_x - min_x
    height = max_y - min_y
    
    if width > 60 or height > 60:
        return None
        
    # We want to place [min_x, max_x] inside [-30, 30].
    # offset_x + min_x >= -30  => offset_x >= -30 - min_x
    # offset_x + max_x <= 30   => offset_x <= 30 - max_x
    # So offset_x must be in [-30 - min_x, 30 - max_x]
    
    min_off_x = -30 - min_x
    max_off_x = 30 - max_x
    
    if min_off_x > max_off_x:
        return None
        
    # Similarly for y
    min_off_y = -30 - min_y
    max_off_y = 30 - max_y
    
    if min_off_y > max_off_y:
        return None
        
    # Choose a valid offset, e.g., the center of the valid range
    off_x = (min_off_x + max_off_x) // 2
    off_y = (min_off_y + max_off_y) // 2
    
    # Verify validity with the chosen offset
    pts = _generate_lattice_points(n1, n2, u, v, (off_x, off_y))
    if pts:
        return (off_x, off_y)
    else:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_points = None
    best_score = 0
    
    # Dimensions for 30 points
    dims = [(6, 5), (5, 6), (10, 3), (3, 10), (15, 2), (2, 15)]
    
    # We will enumerate all primitive basis vectors (u, v) that could potentially fit.
    # To fit in 30x30, the vectors must be small.
    # Max coordinate extent is 30.
    # If n1=15, u can be roughly (2,0). If n1=6, u can be larger.
    # Let's bound the search space for u and v components to [-10, 10].
    # This is a safe upper bound for any lattice fitting in 30x30 with 30 points.
    # Actually, if u=(10,0) and n1=3, extent is 20. Fits.
    # If u=(10,0) and n1=6, extent is 50. Doesn't fit.
    # So the bound depends on n.
    # Let's just use a generous bound [-15, 15] and let the validity check filter.
    
    range_val = 15
    coords = list(range(-range_val, range_val + 1))
    
    # Pre-compute all valid (u, v) pairs that are primitive and non-collinear
    # Primitive: gcd(u_x, u_y) = 1 AND gcd(v_x, v_y) = 1? 
    # Actually, for the lattice to be "full rank" and not have holes that could be filled by smaller steps,
    # we usually want u and v to be primitive vectors. 
    # However, the problem doesn't strictly require the lattice to be the *finest* lattice.
    # But if u = 2*u', we could have used u' and n1/2 (if integer).
    # Since 30 is fixed, if we use a non-primitive vector, we are effectively using a coarser grid.
    # It is generally better to use primitive vectors to maximize the number of small areas.
    # But we should check all.
    
    # Optimization:
    # For each dimension pair (n1, n2), we iterate u and v.
    # To avoid duplicate checks (since swapping u,v and n1,n2 is symmetric in geometry but not in index),
    # we just iterate all.
    
    # To speed up, we can pre-filter u and v based on the bounding box constraint.
    # For a given n1, n2, u, v:
    # max_x = max(0, n1-1)*u[0] + max(0, n2-1)*v[0] + min(0, n1-1)*u[0] + min(0, n2-1)*v[0] ... 
    # Actually, the bounding box width is max(0, (n1-1)*u[0] + (n2-1)*v[0]) - min(0, (n1-1)*u[0] + (n2-1)*v[0]) ...
    # This is getting complicated. Let's just generate and check.
    # With range 15, there are 31^4 ~ 900k pairs. Too many to check all for all 6 dims if slow.
    # But _center_lattice is fast.
    
    # Let's refine the range.
    # For n=15, max step is 30/14 ~ 2. So range 3 is enough.
    # For n=6, max step is 30/5 = 6. So range 7 is enough.
    # For n=3, max step is 30/2 = 15. So range 16 is enough.
    
    # Strategy: Iterate over dims. For each dim, determine max possible vector component.
    # max_comp = 30 // (min(n1, n2) - 1) + 2 if min(n1, n2) > 1 else 30
    
    for n1, n2 in dims:
        if time.time() > deadline - 5:
            break
            
        min_dim = min(n1, n2)
        max_comp = 30 // (min_dim - 1) + 2 if min_dim > 1 else 30
        # Cap it at 15 for safety
        max_comp = min(max_comp, 15)
        
        coords = list(range(-max_comp, max_comp + 1))
        
        # Iterate u
        for ux in coords:
            for uy in coords:
                # Skip zero vector
                if ux == 0 and uy == 0:
                    continue
                # Iterate v
                for vx in coords:
                    for vy in coords:
                        if time.time() > deadline - 2:
                            break
                            
                        # Check non-collinear
                        det = ux * vy - uy * vx
                        if det == 0:
                            continue
                            
                        # Check primitive?
                        # If gcd(ux, uy) > 1, the lattice has "holes". 
                        # It's still a valid set of 30 points.
                        # But it's less likely to be optimal than a primitive one.
                        # We'll include all.
                        
                        # Try to center
                        offset = _center_lattice(n1, n2, (ux, uy), (vx, vy))
                        if offset is None:
                            continue
                            
                        pts = _generate_lattice_points(n1, n2, (ux, uy), (vx, vy), offset)
                        if pts is None:
                            continue
                            
                        # Score it
                        s = _score_vectorized(pts)
                        if s > best_score:
                            best_score = s
                            best_points = pts
                            # Optional: If we find a very high score, we might want to break early?
                            # Max possible is C(30,3) = 4060. Unlikely to reach that.
                            # But if we find a score that is theoretically max for this lattice type, we could stop.
                            # For now, keep searching.
                            
    # If no lattice was found (should not happen), fallback to a grid
    if best_points is None:
        # 6x5 grid
        pts = []
        for i in range(6):
            for j in range(5):
                pts.append((i, j))
        best_points = pts
        best_score = _score_vectorized(best_points)
        
    return best_points

# Quick sanity check / self-test if run directly
if __name__ == "__main__":
    import sys
    start = time.time()
    points = construct()
    end = time.time()
    print(f"Time taken: {end - start:.2f}s")
    print(f"Best Score: {_score_vectorized(points)}")
    print(f"Points: {points}")
