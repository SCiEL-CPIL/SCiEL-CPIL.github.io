"""Heilbronn 11 triangles: Combinatorial Nudge Search with Symmetric Initialization."""
import numpy as np
import time
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4
DEADLINE = 145

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

def _bary_to_cart(bary):
    """Convert barycentric coordinates (u, v) to Cartesian (x, y).
    Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2).
    Point = u*(0,0) + v*(1,0) + (1-u-v)*(0.5, S3H)
    x = v + 0.5*(1-u-v) = 0.5 - 0.5u + 0.5v
    y = S3H*(1-u-v)
    """
    u = bary[:, 0]
    v = bary[:, 1]
    x = 0.5 - 0.5 * u + 0.5 * v
    y = S3H * (1.0 - u - v)
    return np.column_stack([x, y])

def _cart_to_bary(pts):
    """Convert Cartesian to barycentric coordinates."""
    x = pts[:, 0]
    y = pts[:, 1]
    u = 1.0 - x - y / (2.0 * S3H)
    v = x - y / (2.0 * S3H)
    return np.column_stack([u, v])

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)

def _project_to_triangle(pts):
    """Project points into the equilateral triangle."""
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts.copy())
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # Symmetric initial configuration
    # This is a known good starting point for n=11
    init = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1/3, S3H/3],
        [1/6, S3H/6],
        [5/6, S3H/6],
        [1/3, 2*S3H/3],
        [2/3, 2*S3H/3],
    ], dtype=float)
    _update_best(init)
    
    # Also try a few other symmetric/random starts
    for _ in range(5):
        # Random points in triangle
        r = rng.random((N, 2))
        r2 = rng.random((N, 2))
        r2 = 1 - r2
        # Sample from triangle: sqrt(r1) * v2, etc.
        # Simpler: rejection sampling or barycentric random
        bary = rng.random((N, 2))
        s = bary.sum(axis=1, keepdims=True)
        bary = bary / s
        # bary is (u, v) corresponding to weights for (1,0) and (0.5, S3H)?
        # My _bary_to_cart uses:
        # x = 0.5 - 0.5u + 0.5v
        # y = S3H*(1-u-v)
        # This corresponds to weights: 
        # w0 (for 0,0) = 1-u-v
        # w1 (for 1,0) = v
        # w2 (for 0.5, S3H) = u
        # So u,v,w >= 0 and sum=1.
        # My random bary above has u,v >= 0 and u+v <= 1.
        # This is valid.
        pts = _bary_to_cart(bary)
        _update_best(pts)

    # Combinatorial Nudge Search
    # Strategy: 
    # 1. Identify the triplet of points forming the minimum area triangle.
    # 2. Randomly select one of the 3 points in that triplet.
    # 3. Randomly select a new position for that point within the triangle.
    # 4. If the new global minimum area is larger, accept.
    # 5. Repeat.
    
    # We will run this for multiple "restarts" or just one long run?
    # Given the deadline, a long run with occasional random restarts of the nudge process 
    # (or just continuing) is good.
    # To escape local optima, we can also occasionally do a "kick" - move a random point 
    # significantly even if it worsens the score, then continue nudging.
    
    current_pts = best_pts.copy()
    current_area = best_area
    
    # Precompute indices for speed
    # We need to find the min area triplet quickly.
    # _min_triangle_area is vectorized, so it's fast enough (~165 ops).
    
    while _elapsed_ok(10):
        # Find the worst triplet
        pi = current_pts[_T_I]
        pj = current_pts[_T_J]
        pk = current_pts[_T_K]
        dx1 = pj[:, 0] - pi[:, 0]
        dy1 = pj[:, 1] - pi[:, 1]
        dx2 = pk[:, 0] - pi[:, 0]
        dy2 = pk[:, 1] - pi[:, 1]
        cross = dx1 * dy2 - dy1 * dx2
        areas = np.abs(cross) * 0.5
        
        min_idx = np.argmin(areas)
        min_area_val = areas[min_idx]
        
        # Get the 3 points involved
        i, j, k = _TRIPLETS[min_idx]
        
        # Randomly pick one of the 3 to move
        move_idx = rng.choice([i, j, k])
        
        # Generate a random new position in the triangle
        # Use barycentric coordinates for uniform sampling
        bary_new = rng.random(2)
        s = bary_new.sum()
        bary_new = bary_new / s
        new_pt = _bary_to_cart(bary_new.reshape(1, 2))[0]
        
        # Try the move
        old_pt = current_pts[move_idx]
        current_pts[move_idx] = new_pt
        
        new_min_area = _min_triangle_area(current_pts)
        
        if new_min_area > current_area:
            current_area = new_min_area
            _update_best(current_pts)
        else:
            # Revert
            current_pts[move_idx] = old_pt
            
        # Occasionally perform a "kick" to escape local optima
        # This increases exploration
        if rng.random() < 0.05:
            # Move a random point to a random location
            kick_idx = rng.integers(0, N)
            bary_kick = rng.random(2)
            s_k = bary_kick.sum()
            bary_kick = bary_kick / s_k
            kick_pt = _bary_to_cart(bary_kick.reshape(1, 2))[0]
            
            old_kick_pt = current_pts[kick_idx]
            current_pts[kick_idx] = kick_pt
            kick_area = _min_triangle_area(current_pts)
            
            # Accept kick if it's not terrible, or just accept to explore
            # Standard simulated annealing would use temperature, but we don't have T.
            # Let's accept if it's within 20% of current best, or just accept always 
            # and hope nudging recovers? 
            # Better: Accept if it improves, or with small probability if it's close.
            # For simplicity, let's only accept kicks if they improve, 
            # but do more frequent kicks? 
            # Actually, the "nudge" itself is a local search. 
            # A kick is a global move.
            # Let's accept kicks only if they improve the score, otherwise revert.
            # This makes the kick useless for escaping.
            
            # Alternative: If we have time, we can do a "random restart" 
            # of the nudge search from a perturbed version of the best point.
            pass # Keep it simple for now, rely on the nudge logic.

    # If we have time, do a few more random restarts from the best point
    # to see if we can find a better basin
    if _elapsed_ok(20):
        for _ in range(10):
            if not _elapsed_ok(15):
                break
            # Perturb best point
            perturbed = best_pts + rng.normal(0, 0.1, size=(N, 2))
            perturbed = _project_to_triangle(perturbed)
            p_area = _min_triangle_area(perturbed)
            
            if p_area < best_area * 0.8: # Too bad, skip
                continue
                
            # Run a short nudge search from this perturbed point
            cur_pts = perturbed.copy()
            cur_area = p_area
            local_best_pts = cur_pts.copy()
            local_best_area = cur_area
            
            # Short nudge loop
            nudge_time_limit = 5.0
            nudge_start = time.time()
            while (time.time() - nudge_start) < nudge_time_limit:
                pi = cur_pts[_T_I]
                pj = cur_pts[_T_J]
                pk = cur_pts[_T_K]
                dx1 = pj[:, 0] - pi[:, 0]
                dy1 = pj[:, 1] - pi[:, 1]
                dx2 = pk[:, 0] - pi[:, 0]
                dy2 = pk[:, 1] - pi[:, 1]
                cross = dx1 * dy2 - dy1 * dx2
                areas = np.abs(cross) * 0.5
                
                min_idx = np.argmin(areas)
                min_area_val = areas[min_idx]
                i, j, k = _TRIPLETS[min_idx]
                move_idx = rng.choice([i, j, k])
                
                bary_new = rng.random(2)
                s = bary_new.sum()
                bary_new = bary_new / s
                new_pt = _bary_to_cart(bary_new.reshape(1, 2))[0]
                
                old_pt = cur_pts[move_idx]
                cur_pts[move_idx] = new_pt
                new_min_area = _min_triangle_area(cur_pts)
                
                if new_min_area > cur_area:
                    cur_area = new_min_area
                    if cur_area > local_best_area:
                        local_best_area = cur_area
                        local_best_pts = cur_pts.copy()
                else:
                    cur_pts[move_idx] = old_pt
                    
            _update_best(local_best_pts)

    if best_pts is None:
        pts = rng.random((N, 2))
        for i in range(N):
            u, v = pts[i]
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            x = u + 0.5 * v
            y = S3H * v
            pts[i] = [x, y]
        best_pts = _project_to_triangle(pts)

    return best_pts
