"""
Optimized search for integral point sets in general position (no 3 collinear, no 4 concyclic).
Points are (x, y*sqrt(D)) with rational x,y. For D=1, points are (x,y) with integer coords.
All pairwise distances must be rational (integer for D=1 with integer coords).
"""
import math
import time
import random
from itertools import combinations

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _det4(a1,b1,c1,d1, a2,b2,c2,d2, a3,b3,c3,d3, a4,b4,c4,d4):
    """4x4 determinant."""
    return (a1*(b2*(c3*d4 - d3*c4) - c2*(b3*d4 - d3*b4) + d2*(b3*c4 - c3*b4)) -
            b1*(a2*(c3*d4 - d3*c4) - c2*(a3*d4 - d3*a4) + d2*(a3*c4 - c3*a4)) +
            c1*(a2*(b3*d4 - d3*b4) - b2*(a3*d4 - d3*a4) + d2*(a3*b4 - b3*a4)) -
            d1*(a2*(b3*c4 - c3*b4) - b2*(a3*c4 - c3*a4) + c2*(a3*b4 - b3*a4)))

def _check_new_point_valid(existing_points, new_pt):
    """Check if adding new_pt to existing_points maintains all constraints."""
    # Check distinctness
    for p in existing_points:
        if p == new_pt:
            return False
    
    # Check all pairwise distances are integers (for D=1, integer coords)
    for p in existing_points:
        dx = p[0] - new_pt[0]
        dy = p[1] - new_pt[1]
        ds = dx*dx + dy*dy
        if not _is_square(ds):
            return False
    
    # Check no three collinear
    for i in range(len(existing_points)):
        for j in range(i+1, len(existing_points)):
            ax, ay = existing_points[i]
            bx, by = existing_points[j]
            cx, cy = new_pt
            if ax*(by - cy) + bx*(cy - ay) + cx*(ay - by) == 0:
                return False
    
    # Check no four concyclic
    for i in range(len(existing_points)):
        for j in range(i+1, len(existing_points)):
            for k in range(j+1, len(existing_points)):
                ax, ay = existing_points[i]
                bx, by = existing_points[j]
                cx, cy = existing_points[k]
                dx, dy = new_pt
                a2 = ax*ax + ay*ay
                b2 = bx*bx + by*by
                c2 = cx*cx + cy*cy
                d2 = dx*dx + dy*dy
                det_val = _det4(a2, ax, ay, 1,
                                b2, bx, by, 1,
                                c2, cx, cy, 1,
                                d2, dx, dy, 1)
                if det_val == 0:
                    return False
    
    return True

def _generate_integer_distance_candidates(center, max_radius, max_attempts=5000):
    """Generate random integer points at integer distance from center."""
    cx, cy = center
    candidates = []
    seen = set()
    for _ in range(max_attempts):
        # Random direction and integer distance
        r = random.randint(1, max_radius)
        # Find a point at distance r from (cx, cy)
        # (x-cx)^2 + (y-cy)^2 = r^2
        # Try random dx, compute dy
        for _ in range(20):
            dx = random.randint(-r, r)
            dy_sq = r*r - dx*dx
            if dy_sq < 0:
                continue
            if _is_square(dy_sq):
                dy = math.isqrt(dy_sq)
                for s in ([1, -1] if dy != 0 else [1]):
                    pt = (cx + dx, cy + dy*s)
                    if pt not in seen:
                        seen.add(pt)
                        candidates.append(pt)
                        break
                break
    return candidates

def _try_add_point(current_points, deadline):
    """Try to add one more point to current_points."""
    # Compute bounding box
    xs = [p[0] for p in current_points]
    ys = [p[1] for p in current_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y, 10)
    
    # Generate candidates
    candidates = set()
    
    # Strategy 1: Points at integer distance from existing points
    for p in current_points:
        for r in range(1, min(200, span + 50)):
            # Find integer solutions to (x-p[0])^2 + (y-p[1])^2 = r^2
            for dx in range(-r, r+1):
                dy_sq = r*r - dx*dx
                if dy_sq < 0:
                    continue
                if _is_square(dy_sq):
                    dy = math.isqrt(dy_sq)
                    for s in ([1, -1] if dy != 0 else [1]):
                        pt = (p[0] + dx, p[1] + dy*s)
                        candidates.add(pt)
    
    # Strategy 2: Random integer points in expanded region
    for _ in range(3000):
        x = random.randint(min_x - span, max_x + span)
        y = random.randint(min_y - span, max_y + span)
        candidates.add((x, y))
    
    # Filter candidates: must be at integer distance from ALL existing points
    valid_candidates = []
    for pt in candidates:
        ok = True
        for p in current_points:
            dx = p[0] - pt[0]
            dy = p[1] - pt[1]
            if not _is_square(dx*dx + dy*dy):
                ok = False
                break
        if ok:
            valid_candidates.append(pt)
    
    # Try to add one that maintains general position
    random.shuffle(valid_candidates)
    for pt in valid_candidates:
        if time.time() > deadline:
            break
        if _check_new_point_valid(current_points, pt):
            return pt
    
    return None

def _local_search_optimize(points, deadline):
    """Try to swap points to find a better configuration."""
    current = list(points)
    best = list(points)
    best_size = len(current)
    
    while time.time() < deadline:
        # Try adding a point
        if len(current) < 12:
            new_pt = _try_add_point(current, deadline)
            if new_pt is not None:
                current.append(new_pt)
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
                continue
        
        # Try removing a point and re-adding to find a different configuration
        if len(current) > 3:
            # Remove a random point
            idx = random.randint(0, len(current) - 1)
            removed = current.pop(idx)
            # Try to add a different point
            new_pt = _try_add_point(current, deadline)
            if new_pt is not None and new_pt != removed:
                current.append(new_pt)
            else:
                current.append(removed)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Start with known 5-point configuration
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    current = list(base_points)
    best = list(base_points)
    best_size = len(current)
    
    # Verify base points
    for pt in current:
        for other in current:
            if pt != other:
                dx = pt[0] - other[0]
                dy = pt[1] - other[1]
                if not _is_square(dx*dx + dy*dy):
                    # Base points invalid, start fresh
                    current = [(0, 0), (3, 4), (0, 5)]
                    best = list(current)
                    best_size = len(current)
                    break
    
    # Iteratively try to add points
    for target in range(len(current) + 1, 13):
        if time.time() > deadline:
            break
        
        # Try multiple times to add a point
        added = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            new_pt = _try_add_point(current, deadline)
            if new_pt is not None:
                current.append(new_pt)
                added = True
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
                break
        
        if not added and attempt >= 49:
            # Try local search to find a different configuration
            if time.time() < deadline - 10:
                optimized = _local_search_optimize(current, deadline - 5)
                if len(optimized) > best_size:
                    best = list(optimized)
                    best_size = len(optimized)
                current = list(optimized)
    
    # Final check and return
    result = best if len(best) >= 3 else current
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in result]}
