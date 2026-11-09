"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from collections import Counter
import random

def _score(points):
    """Compute the maximum multiplicity of a positive doubled area among all triples."""
    n = len(points)
    if n < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    # Vectorized computation of all triple determinants
    # For points p_i, p_j, p_k, doubled area = |det(p_j - p_i, p_k - p_i)|
    # = |x_i*(y_j - y_k) + x_j*(y_k - y_i) + x_k*(y_i - y_j)|
    
    best_mult = 0
    # Sample a subset of triples for speed if needed, but with 30 points it's 4060 triples
    # 4060 is small, so compute all
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                # doubled area = |ax(by-cy) + bx(cy-ay) + cx(ay-by)|
                d = abs(ax*(by-cy) + bx(cy-ay) + cx(ay-by))
                if d > 0:
                    areas.append(d)
    
    if not areas:
        return 0
    
    cnt = Counter(areas)
    return max(cnt.values())

def _generate_random_points(n=30, bound=30):
    """Generate n distinct random integer points in [-bound, bound]^2."""
    pts = set()
    while len(pts) < n:
        x = random.randint(-bound, bound)
        y = random.randint(-bound, bound)
        pts.add((x, y))
    return list(pts)

def _local_search(points, max_iters=5000, time_limit=None):
    """Local search: try to improve by moving one point at a time."""
    current = list(points)
    current_score = _score(current)
    start_time = time.time()
    
    for it in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        # Pick a random point to move
        idx = random.randint(0, len(current)-1)
        old_pt = current[idx]
        
        # Try several candidate new positions
        best_new_pt = None
        best_new_score = current_score
        
        # Generate candidate positions: nearby points
        candidates = set()
        x0, y0 = old_pt
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                nx, ny = x0 + dx, y0 + dy
                if -30 <= nx <= 30 and -30 <= ny <= 30:
                    if (nx, ny) != old_pt and (nx, ny) not in current:
                        candidates.add((nx, ny))
        
        # Also try some random distant points
        for _ in range(5):
            nx = random.randint(-30, 30)
            ny = random.randint(-30, 30)
            if (nx, ny) != old_pt and (nx, ny) not in current:
                candidates.add((nx, ny))
        
        for cand in candidates:
            current[idx] = cand
            s = _score(current)
            if s > best_new_score:
                best_new_score = s
                best_new_pt = cand
        
        if best_new_pt is not None:
            current[idx] = best_new_pt
            current_score = best_new_score
        else:
            current[idx] = old_pt
    
    return current, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second safety margin
    
    best_points = None
    best_score = 0
    
    # Multiple random restarts with local search
    num_restarts = 0
    while time.time() < deadline - 10:  # Leave 10s buffer
        num_restarts += 1
        
        # Generate initial random points
        pts = _generate_random_points(30, 30)
        s = _score(pts)
        
        if s > best_score:
            best_score = s
            best_points = list(pts)
        
        # Local search
        if time.time() < deadline - 5:
            time_limit = max(1, deadline - 5 - time.time())
            improved_pts, improved_score = _local_search(pts, max_iters=2000, time_limit=time_limit)
            if improved_score > best_score:
                best_score = improved_score
                best_points = list(improved_pts)
    
    if best_points is None:
        # Fallback: return the seed configuration
        return [(x, y) for x in range(6) for y in range(5)]
    
    return best_points

# EVOLVE-BLOCK-END
