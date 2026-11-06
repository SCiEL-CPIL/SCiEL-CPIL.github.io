"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import Counter

def _fast_score(points):
    """Vectorized computation of largest multiplicity of a single positive doubled area."""
    n = len(points)
    if n < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    x1 = pts[:, 0]
    y1 = pts[:, 1]
    
    # Compute all triple determinants using vectorization
    # For each pair (i,j), compute cross product with all k > j
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            # Vectorized: compute det for all k > j
            x2, y2 = pts[j]
            x3 = pts[j+1:, 0]
            y3 = pts[j+1:, 1]
            dets = np.abs(x1[i] * (y2 - y3) + x2 * (y3 - y1[i]) + x3 * (y1[i] - y2))
            nonzero = dets[dets > 0]
            if len(nonzero) > 0:
                areas.extend(nonzero.tolist())
    
    if not areas:
        return 0
    counts = Counter(areas)
    return max(counts.values())

def _score_and_best_area(points):
    """Returns (max_multiplicity, best_area_value)"""
    n = len(points)
    if n < 3:
        return 0, 0
    pts = np.array(points, dtype=np.int64)
    x1 = pts[:, 0]
    y1 = pts[:, 1]
    
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            x2, y2 = pts[j]
            x3 = pts[j+1:, 0]
            y3 = pts[j+1:, 1]
            dets = np.abs(x1[i] * (y2 - y3) + x2 * (y3 - y1[i]) + x3 * (y1[i] - y2))
            nonzero = dets[dets > 0]
            if len(nonzero) > 0:
                areas.extend(nonzero.tolist())
    
    if not areas:
        return 0, 0
    counts = Counter(areas)
    best_area, best_count = counts.most_common(1)[0]
    return best_count, best_area

def _inbounds(points, limit=30):
    for x, y in points:
        if x < -limit or x > limit or y < -limit or y > limit:
            return False
    return True

def _build_grid(w, h, x0=0, y0=0):
    return [(x0 + x, y0 + y) for x in range(w) for y in range(h)]

def _build_parallelogram(w, h, sx, sy, tx, ty, x0=0, y0=0):
    """Build a parallelogram grid: (x0 + i*sx + j*tx, y0 + i*sy + j*ty)"""
    pts = []
    for i in range(w):
        for j in range(h):
            pts.append((x0 + i * sx + j * tx, y0 + i * sy + j * ty))
    return pts

def _local_search(points, deadline, max_steps=20000):
    """Local search with targeted area-boosting strategy."""
    start_time = time.time()
    best_points = list(points)
    best_score, best_area = _score_and_best_area(best_points)
    
    n = len(best_points)
    point_set = set(best_points)
    
    steps = 0
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        # Strategy 1: Try to boost the current best area
        idx = random.randint(0, n - 1)
        old_point = best_points[idx]
        
        # Try several candidate moves
        found_improvement = False
        for attempt in range(8):
            if time.time() > deadline:
                break
            
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            new_point = (new_x, new_y)
            
            if new_point in point_set:
                continue
            
            # Temporarily move point
            best_points[idx] = new_point
            point_set.discard(old_point)
            point_set.add(new_point)
            
            new_score, new_best_area = _score_and_best_area(best_points)
            
            if new_score > best_score:
                best_score = new_score
                best_area = new_best_area
                found_improvement = True
            else:
                # Revert
                best_points[idx] = old_point
                point_set.discard(new_point)
                point_set.add(old_point)
        
        if found_improvement:
            # After improvement, do a few more targeted moves
            for _ in range(3):
                if time.time() > deadline:
                    break
                idx2 = random.randint(0, n - 1)
                old2 = best_points[idx2]
                for _ in range(5):
                    nx = random.randint(-30, 30)
                    ny = random.randint(-30, 30)
                    np2 = (nx, ny)
                    if np2 in point_set:
                        continue
                    best_points[idx2] = np2
                    point_set.discard(old2)
                    point_set.add(np2)
                    s, a = _score_and_best_area(best_points)
                    if s > best_score:
                        best_score = s
                        best_area = a
                    else:
                        best_points[idx2] = old2
                        point_set.discard(np2)
                        point_set.add(old2)
    
    return best_points, best_score

def _generate_candidates():
    """Generate diverse candidate configurations."""
    candidates = []
    
    # Rectangular grids
    grid_configs = [(6, 5), (5, 6), (10, 3), (3, 10), (15, 2), (2, 15)]
    for w, h in grid_configs:
        for x0 in range(-30, 31 - w + 1):
            for y0 in range(-30, 31 - h + 1):
                pts = _build_grid(w, h, x0, y0)
                if _inbounds(pts):
                    candidates.append(pts)
    
    # Parallelogram patterns
    for w, h in [(6, 5), (5, 6)]:
        for sx in [1, 2]:
            for sy in [0, 1]:
                for tx in [-1, 0, 1]:
                    for ty in [0, 1, 2]:
                        if sx * ty - sy * tx == 0:  # Degenerate
                            continue
                        for x0 in range(-15, 16):
                            for y0 in range(-15, 16):
                                pts = _build_parallelogram(w, h, sx, sy, tx, ty, x0, y0)
                                if _inbounds(pts):
                                    candidates.append(pts)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s deadline, 5s safety margin before 60s kill
    
    best_points = None
    best_score = 0
    
    # Generate and evaluate candidates
    candidates = _generate_candidates()
    
    for pts in candidates:
        if time.time() > deadline - 15:  # Reserve time for local improvement
            break
        
        score = _fast_score(pts)
        if score > best_score:
            best_score = score
            best_points = list(pts)
    
    # Local improvement
    if best_points is not None and time.time() < deadline - 2:
        improved, s2 = _local_search(best_points, deadline - 2, max_steps=50000)
        if s2 > best_score:
            best_points, best_score = improved, s2
    
    # Fallback
    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]
    
    return best_points
# EVOLVE-BLOCK-END
