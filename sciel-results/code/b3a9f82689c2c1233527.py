"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import Counter

def _compute_areas(points):
    """Compute all non-zero doubled areas for all triples."""
    n = len(points)
    if n < 3:
        return np.array([], dtype=np.int64)
    pts = np.array(points, dtype=np.int64)
    X = pts[:, 0]
    Y = pts[:, 1]
    
    Xi = X[:, None, None]
    Yi = Y[:, None, None]
    Xj = X[None, :, None]
    Yj = Y[None, :, None]
    Xk = X[None, None, :]
    Yk = Y[None, None, :]
    
    det = Xi * (Yj - Yk) + Xj * (Yk - Yi) + Xk * (Yi - Yj)
    abs_det = np.abs(det)
    
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    
    areas = abs_det[mask]
    areas = areas[areas > 0]
    return areas

def _fast_score(points):
    """Vectorized computation of largest multiplicity of a single positive doubled area."""
    areas = _compute_areas(points)
    if len(areas) == 0:
        return 0
    counts = Counter(areas.tolist())
    return max(counts.values())

def _score_and_best_area(points):
    """Returns (max_multiplicity, best_area_value)"""
    areas = _compute_areas(points)
    if len(areas) == 0:
        return 0, 0
    counts = Counter(areas.tolist())
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

def _local_search(points, deadline, max_steps=50000):
    """Local search with fast scoring."""
    best_points = list(points)
    best_score, best_area = _score_and_best_area(best_points)
    
    n = len(best_points)
    point_set = set(best_points)
    
    steps = 0
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        idx = random.randint(0, n - 1)
        old_point = best_points[idx]
        
        found_improvement = False
        # Try a few random moves
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            new_point = (new_x, new_y)
            
            if new_point in point_set:
                continue
            
            best_points[idx] = new_point
            point_set.discard(old_point)
            point_set.add(new_point)
            
            new_score, new_best_area = _score_and_best_area(best_points)
            
            if new_score > best_score:
                best_score = new_score
                best_area = new_best_area
                found_improvement = True
            else:
                best_points[idx] = old_point
                point_set.discard(new_point)
                point_set.add(old_point)
        
        if not found_improvement:
            # If no improvement in last few steps, maybe restart or just continue
            pass
    
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
    deadline = start_time + 58  # 58s deadline, 2s safety margin before 60s kill
    
    best_points = None
    best_score = 0
    
    # Generate and evaluate candidates
    candidates = _generate_candidates()
    
    for pts in candidates:
        if time.time() > deadline - 20:  # Reserve 20s for local improvement
            break
        
        score = _fast_score(pts)
        if score > best_score:
            best_score = score
            best_points = list(pts)
    
    # Local improvement
    if best_points is not None and time.time() < deadline - 1:
        improved, s2 = _local_search(best_points, deadline - 1, max_steps=100000)
        if s2 > best_score:
            best_points, best_score = improved, s2
    
    # Fallback
    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]
    
    return best_points
# EVOLVE-BLOCK-END
