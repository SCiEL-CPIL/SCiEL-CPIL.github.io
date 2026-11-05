import time
import random
import numpy as np
from math import gcd

# Precompute pair indices
_PAIR_IU = []
_PAIR_JU = []
for i in range(16):
    for j in range(i+1, 16):
        _PAIR_IU.append(i)
        _PAIR_JU.append(j)
_PAIR_IU = np.array(_PAIR_IU, dtype=np.int32)
_PAIR_JU = np.array(_PAIR_JU, dtype=np.int32)
_NUM_PAIRS = len(_PAIR_IU)

def _line_key(i, j, xs, ys):
    """Compute canonical line key for pair (i,j)."""
    dx = xs[j] - xs[i]
    dy = ys[j] - ys[i]
    ax = abs(dx)
    ay = abs(dy)
    g = gcd(ax, ay)
    if g == 0:
        g = 1
    dxn = dx // g
    dyn = dy // g
    if dxn < 0 or (dxn == 0 and dyn < 0):
        dxn = -dxn
        dyn = -dyn
    c = dyn * xs[i] - dxn * ys[i]
    return (dxn, dyn, c)

def _evaluate_full(points):
    """Full evaluation. Returns (is_valid, score, line_counts_dict)."""
    n = len(points)
    if n != 16:
        return False, 0, None
    xs = np.array([p[0] for p in points], dtype=np.int64)
    ys = np.array([p[1] for p in points], dtype=np.int64)
    if np.any(np.abs(xs) > 1000) or np.any(np.abs(ys) > 1000):
        return False, 0, None
    if len(set(map(tuple, zip(xs.tolist(), ys.tolist())))) != 16:
        return False, 0, None
    
    line_counts = {}
    max_count = 0
    for k in range(_NUM_PAIRS):
        i = int(_PAIR_IU[k])
        j = int(_PAIR_JU[k])
        key = _line_key(i, j, xs, ys)
        line_counts[key] = line_counts.get(key, 0) + 1
        if line_counts[key] > max_count:
            max_count = line_counts[key]
    
    if max_count >= 10:
        return False, 0, None
    
    score = sum(1 for v in line_counts.values() if v == 6)
    return True, score, line_counts

def _evaluate_incremental(points, line_counts, moved_idx, old_x, old_y, new_x, new_y):
    """
    Incremental evaluation after moving point moved_idx.
    """
    if abs(new_x) > 1000 or abs(new_y) > 1000:
        return False, 0, None
    
    for i in range(16):
        if i == moved_idx:
            continue
        if points[i][0] == new_x and points[i][1] == new_y:
            return False, 0, None
    
    xs = np.array([p[0] for p in points], dtype=np.int64)
    ys = np.array([p[1] for p in points], dtype=np.int64)
    xs[moved_idx] = new_x
    ys[moved_idx] = new_y
    
    affected_pairs = []
    for k in range(_NUM_PAIRS):
        i = int(_PAIR_IU[k])
        j = int(_PAIR_JU[k])
        if i == moved_idx or j == moved_idx:
            affected_pairs.append(k)
    
    for k in affected_pairs:
        i = int(_PAIR_IU[k])
        j = int(_PAIR_JU[k])
        old_xs = xs.copy()
        old_ys = ys.copy()
        old_xs[moved_idx] = old_x
        old_ys[moved_idx] = old_y
        old_key = _line_key(i, j, old_xs, old_ys)
        if old_key in line_counts:
            line_counts[old_key] -= 1
            if line_counts[old_key] == 0:
                del line_counts[old_key]
    
    for k in affected_pairs:
        i = int(_PAIR_IU[k])
        j = int(_PAIR_JU[k])
        new_key = _line_key(i, j, xs, ys)
        line_counts[new_key] = line_counts.get(new_key, 0) + 1
    
    max_count = 0
    for v in line_counts.values():
        if v > max_count:
            max_count = v
    
    if max_count >= 10:
        return False, 0, None
    
    score = sum(1 for v in line_counts.values() if v == 6)
    return True, score, line_counts

def _local_search(points, max_time=10.0):
    """
    Runs a local search on the given points.
    Returns the best points found.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    current_points = [p[:] for p in points]
    is_valid, current_score, current_line_counts = _evaluate_full(current_points)
    if not is_valid:
        return points, 0
        
    best_points = [p[:] for p in current_points]
    best_score = current_score

    T = 2.0
    T_min = 0.01
    cool_rate = 0.99

    while time.time() < deadline:
        idx = random.randint(0, 15)
        old = current_points[idx]
        old_x, old_y = old[0], old[1]

        # Generate move
        r = random.random()
        if r < 0.4:
            nx = old_x + random.randint(-2, 2)
            ny = old_y + random.randint(-2, 2)
        elif r < 0.7:
            nx = old_x + random.randint(-5, 5)
            ny = old_y + random.randint(-5, 5)
        else:
            nx = random.randint(-100, 100)
            ny = random.randint(-100, 100)

        if abs(nx) > 1000 or abs(ny) > 1000:
            continue

        test_points = [p[:] for p in current_points]
        test_points[idx] = [nx, ny]
        
        is_valid, new_score, new_line_counts = _evaluate_incremental(
            test_points, current_line_counts, idx, old_x, old_y, nx, ny
        )

        if not is_valid:
            continue

        delta = new_score - current_score
        if delta >= 0:
            current_points[idx] = [nx, ny]
            current_score = new_score
            current_line_counts = new_line_counts
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            if random.random() < np.exp(delta / T):
                current_points[idx] = [nx, ny]
                current_score = new_score
                current_line_counts = new_line_counts

        T = max(T_min, T * cool_rate)
        
        # Restart if stuck
        if random.random() < 0.01:
            current_points = [p[:] for p in best_points]
            current_score = best_score
            _, _, current_line_counts = _evaluate_full(current_points)
            T = 2.0

    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Use a fixed seed for reproducibility to ensure we find the good solution
    random.seed(42)
    
    best_points_global = None
    best_score_global = -1

    # Try a few different initial configurations
    # 1. 4x4 Grid
    init_grid = [[x, y] for y in range(4) for x in range(4)]
    
    # 2. Affine transform of grid
    # ... generate a few
    
    inits = [init_grid]
    
    # Add a few random affine transforms
    for _ in range(5):
        A = np.array([[random.randint(-2, 2), random.randint(-2, 2)],
                       [random.randint(-2, 2), random.randint(-2, 2)]], dtype=np.int64)
        det = int(A[0,0]*A[1,1] - A[0,1]*A[1,0])
        if det == 0:
            continue
        base = np.array(init_grid, dtype=np.int64)
        t = A @ base.T
        if np.any(np.abs(t) > 900):
            continue
        pts = t.T.tolist()
        inits.append(pts)

    for init in inits:
        if time.time() > deadline - 5:
            break
        # Run local search with a time limit
        # Distribute time among inits
        time_per_init = min(20.0, (deadline - time.time()) / max(1, len(inits)))
        pts, score = _local_search(init, max_time=time_per_init)
        
        if score > best_score_global:
            best_score_global = score
            best_points_global = pts
            
        if best_score_global >= 15:
            break

    if best_points_global is None:
        return init_grid

    return best_points_global
