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
    Incremental evaluation after moving point moved_idx from (old_x, old_y) to (new_x, new_y).
    Returns (is_valid, score, new_line_counts) or (False, 0, None) if invalid.
    """
    # Check bounds
    if abs(new_x) > 1000 or abs(new_y) > 1000:
        return False, 0, None
    
    # Check distinctness: new point must not equal any other point
    for i in range(16):
        if i == moved_idx:
            continue
        if points[i][0] == new_x and points[i][1] == new_y:
            return False, 0, None
    
    # Build updated xs, ys arrays
    xs = np.array([p[0] for p in points], dtype=np.int64)
    ys = np.array([p[1] for p in points], dtype=np.int64)
    xs[moved_idx] = new_x
    ys[moved_idx] = new_y
    
    # Identify all pairs involving moved_idx
    affected_pairs = []
    for k in range(_NUM_PAIRS):
        i = int(_PAIR_IU[k])
        j = int(_PAIR_JU[k])
        if i == moved_idx or j == moved_idx:
            affected_pairs.append(k)
    
    # Remove old line contributions for affected pairs
    for k in affected_pairs:
        i = int(_PAIR_IU[k])
        j = int(_PAIR_JU[k])
        # Use old coordinates for this point
        old_xs = xs.copy()
        old_ys = ys.copy()
        old_xs[moved_idx] = old_x
        old_ys[moved_idx] = old_y
        old_key = _line_key(i, j, old_xs, old_ys)
        if old_key in line_counts:
            line_counts[old_key] -= 1
            if line_counts[old_key] == 0:
                del line_counts[old_key]
    
    # Add new line contributions for affected pairs
    for k in affected_pairs:
        i = int(_PAIR_IU[k])
        j = int(_PAIR_JU[k])
        new_key = _line_key(i, j, xs, ys)
        line_counts[new_key] = line_counts.get(new_key, 0) + 1
    
    # Check max count
    max_count = 0
    for v in line_counts.values():
        if v > max_count:
            max_count = v
    
    if max_count >= 10:
        return False, 0, None
    
    score = sum(1 for v in line_counts.values() if v == 6)
    return True, score, line_counts

def _affine_seed():
    """Generate an affinely transformed 4x4 grid."""
    base = np.array([[x, y] for y in range(4) for x in range(4)], dtype=np.int64)
    for _ in range(300):
        A = np.array([[random.randint(-4, 4), random.randint(-4, 4)],
                       [random.randint(-4, 4), random.randint(-4, 4)]], dtype=np.int64)
        det = int(A[0,0]*A[1,1] - A[0,1]*A[1,0])
        if det == 0:
            continue
        t = A @ base.T
        if np.any(np.abs(t) > 900):
            continue
        pts = t.T.tolist()
        is_valid, score, _ = _evaluate_full(pts)
        if is_valid:
            return pts, score
    return None, 0

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_points = None
    best_score = -1

    # Try diverse affine seeds
    for _ in range(50):
        if time.time() > deadline:
            break
        pts, score = _affine_seed()
        if pts is not None and score > best_score:
            best_score = score
            best_points = [p[:] for p in pts]
        if best_score >= 14:
            break

    # Fallback
    if best_points is None:
        best_points = [[x, y] for y in range(4) for x in range(4)]
        is_valid, best_score, _ = _evaluate_full(best_points)

    current_points = [p[:] for p in best_points]
    current_score = best_score

    # Get initial line counts
    _, _, current_line_counts = _evaluate_full(current_points)
    if current_line_counts is None:
        return best_points

    # Simulated annealing local search with incremental evaluation
    T = 3.0
    T_min = 0.1
    cool_rate = 0.997

    while time.time() < deadline:
        idx = random.randint(0, 15)
        old = current_points[idx]
        old_x, old_y = old[0], old[1]

        r = random.random()
        if r < 0.30:
            nx = old_x + random.randint(-3, 3)
            ny = old_y + random.randint(-3, 3)
        elif r < 0.55:
            nx = old_x + random.randint(-10, 10)
            ny = old_y + random.randint(-10, 10)
        elif r < 0.75:
            nx = old_x + random.randint(-30, 30)
            ny = old_y + random.randint(-30, 30)
        else:
            nx = random.randint(-1000, 1000)
            ny = random.randint(-1000, 1000)

        if abs(nx) > 1000 or abs(ny) > 1000:
            continue

        # Try the move using incremental evaluation
        # We need to pass the current points with idx updated
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
            # Accept with probability exp(delta/T)
            if random.random() < np.exp(delta / T):
                current_points[idx] = [nx, ny]
                current_score = new_score
                current_line_counts = new_line_counts
            # else: reject, keep old state

        T = max(T_min, T * cool_rate)

        # Occasionally restart from best with perturbation
        if random.random() < 0.005 and time.time() < deadline - 10:
            current_points = [p[:] for p in best_points]
            current_score = best_score
            # Perturb a few points
            for _ in range(random.randint(2, 5)):
                pi = random.randint(0, 15)
                px = best_points[pi][0] + random.randint(-20, 20)
                py = best_points[pi][1] + random.randint(-20, 20)
                if abs(px) > 1000 or abs(py) > 1000:
                    px = random.randint(-1000, 1000)
                    py = random.randint(-1000, 1000)
                current_points[pi] = [px, py]
            is_valid, current_score, current_line_counts = _evaluate_full(current_points)
            if not is_valid or current_line_counts is None:
                current_points = [p[:] for p in best_points]
                current_score = best_score
                _, _, current_line_counts = _evaluate_full(current_points)
                if current_line_counts is None:
                    return best_points

    return best_points
