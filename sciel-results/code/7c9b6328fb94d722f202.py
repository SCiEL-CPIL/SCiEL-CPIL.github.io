import time
import random
import numpy as np

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _evaluate_fast(points):
    """Evaluate a list of 16 [x,y] points.
    Returns (is_valid, score).
    is_valid: 16 distinct integer points, |x|,|y|<=1000, no 5 collinear.
    score: number of lines containing exactly 4 points.
    """
    n = len(points)
    if n != 16:
        return False, 0
    xs = np.array([p[0] for p in points], dtype=np.int64)
    ys = np.array([p[1] for p in points], dtype=np.int64)
    if np.any(np.abs(xs) > 1000) or np.any(np.abs(ys) > 1000):
        return False, 0
    # distinctness
    pts = np.column_stack((xs, ys))
    if len(set(map(tuple, pts.tolist()))) != 16:
        return False, 0
    # count pairs per line
    # use vectorized line keys
    iu, ju = np.triu_indices(16, k=1)
    dx = xs[ju] - xs[iu]
    dy = ys[ju] - ys[iu]
    # gcd
    ax = np.abs(dx)
    ay = np.abs(dy)
    # compute gcd elementwise
    g = np.gcd(ax, ay)
    # avoid division by zero: where g==0 (both 0) can't happen since distinct points
    g_safe = np.where(g == 0, 1, g)
    dxn = dx // g_safe
    dyn = dy // g_safe
    # canonicalize sign: make first nonzero positive
    neg = (dxn < 0) | ((dxn == 0) & (dyn < 0))
    dxn = np.where(neg, -dxn, dxn)
    dyn = np.where(neg, -dyn, dyn)
    # c = dyn*x1 - dxn*y1
    c = dyn * xs[iu] - dxn * ys[iu]
    keys = np.column_stack((dxn, dyn, c))
    # count occurrences
    keys_t = list(map(tuple, keys.tolist()))
    from collections import Counter
    cnt = Counter(keys_t)
    max_count = 0
    score = 0
    for k, v in cnt.items():
        if v > max_count:
            max_count = v
        if v == 6:  # exactly 4 points -> 6 pairs
            score += 1
    if max_count >= 10:  # 5 or more points on a line
        return False, 0
    return True, score

def _affine_seed():
    """Generate a 4x4 grid affinely transformed."""
    base = np.array([[x, y] for y in range(4) for x in range(4)], dtype=np.int64)
    # random integer affine matrix with det != 0, keep within bounds
    for _ in range(200):
        A = np.array([[random.randint(-3, 3), random.randint(-3, 3)],
                       [random.randint(-3, 3), random.randint(-3, 3)]], dtype=np.int64)
        det = int(A[0,0]*A[1,1] - A[0,1]*A[1,0])
        if det == 0:
            continue
        t = A @ base.T
        if np.any(np.abs(t) > 900):
            continue
        pts = t.T.tolist()
        is_valid, score = _evaluate_fast(pts)
        if is_valid:
            return pts, score
    return None, 0

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin under 150s

    best_points = None
    best_score = -1

    # Try several affine seeds to find a good starting point
    for _ in range(30):
        if time.time() > deadline:
            break
        pts, score = _affine_seed()
        if pts is not None and score > best_score:
            best_score = score
            best_points = pts
        if best_score >= 12:
            break

    # Fallback default
    if best_points is None:
        best_points = [[x, y] for y in range(4) for x in range(4)]
        is_valid, best_score = _evaluate_fast(best_points)

    current_points = [p[:] for p in best_points]
    current_score = best_score

    # Local search
    while time.time() < deadline:
        idx = random.randint(0, 15)
        old = current_points[idx]
        r = random.random()
        if r < 0.35:
            nx = old[0] + random.randint(-4, 4)
            ny = old[1] + random.randint(-4, 4)
        elif r < 0.6:
            nx = old[0] + random.randint(-15, 15)
            ny = old[1] + random.randint(-15, 15)
        else:
            nx = random.randint(-1000, 1000)
            ny = random.randint(-1000, 1000)
        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
        current_points[idx] = [nx, ny]
        is_valid, new_score = _evaluate_fast(current_points)
        if is_valid:
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
            # accept if not worse, else revert (with occasional random accept)
            if new_score >= current_score or random.random() < 0.05:
                current_score = new_score
            else:
                current_points[idx] = old
        else:
            current_points[idx] = old

    return best_points
