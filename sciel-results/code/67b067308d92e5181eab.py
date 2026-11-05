import time
import random
import numpy as np
from collections import Counter

def _evaluate_fast(points):
    """Evaluate 16 [x,y] points. Returns (is_valid, score)."""
    n = len(points)
    if n != 16:
        return False, 0
    xs = np.array([p[0] for p in points], dtype=np.int64)
    ys = np.array([p[1] for p in points], dtype=np.int64)
    if np.any(np.abs(xs) > 1000) or np.any(np.abs(ys) > 1000):
        return False, 0
    pts = np.column_stack((xs, ys))
    if len(set(map(tuple, pts.tolist()))) != 16:
        return False, 0
    iu, ju = np.triu_indices(16, k=1)
    dx = xs[ju] - xs[iu]
    dy = ys[ju] - ys[iu]
    ax = np.abs(dx)
    ay = np.abs(dy)
    g = np.gcd(ax, ay)
    g_safe = np.where(g == 0, 1, g)
    dxn = dx // g_safe
    dyn = dy // g_safe
    neg = (dxn < 0) | ((dxn == 0) & (dyn < 0))
    dxn = np.where(neg, -dxn, dxn)
    dyn = np.where(neg, -dyn, dyn)
    c = dyn * xs[iu] - dxn * ys[iu]
    keys = np.column_stack((dxn, dyn, c))
    keys_t = list(map(tuple, keys.tolist()))
    cnt = Counter(keys_t)
    max_count = 0
    score = 0
    for k, v in cnt.items():
        if v > max_count:
            max_count = v
        if v == 6:
            score += 1
    if max_count >= 10:
        return False, 0
    return True, score

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
        is_valid, score = _evaluate_fast(pts)
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
        is_valid, best_score = _evaluate_fast(best_points)

    current_points = [p[:] for p in best_points]
    current_score = best_score

    # Simulated annealing local search
    T = 3.0
    T_min = 0.1
    cool_rate = 0.997

    while time.time() < deadline:
        idx = random.randint(0, 15)
        old = current_points[idx]

        r = random.random()
        if r < 0.30:
            nx = old[0] + random.randint(-3, 3)
            ny = old[1] + random.randint(-3, 3)
        elif r < 0.55:
            nx = old[0] + random.randint(-10, 10)
            ny = old[1] + random.randint(-10, 10)
        elif r < 0.75:
            nx = old[0] + random.randint(-30, 30)
            ny = old[1] + random.randint(-30, 30)
        else:
            nx = random.randint(-1000, 1000)
            ny = random.randint(-1000, 1000)

        if abs(nx) > 1000 or abs(ny) > 1000:
            continue

        current_points[idx] = [nx, ny]
        is_valid, new_score = _evaluate_fast(current_points)

        if not is_valid:
            current_points[idx] = old
            continue

        delta = new_score - current_score
        if delta >= 0:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            # Accept with probability exp(delta/T)
            if random.random() < np.exp(delta / T):
                current_score = new_score
            else:
                current_points[idx] = old

        T = max(T_min, T * cool_rate)

        # Occasionally restart from best with perturbation
        if random.random() < 0.005 and time.time() < deadline - 10:
            current_points = [p[:] for p in best_points]
            current_score = best_score
            # Perturb a few points
            for _ in range(random.randint(2, 5)):
                pi = random.randint(0, 15)
                current_points[pi] = [
                    best_points[pi][0] + random.randint(-20, 20),
                    best_points[pi][1] + random.randint(-20, 20)
                ]
                if abs(current_points[pi][0]) > 1000 or abs(current_points[pi][1]) > 1000:
                    current_points[pi] = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
            is_valid, current_score = _evaluate_fast(current_points)
            if not is_valid:
                current_points = [p[:] for p in best_points]
                current_score = best_score

    return best_points
