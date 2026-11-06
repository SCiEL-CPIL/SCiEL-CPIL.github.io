"""Integral point sets in general position — fresh design.
Strategy: precomputed 5-point base + fast local search to extend.
All checks use exact integer arithmetic (no floating point).
"""
import time
import random
import math
import sys

_DEADLINE = time.time() + 110.0  # safety margin under 120s
_MAX_COORD = 1000

# ---------- exact integer helpers ----------

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _sq_dist(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy

def _sq_dist_is_square(p, q):
    return _is_square(_sq_dist(p, q))

def _collinear(p, q, r):
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]) == 0

def _concyclic(p, q, r, s):
    # Determinant of 4x4 matrix; zero iff concyclic
    # | x  y  x^2+y^2  1 |
    # | x' y' x'^2+y'^2 1 |
    # | x''y'' x''^2+y''^2 1 |
    # | x'''y''' x'''^2+y'''^2 1 |
    # Expand using the last column (all 1s) via cofactors:
    # det = sum_{i} (-1)^{i+3} * M_{i,3}
    # where M_{i,3} is the 3x3 minor removing row i and col 3.
    # M_{0,3} = |y  x^2+y^2|  (rows 1,2,3; cols 0,1,2)
    #           |y' x'^2+y'^2|
    #           |y''x''^2+y''^2|
    # etc.
    def det3(a, b, c):
        return a * (b * c[2] - c[1] * b[2]) - b * (a[2] * c[1] - c[0] * a[1]) + c * (a[1] * c[0] - b[0] * a[1])
    
    pts = [p, q, r, s]
    # Build 3x3 minors for each row removed (col 3 removed)
    signs = [1, -1, 1, -1]  # (-1)^{i+3} for i=0,1,2,3
    total = 0
    for i in range(4):
        rows = [pts[j] for j in range(4) if j != i]
        # 3x3 matrix: [y, x^2+y^2, 1] for each row
        # Actually cols are 0,1,2: x, y, x^2+y^2
        # Minor removes row i and col 3 (the "1" column)
        # So the 3x3 is: [x, y, x^2+y^2] for the 3 remaining rows
        a = (rows[0][0], rows[0][1], rows[0][0]*rows[0][0] + rows[0][1]*rows[0][1])
        b = (rows[1][0], rows[1][1], rows[1][0]*rows[1][0] + rows[1][1]*rows[1][1])
        c = (rows[2][0], rows[2][1], rows[2][0]*rows[2][0] + rows[2][1]*rows[2][1])
        # det3 for 3x3: |a0 a1 a2|
        #                |b0 b1 b2|
        #                |c0 c1 c2|
        d = (a[0]*(b[1]*c[2] - b[2]*c[1]) - a[1]*(b[0]*c[2] - b[2]*c[0]) + a[2]*(b[0]*c[1] - b[1]*c[0]))
        total += signs[i] * d
    return total == 0

def _check_pairwise_rational(pts):
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if not _sq_dist_is_square(pts[i], pts[j]):
                return False
    return True

def _check_no_three_collinear(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(pts[i], pts[j], pts[k]):
                    return False
    return True

def _check_no_four_concyclic(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _concyclic(pts[i], pts[j], pts[k], pts[l]):
                        return False
    return True

def _is_valid_set(pts):
    if len(pts) < 3:
        return False
    if len(set(tuple(p) for p in pts)) != len(pts):
        return False
    if not _check_pairwise_rational(pts):
        return False
    if not _check_no_three_collinear(pts):
        return False
    if not _check_no_four_concyclic(pts):
        return False
    return True

# ---------- fast incremental checks ----------

def _can_add_point(pts, cand):
    """Check if cand can be added to pts maintaining all constraints."""
    if any(cand == p for p in pts):
        return False
    # rational distances to all existing
    for p in pts:
        if not _sq_dist_is_square(cand, p):
            return False
    # no three collinear: for each pair in pts, check with cand
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            if _collinear(pts[i], pts[j], cand):
                return False
    # no four concyclic: for each triple in pts, check with cand
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _concyclic(pts[i], pts[j], pts[k], cand):
                    return False
    return True

def _can_replace(pts, idx, new_pt):
    """Check if replacing pts[idx] with new_pt maintains all constraints."""
    if any(new_pt == p for i, p in enumerate(pts) if i != idx):
        return False
    # rational distances to all other points
    for i, p in enumerate(pts):
        if i != idx and not _sq_dist_is_square(new_pt, p):
            return False
    # no three collinear
    others = [p for i, p in enumerate(pts) if i != idx]
    # check triples involving new_pt
    for i in range(len(others)):
        for j in range(i + 1, len(others)):
            if _collinear(others[i], others[j], new_pt):
                return False
    # check triples among others (unchanged, but re-verify for safety)
    for i in range(len(others)):
        for j in range(i + 1, len(others)):
            for k in range(j + 1, len(others)):
                if _collinear(others[i], others[j], others[k]):
                    return False
    # no four concyclic
    all_pts = [new_pt] + others
    n = len(all_pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _concyclic(all_pts[i], all_pts[j], all_pts[k], all_pts[l]):
                        return False
    return True

# ---------- precomputed base set ----------

_BASE_POINTS = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]

# Verify at import time
_assert_valid = _is_valid_set(_BASE_POINTS)
# If the base set is somehow invalid (shouldn't happen), fall back to a trivial valid set
if not _assert_valid:
    _BASE_POINTS = [(0, 0), (3, 4), (0, 5)]  # right triangle, valid

# ---------- search ----------

def _generate_candidates(pts, rng, limit=5000):
    """Generate candidate lattice points near existing points."""
    candidates = []
    seen = set()
    for p in pts:
        for dx in range(-50, 51):
            for dy in range(-50, 51):
                if dx == 0 and dy == 0:
                    continue
                cand = (p[0] + dx, p[1] + dy)
                if -_MAX_COORD <= cand[0] <= _MAX_COORD and -_MAX_COORD <= cand[1] <= _MAX_COORD:
                    if cand not in seen:
                        seen.add(cand)
                        candidates.append(cand)
    # Add some random candidates
    for _ in range(200):
        cand = (rng.randint(-_MAX_COORD, _MAX_COORD), rng.randint(-_MAX_COORD, _MAX_COORD))
        if cand not in seen:
            seen.add(cand)
            candidates.append(cand)
    rng.shuffle(candidates)
    return candidates[:limit]

def _search_greedy(pts, rng, time_limit):
    """Try to add points greedily."""
    pts = list(pts)
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        candidates = _generate_candidates(pts, rng, limit=3000)
        for cand in candidates:
            if time.time() >= time_limit:
                break
            if _can_add_point(pts, cand):
                pts.append(cand)
                improved = True
                break
    return pts

def _search_replace(pts, rng, time_limit):
    """Try to improve by replacing one point with a better one (may enable future additions)."""
    pts = list(pts)
    n = len(pts)
    for _ in range(200):  # up to 200 replacement attempts
        if time.time() >= time_limit:
            break
        idx = rng.randrange(n)
        old = pts[idx]
        # Generate candidates near old point
        found = False
        for _ in range(200):
            if time.time() >= time_limit:
                break
            dx = rng.randint(-30, 30)
            dy = rng.randint(-30, 30)
            new_pt = (old[0] + dx, old[1] + dy)
            if -_MAX_COORD <= new_pt[0] <= _MAX_COORD and -_MAX_COORD <= new_pt[1] <= _MAX_COORD:
                if _can_replace(pts, idx, new_pt):
                    pts[idx] = new_pt
                    found = True
                    break
        if not found:
            # Try a random replacement
            for _ in range(50):
                if time.time() >= time_limit:
                    break
                new_pt = (rng.randint(-_MAX_COORD, _MAX_COORD), rng.randint(-_MAX_COORD, _MAX_COORD))
                if _can_replace(pts, idx, new_pt):
                    pts[idx] = new_pt
                    found = True
                    break
    return pts

def _search_combined(pts, rng, time_limit):
    """Combined search: alternate between greedy addition and replacement."""
    pts = list(pts)
    best = list(pts)
    best_len = len(pts)
    
    while time.time() < time_limit:
        # Try greedy addition
        candidates = _generate_candidates(pts, rng, limit=2000)
        added = False
        for cand in candidates:
            if time.time() >= time_limit:
                break
            if _can_add_point(pts, cand):
                pts.append(cand)
                added = True
                if len(pts) > best_len:
                    best = list(pts)
                    best_len = len(pts)
                break
        if added:
            continue
        # Try replacement to potentially enable future additions
        n = len(pts)
        if n < 3:
            break
        idx = rng.randrange(n)
        old = pts[idx]
        replaced = False
        for _ in range(100):
            if time.time() >= time_limit:
                break
            dx = rng.randint(-40, 40)
            dy = rng.randint(-40, 40)
            new_pt = (old[0] + dx, old[1] + dy)
            if -_MAX_COORD <= new_pt[0] <= _MAX_COORD and -_MAX_COORD <= new_pt[1] <= _MAX_COORD:
                if _can_replace(pts, idx, new_pt):
                    pts[idx] = new_pt
                    replaced = True
                    break
        if not replaced:
            # Give up if stuck
            break
    return best if best_len >= len(pts) else pts

# ---------- main entry point ----------

def construct():
    global _DEADLINE
    rng = random.Random(42)
    
    # Start with the verified base set
    pts = list(_BASE_POINTS)
    best = list(pts)
    best_len = len(pts)
    
    # Phase 1: Greedy addition
    t1 = time.time() + 30
    pts = _search_greedy(pts, rng, t1)
    if len(pts) > best_len:
        best = list(pts)
        best_len = len(pts)
    
    # Phase 2: Combined search (addition + replacement)
    t2 = time.time() + 80
    if time.time() < t2:
        pts2 = _search_combined(best, rng, t2)
        if len(pts2) > best_len:
            best = list(pts2)
            best_len = len(pts2)
    
    # Phase 3: Final greedy push if time remains
    t3 = _DEADLINE - 5
    if time.time() < t3 and best_len < 12:
        pts3 = _search_greedy(best, rng, t3)
        if len(pts3) > best_len:
            best = list(pts3)
            best_len = len(pts3)
    
    # Final verification
    if not _is_valid_set(best):
        # Fallback to base set
        best = list(_BASE_POINTS)
    
    # Format output
    result = {
        'D': 1,
        'points': [[[x, 1], [y, 1]] for x, y in best]
    }
    return result


# Verify base set at module load (fast)
# _is_valid_set(_BASE_POINTS) already checked above
