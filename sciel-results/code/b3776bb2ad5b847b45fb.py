"""Program using MODEL-AND-OPTIMIZE: local search + geometric verification for integral point sets.

Model:
- Parameterize as (D, list of points with rational x and y*sqrt(D)).
- Objective: maximize number of distinct points n (3..12) such that:
  * all pairwise squared distances are rational (equivalently integral after scaling for a fixed D)
  * no three points are collinear (exact determinant 0 check in affine 2-space)
  * no four points are concyclic (determinant-based cyclic condition using squared distances)
  * |numerators|, |denominators| <= 1e40, D squarefree <= 1e12
- For D=1, reduce to integer lattice with integer squared distances.

Optimization:
- Start from known integral point sets.
- Use Pythagorean triple-based enumeration to efficiently find candidate points
  at integer distance from all existing points.
- Iterate: try to extend by 1 point using efficient intersection of distance constraints.
- Always return a valid configuration of maximum found n.

Deadline:
- Use time.perf_counter(); stop loops before ~110s wall-clock, then finalize.
"""
import time
from math import isqrt
from itertools import combinations

TIME_LIMIT = 150.0
SAFETY = 30.0
DEADLINE = TIME_LIMIT - SAFETY

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _pyth_offsets(dx, max_r):
    """Yield all (dy, r) such that dx^2 + dy^2 = r^2 for |dy| <= max_r.
    Uses factorization: (r-dx)(r+dx) = dy^2.
    """
    if dx == 0:
        # dy^2 + 0 = r^2 => r = |dy|
        for dy in range(-max_r, max_r + 1):
            yield (dy, abs(dy))
        return
    
    dx2 = dx * dx
    # For each r from |dx| to max_r, check if r^2 - dx^2 is a perfect square
    # More efficient: iterate over possible r values
    r_min = abs(dx)
    if r_min == 0:
        r_min = 1
    for r in range(r_min, max_r + 1):
        rem = r * r - dx2
        if rem < 0:
            continue
        dy = isqrt(rem)
        if dy * dy == rem:
            yield (dy, r)
            if dy != 0:
                yield (-dy, r)

def _gcd(a, b):
    a, b = abs(int(a)), abs(int(b))
    while b:
        a, b = b, a % b
    return a

def _sqdist(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy

def _collinear3(A, B, C):
    # (B.x - A.x)*(C.y - A.y) - (B.y - A.y)*(C.x - A.x)
    return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])

def _conic_coeff(A, B, C, D):
    # determinant
    # | x1 y1 x1^2+y1^2 1 |
    # | x2 y2 x2^2+y2^2 1 |
    # | x3 y3 x3^2+y3^2 1 |
    # | x4 y4 x4^2+y4^2 1 |
    (x1, y1) = A
    (x2, y2) = B
    (x3, y3) = C
    (x4, y4) = D
    s1 = x1*x1 + y1*y1
    s2 = x2*x2 + y2*y2
    s3 = x3*x3 + y3*y3
    s4 = x4*x4 + y4*y4
    # Expand determinant with column 4 elimination (use row1 as pivot)
    # D = x1*y2*(s3 - s4) - x1*y3*(s2 - s4) + x1*y4*(s2 - s3)
    #   - y1*x2*(s3 - s4) + y1*x3*(s2 - s4) - y1*x4*(s2 - s3)
    #   + (s2*y3*x4 - s2*y4*x3) - (s3*y2*x4 - s3*y4*x2) + (s4*y2*x3 - s4*y3*x2)
    d = (
        x1 * y2 * (s3 - s4) - x1 * y3 * (s2 - s4) + x1 * y4 * (s2 - s3)
        - y1 * x2 * (s3 - s4) + y1 * x3 * (s2 - s4) - y1 * x4 * (s2 - s3)
        + s2 * (y3 * x4 - y4 * x3)
        - s3 * (y2 * x4 - y4 * x2)
        + s4 * (y2 * x3 - y3 * x2)
    )
    return d

def _valid_set(pts):
    # Check distinct
    seen = set(pts)
    if len(seen) != len(pts):
        return False
    # Check all squared distances are perfect squares (D=1)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if not _is_square(_sqdist(pts[i], pts[j])):
                return False
    # No 3 collinear
    for a, b, c in combinations(pts, 3):
        if _collinear3(a, b, c) == 0:
            return False
    # No 4 concyclic
    for a, b, c, d in combinations(pts, 4):
        if _conic_coeff(a, b, c, d) == 0:
            return False
    return True

def _format_points(pts):
    # D=1 so y is rational y with den=1
    return {"D": 1, "points": [[[x, 1], [y, 1]] for (x, y) in pts]}

def _try_extend(current, deadline):
    """Try to extend an integral point set by one point using efficient search.
    
    Uses Pythagorean triple enumeration from the first point to generate
    candidate points at integer distance, then filters by distance to other points.
    """
    n = len(current)
    if n < 2:
        return current
    
    P0 = current[0]
    x0, y0 = P0
    
    # Bound the search region based on existing points
    xs = [p[0] for p in current]
    ys = [p[1] for p in current]
    spread_x = max(xs) - min(xs)
    spread_y = max(ys) - min(ys)
    margin = max(spread_x, spread_y) * 2 + 200
    xmin = min(xs) - margin
    xmax = max(xs) + margin
    ymin = min(ys) - margin
    ymax = max(ys) + margin
    
    existing = set(current)
    max_r = margin + 100
    
    t0 = time.perf_counter()
    found = None
    checked = set()
    
    # Strategy: for each candidate x, generate (dy, r) pairs from P0
    # that give integer distance, then check against all other points
    for x in range(xmin, xmax + 1):
        if time.perf_counter() > deadline:
            break
        
        dx0 = x - x0
        
        for dy0, r0 in _pyth_offsets(dx0, max_r):
            y = y0 + dy0
            if y < ymin or y > ymax:
                continue
            if (x, y) in existing or (x, y) in checked:
                continue
            checked.add((x, y))
            
            # Check distances to all other existing points
            ok = True
            for i in range(1, n):
                dx = x - current[i][0]
                dy = y - current[i][1]
                if not _is_square(dx * dx + dy * dy):
                    ok = False
                    break
            if not ok:
                continue
            
            # Quick collinearity test
            bad = False
            for a in range(n):
                for b in range(a + 1, n):
                    if _collinear3(current[a], current[b], (x, y)) == 0:
                        bad = True
                        break
                if bad:
                    break
            if bad:
                continue
            
            # Quick concyclicity check with triples from existing points
            for a, b, c in combinations(range(n), 3):
                if _conic_coeff(current[a], current[b], current[c], (x, y)) == 0:
                    bad = True
                    break
            if bad:
                continue
            
            # Full validation
            new_set = current + [(x, y)]
            if _valid_set(new_set):
                return new_set
    
    return current

def construct():
    # Known 5-point integral point sets to try as seeds
    candidates = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (9, 12), (0, 15), (12, 9), (15, 0)],
        [(0, 0), (7, 24), (0, 25), (24, 7), (25, 0)],
    ]
    
    t0 = time.perf_counter()
    deadline = t0 + DEADLINE
    
    best = None
    best_n = 0
    
    for seed in candidates:
        if time.perf_counter() > deadline - 5:
            break
        if not _valid_set(seed):
            continue
        if best is None or len(seed) > best_n:
            best = seed[:]
            best_n = len(seed)
    
    if best is None:
        # Minimal valid set
        best = [(0, 0), (3, 4), (0, 5)]
        best_n = 3
    
    # Try iterative extension with time guard
    while len(best) < 12 and (time.perf_counter() < deadline - 3):
        extended = _try_extend(best, deadline)
        if len(extended) <= len(best):
            break
        best = extended
        best_n = len(best)
    
    # Final safeguard
    if not _valid_set(best):
        best = [(0, 0), (3, 4), (0, 5)]
    
    return _format_points(best)
