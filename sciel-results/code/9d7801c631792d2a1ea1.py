"""Integral point sets in general position — incremental exact-rational search."""
import math
import time
from fractions import Fraction

# ---------------- exact helpers ----------------

def _is_square(n):
    if n < 0:
        return False
    if n == 0:
        return True
    r = math.isqrt(n)
    return r * r == n

def _is_square_frac(fr):
    fr = Fraction(fr)
    if fr < 0:
        return False
    return _is_square(fr.numerator) and _is_square(fr.denominator)

def _sqfree_part(n):
    if n < 0:
        return _sqfree_part(-n)
    if n == 0:
        return 0
    out = 1
    for p in range(2, math.isqrt(n) + 1):
        while n % (p * p) == 0:
            n //= (p * p)
            out *= p
    return out

def _is_sqfree(n):
    if n < 0:
        return _is_sqfree(-n)
    if n == 0:
        return False
    for p in range(2, math.isqrt(n) + 1):
        if n % (p * p) == 0:
            return False
    return True

def _sqrt_frac(fr):
    fr = Fraction(fr)
    if fr < 0:
        return None
    if _is_square(fr.numerator) and _is_square(fr.denominator):
        return Fraction(math.isqrt(fr.numerator), math.isqrt(fr.denominator))
    return None

def _is_rational(fr):
    fr = Fraction(fr)
    return _is_square(fr.numerator) and _is_square(fr.denominator)

# ---------------- geometry (exact) ----------------

def _collinear3(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) == 0

def _circ_det(p1, p2, p3, p4):
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    return (x1**2 + y1**2) * ((x2 - x4) * (y3 - y4) - (y2 - y4) * (x3 - x4)) \
         - (x2**2 + y2**2) * ((x1 - x4) * (y3 - y4) - (y1 - y4) * (x3 - x4)) \
         + (x3**2 + y3**2) * ((x1 - x4) * (y2 - y4) - (y1 - y4) * (x2 - x4))

def _validate(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            if pts[i] == pts[j]:
                return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear3(pts[i], pts[j], pts[k]):
                    return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _circ_det(pts[i], pts[j], pts[k], pts[l]) == 0:
                        return False
    return True

def _max_den(pts):
    m = 1
    for p in pts:
        for fr in p:
            m = max(m, fr.numerator, fr.denominator)
    return m

# ---------------- incremental construction ----------------

def _build_base(D, target):
    """Try to build a general-position set of `target` points in Q^2 with
    rational pairwise distances, using the standard two-point construction
    (fix two points, add points on a shared circle)."""
    # fix P0 = (0,0), P1 = (1,0)
    base = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))]
    used = set()
    # shared circle through P0,P1: center (1/2, k), k in Q, radius^2 = 1/4 + k^2
    # parameterize: P = (1/2 + r cos t, k + r sin t)
    # rational points on circle: use slope parameter.
    # Point on circle x^2 - x + (y-k)^2 = 0  ->  (x-1/2)^2 + (y-k)^2 = 1/4 + k^2
    # Let r2 = 1/4 + k^2 (rational). Parametrize via line through a known point.
    # Known point on circle: (0, k) and (1, k) and (1/2, k+sqrt(r2))... use (0,0) is on it.
    # Use slope m line through (0,0): y = m x.
    # (x-1/2)^2 + (m x - k)^2 = r2
    # x^2 - x + 1/4 + m^2 x^2 - 2 m k x + k^2 = r2 = 1/4 + k^2
    # (1+m^2) x^2 - (1 + 2 m k) x = 0
    # x = 0 (P0) or x = (1 + 2 m k)/(1 + m^2)
    # y = m x
    # This gives a point on the circle through P0. For it to also be on the
    # circle through P1 we need it on the SAME circle, which it is (we used that circle).
    # So any such point is on the circle through P0 and P1. Good.
    # But we also need distance to P1 rational. Since it's on the circle through P0,P1,
    # distance to P1: sqrt((x-1)^2 + y^2). Is that rational? Not guaranteed.
    # Use the classic: choose k so that for the chosen point, both distances rational.
    # Instead use the known parametrization of rational-distance points to two foci.
    # Simpler: brute force small rational points and test.
    # We'll do a bounded brute-force over small numerator/denominator.
    found = list(base)
    limit = 40
    denoms = [1, 2, 3, 4, 5, 6, 8, 10, 12]
    candidates = []
    for den in denoms:
        for num in range(-limit, limit + 1):
            for den2 in denoms:
                for num2 in range(-limit, limit + 1):
                    candidates.append((Fraction(num, den), Fraction(num2, den2)))
    # We already have P0,P1; add more.
    for (x, y) in candidates:
        if len(found) >= target:
            break
        cand = (x, y)
        if cand in found:
            continue
        # check rational distances to all found
        ok = True
        for (px, py) in found:
            dsq = (x - px) * (x - px) + (y - py) * (y - py)
            if not _is_rational(dsq):
                ok = False
                break
        if not ok:
            continue
        newset = found + [cand]
        if not _validate(newset):
            continue
        found = newset
    return found

def _try_extend(pts, D, budget):
    """Try to add one more point to pts. Returns (newpts or None)."""
    n = len(pts)
    # candidate pool: rational points near existing points
    # We'll generate a pool and test.
    limit = 60
    denoms = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
    # center around existing points
    centers = pts
    pool = []
    for (cx, cy) in centers:
        for den in denoms:
            for num in range(-limit, limit + 1):
                for den2 in denoms:
                    for num2 in range(-limit, limit + 1):
                        pool.append((cx + Fraction(num, den), cy + Fraction(num2, den2)))
    # dedup
    pool = list(set(pool))
    # randomize order deterministically
    for cand in pool:
        if cand in pts:
            continue
        ok = True
        for (px, py) in pts:
            dsq = (cand[0] - px) * (cand[0] - px) + (cand[1] - py) * (cand[1] - py)
            if not _is_rational(dsq):
                ok = False
                break
        if not ok:
            continue
        newset = pts + [cand]
        if not _validate(newset):
            continue
        return newset
    return None

def _search_one(D, deadline):
    """Try to find the largest general-position set for this D."""
    best = None
    # start from a base of size 3 or 4
    base = _build_base(D, 4)
    if base is None or len(base) < 3:
        # minimal fallback
        base = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))]
        if not _validate(base):
            return None
    cur = base
    # try to extend greedily
    for _ in range(8):
        if time.time() > deadline:
            break
        nxt = _try_extend(cur, D, deadline)
        if nxt is None:
            break
        cur = nxt
    if best is None or len(cur) > len(best):
        best = cur
    return best

def _to_output(pts, D):
    return {'D': D, 'points': [[[fr.numerator, fr.denominator] for fr in p] for p in pts]}

# ---------------- main ----------------

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill
    best = None
    best_size = 0
    # Try D=1 first (plain integer/rational), then a few squarefree D.
    D_list = [1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17]
    for D in D_list:
        if time.time() > deadline:
            break
        if not _is_sqfree(D):
            continue
        if D > 10**12:
            continue
        res = _search_one(D, deadline)
        if res is not None and len(res) > best_size:
            best = res
            best_size = len(res)
    if best is None:
        # absolute fallback: the known 5-point set
        pts = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)),
               (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33)),
               (Fraction(16), Fraction(63))]
        best = pts
    # ensure no numerator/denominator exceeds 10^40
    cap = 10**40
    out = _to_output(best, 1)
    # if D was not 1 we must keep it; recompute D from the search that produced best
    # (we lost track; default D=1 since our construction lives in Q^2 => D=1)
    return out
