"""Construct 12 double-root locations to minimize B/(2*pi)."""
import time
import math
import random
import sympy
from sympy import Rational, Symbol, Poly, laguerre, real_roots, Matrix, nsimplify

# ---- deadline guard ----
_START = time.time()
_DEADLINE = 600.0
_MARGIN = 30.0

def _remaining():
    return _DEADLINE - _MARGIN - (time.time() - _START)

def _timeup():
    return (time.time() - _START) > (_DEADLINE - _MARGIN)

def _to_rational(x):
    return Rational(nsimplify(float(x), rational=True))

# Pre-compute Laguerre polynomials L_{2k}^{(-1/2)}(x) for k=0..25
_x = Symbol('x')
_alpha = Rational(-1, 2)
_laguerres = [laguerre(2 * k, _x, _alpha) for k in range(26)]
_laguerre_diffs = [l.diff(_x) for l in _laguerres]

def _score(zs):
    """Exact score evaluation for a list of z values. Returns B/(2*pi) or None if invalid."""
    zs_r = sorted([_to_rational(z) for z in zs])
    
    # Check distinctness and positivity
    if len(set(zs_r)) != 12:
        return None
    if any(zi <= 0 for zi in zs_r):
        return None
    
    # Build 26x26 system
    M = []
    b = []
    
    # g(0) = 0
    M.append([l.subs(_x, 0) for l in _laguerres])
    b.append(Rational(0))
    
    # g'(0) = 1
    M.append([l.subs(_x, 0) for l in _laguerre_diffs])
    b.append(Rational(1))
    
    # g(z_i) = 0 and g'(z_i) = 0 for each z_i
    for zi in zs_r:
        M.append([l.subs(_x, zi) for l in _laguerres])
        b.append(Rational(0))
        M.append([l.subs(_x, zi) for l in _laguerre_diffs])
        b.append(Rational(0))
    
    A = Matrix(M)
    bv = Matrix(b)
    
    try:
        if A.det() == 0:
            return None
        c = A.LUsolve(bv)
    except Exception:
        return None
    
    # Build g(x)
    g = sum(c[k] * _laguerres[k] for k in range(26))
    gP = Poly(g, _x)
    
    # Build denominator D = x * prod (x - zi)^2
    D = Poly(_x, _x)
    for zi in zs_r:
        D = D * Poly((_x - zi) ** 2, _x)
    
    try:
        q, rem = gP.div(D)
    except Exception:
        return None
    
    if q.is_zero:
        return 0.0
    
    try:
        roots = real_roots(q)
    except Exception:
        return None
    
    B = 0.0
    for r in roots:
        rv = float(r)
        if rv <= 0:
            continue
        try:
            lo = float(q.evalf(rv - 1e-198))
            hi = float(q.evalf(rv + 1e-198))
        except Exception:
            continue
        if lo == 0.0 or hi == 0.0:
            continue
        if (lo < 0 and hi > 0) or (lo > 0 and hi < 0):
            if rv > B:
                B = rv
    
    return B / (2.0 * math.pi)

# ---- candidate pool ----
def _build_pool():
    lo = 0.1
    hi = 15.0
    n = 150
    step = (hi - lo) / (n - 1)
    pool = []
    for i in range(n):
        v = lo + i * step
        pool.append(round(v, 6))
    # Add some sparse larger values
    for v in [20.0, 30.0, 40.0, 50.0]:
        pool.append(v)
    return pool

_POOL = _build_pool()
_POOL_SET = set(_POOL)

def _valid(zs):
    if len(zs) != 12:
        return False
    seen = set()
    for z in zs:
        if not math.isfinite(z) or z <= 0:
            return False
        r = _to_rational(z)
        if r in seen:
            return False
        seen.add(r)
    return True

def _local_search(zs):
    """Coordinate descent over pool values."""
    zs = list(zs)
    try:
        cur = _score(zs)
    except Exception:
        return None, None
    if cur is None:
        return None, None
    best_zs = list(zs)
    best = cur
    while _remaining() > 60.0:
        improved = False
        for i in range(12):
            old = zs[i]
            for cand in _POOL:
                if cand == old:
                    continue
                trial = list(zs)
                trial[i] = cand
                if not _valid(trial):
                    continue
                try:
                    s = _score(trial)
                except Exception:
                    continue
                if s is not None and s < best - 1e-12:
                    zs = trial
                    best = s
                    best_zs = list(zs)
                    improved = True
                    break
            if _timeup():
                break
        if not improved:
            break
    return best_zs, best

def _initial_candidates():
    cands = []
    # Evenly spaced, various ranges
    cands.append([0.5 + 1.0 * i for i in range(12)])
    cands.append([0.3 + 0.8 * i for i in range(12)])
    cands.append([0.2 + 0.5 * i for i in range(12)])
    cands.append([1.0 + 1.5 * i for i in range(12)])
    # Geometric
    cands.append([0.5 * (1.15 ** i) for i in range(12)])
    cands.append([0.3 * (1.2 ** i) for i in range(12)])
    cands.append([0.8 * (1.1 ** i) for i in range(12)])
    # Clustered small + tail
    cands.append([0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.5, 4.0, 6.0, 9.0])
    cands.append([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0])
    # Original seed
    cands.append([4.0 * (i + 1) for i in range(12)])
    return cands

def construct():
    """Return 12 double-root locations as a list of floats."""
    best_zs = None
    best = None
    
    # Evaluate initial candidates
    for cand in _initial_candidates():
        if _timeup():
            break
        if not _valid(cand):
            continue
        try:
            s = _score(cand)
        except Exception:
            continue
        if s is not None and (best is None or s < best):
            best = s
            best_zs = list(cand)
    
    # Local search from best
    if best_zs is not None and _remaining() > 60.0:
        lz, ls = _local_search(best_zs)
        if lz is not None and (best is None or ls < best):
            best = ls
            best_zs = lz
    
    # Random restarts while time remains
    rng = random.Random(42)
    while _remaining() > 90.0:
        zs = [rng.choice(_POOL) for _ in range(12)]
        if not _valid(zs):
            continue
        try:
            s = _score(zs)
        except Exception:
            continue
        if s is not None and (best is None or s < best):
            best = s
            best_zs = list(zs)
            if _remaining() > 120.0:
                lz, ls = _local_search(best_zs)
                if lz is not None and (best is None or ls < best):
                    best = ls
                    best_zs = lz
    
    # Fallback
    if best_zs is None:
        best_zs = [4.0 * (i + 1) for i in range(12)]
    
    # Final validation
    if not _valid(best_zs):
        best_zs = [4.0 * (i + 1) for i in range(12)]
    
    return [float(z) for z in best_zs]

if __name__ == "__main__":
    z = construct()
    print(f"Constructed z: {z}")
