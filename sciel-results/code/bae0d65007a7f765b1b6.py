"""Optimize the uncertainty-inequality constant via Gram-matrix (SOS) + bisection."""
import time
import numpy as np
from fractions import Fraction

# ---------- exact Hermite coefficients (physicists' H_n) ----------
def _hermite_coeffs(n):
    if n == 0:
        return [Fraction(1)]
    if n == 1:
        return [Fraction(0), Fraction(2)]
    c1 = _hermite_coeffs(n - 1)
    c2 = _hermite_coeffs(n - 2)
    a = [Fraction(0)] + [Fraction(2) * v for v in c1]
    b = [Fraction(-2 * (n - 1)) * v for v in c2]
    m = max(len(a), len(b))
    res = [Fraction(0)] * m
    for i, v in enumerate(a):
        res[i] += v
    for i, v in enumerate(b):
        res[i] += v
    while len(res) > 1 and res[-1] == 0:
        res.pop()
    return res

_HC = {n: _hermite_coeffs(n) for n in (0, 4, 8, 12)}
_HF = {n: np.array([float(v) for v in c], dtype=np.float64) for n, c in _HC.items()}
_H0_0 = float(_HC[0][0])
_H4_0 = float(_HC[4][0])
_H8_0 = float(_HC[8][0])
_H12_0 = float(_HC[12][0])

# ---------- coefficient extraction ----------
def _build_q_coeffs(c1, c2):
    """q(x)=g(x)/x^2 with c0=1. Returns ascending coeffs of q in powers x^{2k}."""
    c3 = -(_H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    g = 1.0 * _HF[0] + c1 * _HF[4] + c2 * _HF[8] + c3 * _HF[12]
    q = g[2:13:2]
    while len(q) > 1 and q[-1] == 0.0:
        q = q[:-1]
    return np.asarray(q, dtype=np.float64)


def _build_Q_coeffs(c1, c2, t):
    """Q(y)=q(sqrt(y))-t, ascending coeffs in y (degree 5)."""
    q = _build_q_coeffs(c1, c2)
    Q = np.zeros(6, dtype=np.float64)
    m = min(len(q), 6)
    Q[:m] = q[:m]
    Q[0] -= t
    return Q


# ---------- true objective (root based) ----------
def _value_from(c1, c2):
    try:
        q = _build_q_coeffs(c1, c2)
        if len(q) < 2:
            return np.inf
        try:
            roots_y = np.roots(q[::-1])
        except Exception:
            return np.inf
        real_pos = []
        for r in roots_y:
            if abs(r.imag) <= 1e-10 * max(1.0, abs(r.real)) and r.real > 1e-12:
                real_pos.append(float(r.real))
        if not real_pos:
            return 1e10
        def Qval(y):
            return np.polyval(q[::-1], y)
        best_y = None
        for y in sorted(real_pos, reverse=True):
            h = 1e-7 * max(1.0, abs(y))
            if Qval(y - h) * Qval(y + h) < 0:
                best_y = y
                break
        if best_y is None:
            for y in sorted(real_pos, reverse=True):
                for h in [1e-6, 1e-5, 1e-4, 1e-3]:
                    h = h * max(1.0, abs(y))
                    if Qval(y - h) * Qval(y + h) < 0:
                        best_y = y
                        break
                if best_y is not None:
                    break
        if best_y is None:
            return 1e10
        return best_y / (2.0 * np.pi)
    except Exception:
        return np.inf


# ---------- SOS / Gram-matrix feasibility ----------
def _sdp_feasible(Q, t):
    """True iff Q(y)>=0 for all y>=0 (certified by SOS Gram matrix)."""
    Q = np.asarray(Q, dtype=np.float64)
    if len(Q) < 6:
        Q = np.pad(Q, (0, 6 - len(Q)))
    # S (3x3) for s(y)^2 ; P (2x2) for y*p(y)^2
    S00 = Q[0]
    S01 = 0.5 * Q[1]
    S02 = 0.5 * Q[2]
    S11 = Q[2]
    S12 = 0.5 * Q[3]
    S22 = Q[4]
    P00 = Q[3]
    P01 = 0.5 * Q[4]
    P11 = Q[5]
    S = np.array([[S00, S01, S02], [S01, S11, S12], [S02, S12, S22]], dtype=np.float64)
    P = np.array([[P00, P01], [P01, P11]], dtype=np.float64)
    for M in (S, P):
        ev = np.linalg.eigvalsh(M)
        if ev[0] < -1e-6:
            return False
    return True


def _find_best_Q(c1, c2):
    """Minimise t such that Q_t(y)>=0 for all y>=0. Returns (t, Q)."""
    Q0 = _build_Q_coeffs(c1, c2, 0.0)
    if _sdp_feasible(Q0, 0.0):
        return 0.0, Q0
    # find upper bound where Q_t >= 0 (large t shifts Q up)
    hi = 1.0
    while hi < 1e6 and not _sdp_feasible(_build_Q_coeffs(c1, c2, hi), hi):
        hi *= 2.0
    if hi >= 1e6:
        return np.inf, None
    lo = 0.0
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if _sdp_feasible(_build_Q_coeffs(c1, c2, mid), mid):
            hi = mid
        else:
            lo = mid
    return hi, _build_Q_coeffs(c1, c2, hi)


def _sdp_value(c1, c2):
    t, _ = _find_best_Q(c1, c2)
    return t / (2.0 * np.pi)


def construct():
    start = time.time()
    DEADLINE = 200.0

    seed_c0 = -113.0 / 100.0
    seed_c1 = 1.0 / 25.0
    seed_c2 = 1.0 / 3240.0
    seed_c1r = seed_c1 / seed_c0
    seed_c2r = seed_c2 / seed_c0

    best_c1 = seed_c1r
    best_c2 = seed_c2r
    best_val = _value_from(best_c1, best_c2)

    def time_left():
        return DEADLINE - (time.time() - start)

    # ---- Phase 1: grid search (exact objective) ----
    for span, n in [(2.0, 25), (0.5, 25), (0.12, 25)]:
        c1_range = np.linspace(seed_c1r - span, seed_c1r + span, n)
        c2_range = np.linspace(seed_c2r - span * 0.3, seed_c2r + span * 0.3, n)
        for c1 in c1_range:
            if time_left() < 20:
                break
            for c2 in c2_range:
                if time_left() < 8:
                    break
                v = _value_from(c1, c2)
                if np.isfinite(v) and v < best_val:
                    best_val = v
                    best_c1, best_c2 = c1, c2

    # ---- Phase 2: local optimisation on smooth SOS objective ----
    def obj_sdp(params):
        if time_left() < 6:
            return best_val
        return _sdp_value(params[0], params[1])

    from scipy.optimize import minimize

    if time_left() > 30:
        try:
            res = minimize(obj_sdp, [best_c1, best_c2], method='Nelder-Mead',
                           options={'maxiter': 4000, 'xatol': 1e-9, 'fatol': 1e-10})
            if np.isfinite(res.fun):
                v = _value_from(res.x[0], res.x[1])
                if np.isfinite(v) and v < best_val:
                    best_val = v
                    best_c1, best_c2 = float(res.x[0]), float(res.x[1])
        except Exception:
            pass

    # ---- Phase 3: direct Nelder-Mead on exact objective (polish) ----
    def obj_exact(params):
        if time_left() < 6:
            return best_val
        return _value_from(params[0], params[1])

    rng = np.random.RandomState(7)
    starts = [[best_c1, best_c2], [seed_c1r, seed_c2r]]
    for scale in [0.3, 0.1, 0.03, 0.01]:
        for _ in range(3):
            starts.append([best_c1 + rng.randn() * scale,
                           best_c2 + rng.randn() * scale * 0.3])

    for s in starts:
        if time_left() < 12:
            break
        try:
            res = minimize(obj_exact, s, method='Nelder-Mead',
                           options={'maxiter': 12000, 'xatol': 1e-13, 'fatol': 1e-15})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = float(res.fun)
                best_c1, best_c2 = float(res.x[0]), float(res.x[1])
        except Exception:
            pass

    # ---- Phase 4: final ultra-fine polish ----
    if time_left() > 8:
        try:
            res = minimize(obj_exact, [best_c1, best_c2], method='Nelder-Mead',
                           options={'maxiter': 30000, 'xatol': 1e-15, 'fatol': 1e-17,
                                    'initial_simplex': [[best_c1, best_c2],
                                                         [best_c1 + 1e-7, best_c2],
                                                         [best_c1, best_c2 + 1e-7]]})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = float(res.fun)
                best_c1, best_c2 = float(res.x[0]), float(res.x[1])
        except Exception:
            pass

    if not np.isfinite(best_val):
        return np.array([seed_c0, seed_c1, seed_c2])

    return np.array([1.0, float(best_c1), float(best_c2)])
