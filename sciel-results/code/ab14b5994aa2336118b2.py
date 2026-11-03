"""Uncertainty inequality: tune Hermite-Gaussian test function to reduce the
upper bound C4 on A(f)*A(Fourier(f)).

g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12)(x) * exp(-x^2/2),  a12 fixed by g(0)=0.
Score = upper bound on C4 (lower is better). We minimize that bound.
"""
import time
import warnings
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ---- deadline guard (evaluator kills at 210s wall-clock; keep safety margin) ----
_T0 = time.time()
_DEADLINE = _T0 + 175.0


def _time_left():
    return _DEADLINE - time.time()


# ---------------------------------------------------------------------------
# Hermite moments (physicists' convention H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2})
# I_{n,m} = integral_{-inf}^{inf} H_n(x) H_m(x) e^{-x^2} dx
# g(x) = sum a_i H_i(x) e^{-x^2/2};  g(0)=0 fixes a12.
# ---------------------------------------------------------------------------
def _build_moments():
    import sympy as sp
    x = sp.symbols('x', real=True)
    M = sp.Matrix([[sp.integrate(sp.hermite(i, x) * sp.hermite(j, x) * sp.exp(-x**2), x, -sp.oo, sp.oo)
                    for j in range(13)] for i in range(13)])
    vals = [[sp.nsimplify(sp.simplify(M[i, j]), [sp.pi])] for i in range(13) for j in range(13)]
    flat = list(vals)
    # rationalize everything (pi factors should be rational multiples)
    rat = []
    for v in flat:
        if v.has(sp.pi):
            # v = c * pi ; c rational
            c = sp.simplify(v / sp.pi)
            rat.append(Fraction(sp.nsimplify(c, rational=True)))
        else:
            rat.append(Fraction(sp.nsimplify(v, rational=True)))
    # g(0) = a0 + a4*24 + a8*1344 + a12*46080 = 0  ->  a12 = -(a0+24 a4+1344 a8)/46080
    return rat


try:
    _MOM = _build_moments()
    _HAS_EXACT = True
except Exception:
    _HAS_EXACT = False
    _MOM = None


def _mom(i, j):
    if _HAS_EXACT:
        return _MOM[i * 13 + j]
    # numeric fallback
    from scipy.special import eval_hermite
    from scipy.integrate import quad
    f = lambda t: eval_hermite(i, t) * eval_hermite(j, t) * np.exp(-t * t)
    v, _ = quad(f, -np.inf, np.inf, limit=200)
    return Fraction(v).limit_denominator(10**12)


def _coeffs(a0, a4, a8):
    a12 = -(a0 + 24 * a4 + 1344 * a8) / 46080
    return [a0, a4, a8, a12]


def _bound(a0, a4, a8):
    a = _coeffs(a0, a4, a8)
    # S = sum_{n,m} a_n a_m I_{n,m}  ( = ||g||_2^2 )
    S = Fraction(0)
    # Q = sum_{n,m} a_n a_m I_{n+2,m+2}  ( = ||x g||_2^2 )
    Q = Fraction(0)
    for n in range(4):
        for m in range(4):
            S += a[n] * a[m] * _mom(n, m)
            Q += a[n] * a[m] * _mom(n + 2, m + 2)
    if S <= 0:
        return 1e300
    # upper bound C4 = (2*Q/S) ** 0.5
    return float((2.0 * float(Q) / float(S)) ** 0.5)


def _f(x):
    try:
        return _bound(x[0], x[1], x[2])
    except Exception:
        return 1e300


def construct():
    global _T0, _DEADLINE
    best = (-113 / 100, 1 / 25, 1 / 3240)
    best_val = _bound(*best)

    # multi-start SLSQP with deadline guard
    rng = np.random.default_rng(20240501)
    starts = [np.array(best, dtype=float)]
    for _ in range(40):
        starts.append(rng.normal(0.0, 1.0, size=3))

    for s in starts:
        if _time_left() < 2.0:
            break
        try:
            res = minimize(_f, s, method="SLSQP",
                           options=dict(maxiter=200, ftol=1e-12, disp=False))
            if res.success and res.fun < best_val:
                best = (float(res.x[0]), float(res.x[1]), float(res.x[2]))
                best_val = float(res.fun)
        except Exception:
            pass

    # high-precision polish pass (tighter tolerance, more iterations, exact local solve)
    if _time_left() > 3.0:
        try:
            res = minimize(_f, np.array(best, dtype=float), method="SLSQP",
                           options=dict(maxiter=600, ftol=1e-16, disp=False))
            if res.success and res.fun < best_val:
                best = (float(res.x[0]), float(res.x[1]), float(res.x[2]))
                best_val = float(res.fun)
        except Exception:
            pass

    return np.array(best)
