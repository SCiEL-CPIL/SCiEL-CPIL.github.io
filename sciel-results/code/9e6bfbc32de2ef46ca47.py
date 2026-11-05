"""Growth-index approximation for CPL dark-energy cosmologies.

Performs a deadline-guarded local search at import time to optimize six
coefficients for the compact growth-index ansatz.  construct() returns the
pre-computed coefficient list.
"""

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

# ── Deadline ────────────────────────────────────────────────────────────────
_T0 = time.time()
_DEADLINE = _T0 + 82.0

def _ok():
    return time.time() < _DEADLINE

# ── Physics ─────────────────────────────────────────────────────────────────

def _Om_m(a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    a = np.clip(a, 1e-15, 1e6)
    a3 = a ** 3
    de = (1.0 - Om0) * a ** (-3.0 * (1.0 + w0 + wa)) * np.exp(3.0 * wa * (a - 1.0))
    m = Om0 / a3
    return m / (m + de)

def _gamma(c, a, Om0, w0, wa):
    c0, c1, c2, c3, c4, c5 = c
    Om = _Om_m(a, Om0, w0, wa)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c0 + c1*q + c2*q*q + c3*u + c4*wa*a + c5*u*q

# ── Reference ODE ───────────────────────────────────────────────────────────

def _ref(Om0, w0, wa, n=800):
    a_min, a_max = 1e-4, 1.0
    x_min, x_max = np.log(a_min), 0.0

    def ode(x, y):
        a = np.exp(x)
        Om = _Om_m(a, Om0, w0, wa)
        w = w0 + wa * (1.0 - a)
        dOm_dx = -3.0 * Om * (Om - 1.0)
        D, F = y
        return [F, -(1.5 + 1.0 + w - 1.5 * dOm_dx) * F + 1.5 * Om * D]

    D0 = a_min ** (5.0/6.0)
    F0 = (5.0/6.0) * a_min ** (-1.0/6.0)
    sol = solve_ivp(ode, [x_min, x_max], [D0, F0],
                    max_step=0.005, rtol=1e-11, atol=1e-13, dense_output=True)
    if not sol.success:
        return None
    D = sol.y[0] / sol.y[0][-1]
    return sol.t, sol.y[1], D

# ── Validation set ──────────────────────────────────────────────────────────

_COSMO = [
    (0.30, -0.5,  0.0),
    (0.30, -1.0,  0.0),
    (0.30, -1.0,  0.5),
    (0.30, -1.0, -0.5),
    (0.30, -0.8,  0.3),
    (0.25, -1.0,  0.5),
    (0.35, -1.0, -0.3),
    (0.30, -1.2,  0.4),
    (0.30, -0.9, -0.4),
]

_REFS = []
for _c in _COSMO:
    try:
        r = _ref(*_c)
        if r is not None:
            _REFS.append(r + (_c,))
    except Exception:
        pass

_EVAL_A = np.logspace(np.log10(1e-3), 0.0, 200)
_EVAL_X = np.log(_EVAL_A)

# ── Objective ───────────────────────────────────────────────────────────────

def _obj(c):
    c = np.asarray(c, dtype=float)
    if np.any(~np.isfinite(c)):
        return 1e6
    if np.any(c < -10) or np.any(c > 10):
        return 1e6

    mx = 0.0
    for (x_ref, f_ref, D_ref, Om0, w0, wa) in _REFS:
        a_ref = np.exp(x_ref)
        f_i = np.interp(_EVAL_X, x_ref, f_ref)
        D_i = np.interp(_EVAL_X, x_ref, D_ref)

        Om = _Om_m(_EVAL_A, Om0, w0, wa)
        g = _gamma(c, _EVAL_A, Om0, w0, wa)
        if np.any(g < 0.1) or np.any(g > 1.2):
            pen = np.maximum(0, 0.1 - g).max() + np.maximum(0, g - 1.2).max()
            mx = max(mx, 10.0 * pen)
            continue

        f_m = np.clip(Om, 1e-15, 1.0) ** g

        # Model D via cumulative integration
        dx = np.diff(_EVAL_X)
        f_avg = 0.5 * (f_m[1:] + f_m[:-1])
        cs = np.cumsum(f_avg * dx)[::-1]
        D_m = np.exp(-cs)
        D_m /= D_m[-1]

        df = np.abs(f_m - f_i) / np.maximum(np.abs(f_i), 1e-10)
        dD = np.abs(D_m - D_i) / np.maximum(np.abs(D_i), 1e-10)
        mx = max(mx, df.max(), dD.max())

    return 100.0 * mx

# ── Optimize ────────────────────────────────────────────────────────────────

_x0 = np.array([0.545, 0.25, 0.10, -0.03, 0.02, 0.01])
_bc = _x0.copy()
_be = _obj(_bc)
print(f"init {be:.6f}", flush=True)

if _ok():
    try:
        r = minimize(_obj, _bc, method='Nelder-Mead',
                     options={'maxiter':400,'xatol':1e-5,'fatol':1e-7,'adaptive':True})
        if r.fun < _be: _be, _bc = r.fun, r.x.copy()
        print(f"NM {_be:.6f}", flush=True)
    except Exception as e:
        print(f"NM err {e}", flush=True)

if _ok():
    try:
        r = minimize(_obj, _bc, method='L-BFGS-B', bounds=[(-3,3)]*6,
                     options={'maxiter':200,'ftol':1e-12})
        if r.fun < _be: _be, _bc = r.fun, r.x.copy()
        print(f"LB {_be:.6f}", flush=True)
    except Exception as e:
        print(f"LB err {e}", flush=True)

if _ok():
    rng = np.random.default_rng(7)
    for _ in range(4):
        if not _ok(): break
        xt = np.clip(_bc + rng.normal(0, 0.04, 6), -5, 5)
        try:
            r = minimize(_obj, xt, method='Nelder-Mead',
                         options={'maxiter':150,'xatol':1e-6,'fatol':1e-8,'adaptive':True})
            if r.fun < _be: _be, _bc = r.fun, r.x.copy()
            print(f"pert {_be:.6f}", flush=True)
        except Exception:
            pass

_bc = np.clip(_bc, -10, 10)
print(f"final {_be:.6f} {_bc}", flush=True)

def construct():
    return list(_bc)
# EVOLVE-BLOCK-END
