import numpy as np
import time
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = 190.0

_NQ = 80
_xq, _wq = np.polynomial.hermite.hermgauss(_NQ)

_HK = {k: np.polynomial.hermite.hermk(k) for k in (0, 4, 8, 12)}
_HMONO = {k: np.polynomial.hermite.herm2poly(_HK[k]) for k in _HK}
_H4_0 = float(np.polynomial.polynomial.polyval(0.0, _HMONO[4]))
_H8_0 = float(np.polynomial.polynomial.polyval(0.0, _HMONO[8]))
_H12_0 = float(np.polynomial.polynomial.polyval(0.0, _HMONO[12]))

_s4 = 1.0 / np.sqrt(4.0 * np.pi)
_s2 = 1.0 / np.sqrt(2.0 * np.pi)
_P4 = _xq * _s4
_P2 = _xq * _s2
_W4 = _wq * _s4
_W2 = _wq * _s2

_qn = [np.array([1.0 + 0j])]
for _n in range(12):
    _q = _qn[-1]
    if len(_q) > 1:
        _d = np.arange(1, len(_q)) * _q[1:]
        _dq = np.concatenate(([0.0 + 0j], _d))
    else:
        _dq = np.array([0.0 + 0j])
    _xq_ = np.concatenate((np.zeros(1, complex), _q))
    _qn.append((1j / (2 * np.pi)) * _dq - 1j * _xq_)

def _bound_uv(uv):
    u = float(uv[0]); v = float(uv[1])
    a0, a4, a8 = 1.0, u, v
    a12 = -(a0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    b = np.zeros(13)
    for a, k in ((a0, 0), (a4, 4), (a8, 8), (a12, 12)):
        b += a * _HMONO[k]
    g4 = np.polynomial.polynomial.polyval(_P4, b)
    g2 = np.polynomial.polynomial.polyval(_P2, b)
    n4_4 = float(np.sum(_W4 * np.abs(g4) ** 4))
    n2_2 = float(np.sum(_W2 * np.abs(g2) ** 2))
    qhat = np.zeros_like(_P4, dtype=complex)
    for n in range(13):
        if b[n] != 0:
            qhat += b[n] * np.polynomial.polynomial.polyval(_P4, _qn[n])
    nf4_4 = float(np.sum(_W4 * np.abs(qhat) ** 4))
    return np.sqrt(n4_4 * nf4_4) / n2_2

def _time_left():
    return _DEADLINE - (time.time() - _T0)

def construct():
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_uv = np.array([seed[1] / seed[0], seed[2] / seed[0]])
    best_val = _bound_uv(best_uv)

    starts = [best_uv.copy()]
    us = np.linspace(-0.6, 0.6, 49)
    vs = np.linspace(-0.03, 0.03, 49)
    for u in us:
        for v in vs:
            starts.append(np.array([float(u), float(v)]))
    rng = np.random.default_rng(12345)
    for _ in range(400):
        starts.append(np.array([float(rng.uniform(-1.5, 1.5)), float(rng.uniform(-0.1, 0.1))]))

    for s in starts:
        if _time_left() < 4.0:
            break
        try:
            res = minimize(_bound_uv, s, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'xtol': 1e-15})
            val = float(res.fun)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_uv = np.array([float(res.x[0]), float(res.x[1])])
        except Exception:
            pass

    if _time_left() > 8.0:
        try:
            res = minimize(_bound_uv, best_uv, method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15, 'xtol': 1e-16})
            val = float(res.fun)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_uv = np.array([float(res.x[0]), float(res.x[1])])
        except Exception:
            pass

    out = np.array([1.0, best_uv[0], best_uv[1]])
    if not np.all(np.isfinite(out)):
        return seed
    return out
