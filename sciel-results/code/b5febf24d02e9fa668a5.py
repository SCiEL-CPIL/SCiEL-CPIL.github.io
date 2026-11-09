"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time as _time
import numpy as _np
# EVOLVE-BLOCK-START
import numpy as _np
import time as _time

_DEADLINE = _time.time() + 82.0   # 8 s safety margin under the 90 s kill
_RNG = _np.random.default_rng(20240611)


def _time_left():
    return _DEADLINE - _time.time()


def _Om(a, om0, w0, wa):
    # Omega_m(a) = Om0 * a^-3 / (Om0*a^-3 + (1-Om0)*a^(-3(1+w0+wa/2)))
    om0 = _np.asarray(om0)
    a = _np.asarray(a)
    de = _np.exp(_np.log(a) * (-3.0 * (1.0 + w0 + 0.5 * wa)))
    m = _np.exp(_np.log(a) * (-3.0))
    return om0 * m / (om0 * m + (1.0 - om0) * de)


def _gamma(c, om0, w0, wa, a):
    c = _np.asarray(c)
    a = _np.asarray(a)
    om = _Om(a, om0, w0, wa)
    q = 1.0 - om
    u = 1.0 + w0 + wa * (1.0 - a)
    g = c[0] + c[1] * q + c[2] * q * q + c[3] * u + c[4] * wa * a + c[5] * u * q
    return _np.clip(g, 0.1, 1.2)


def _f(c, om0, w0, wa, a):
    return _Om(a, om0, w0, wa) ** _gamma(c, om0, w0, wa, a)


def _fref(om0, w0, wa, a):
    # Reference growth index: 5/7 * (Om / (1+w0+wa(1-a)))^0.38
    om = _Om(a, om0, w0, wa)
    u = 1.0 + w0 + wa * (1.0 - a)
    g = _np.clip((5.0 / 7.0) * (om / u) ** 0.38, 0.1, 1.2)
    return om ** g


def _D(c, om0, w0, wa, a):
    # D(a)/D(1) = exp(-int_a^1 f(e^x) dx), evaluated at x=ln(a)
    xs = _np.log(a)
    x1 = 0.0
    f1 = _f(c, om0, w0, wa, _np.exp(xs))
    # Simpson on [x, 0] (positive orientation)
    n = 48
    if n % 2:
        n += 1
    xs_mid = 0.5 * (xs + x1)
    h = (x1 - xs) / n
    h2 = h / 2.0
    t = _np.arange(n + 1)
    x = xs + t * h
    f = _f(c, om0, w0, wa, _np.exp(x))
    S = h / 3.0 * (_np.sum(f[0::2]) + 4.0 * _np.sum(f[1::2]) + _np.sum(f[2::2]))
    # midpoint correction
    fm = _f(c, om0, w0, wa, _np.exp(xs_mid))
    S = S + 2.0 * h2 * fm
    return _np.exp(-S)


def _Dref(om0, w0, wa, a):
    xs = _np.log(a)
    x1 = 0.0
    f1 = _fref(om0, w0, wa, _np.exp(xs))
    n = 48
    if n % 2:
        n += 1
    xs_mid = 0.5 * (xs + x1)
    h = (x1 - xs) / n
    h2 = h / 2.0
    t = _np.arange(n + 1)
    x = xs + t * h
    f = _fref(om0, w0, wa, _np.exp(x))
    S = h / 3.0 * (_np.sum(f[0::2]) + 4.0 * _np.sum(f[1::2]) + _np.sum(f[2::2]))
    fm = _fref(om0, w0, wa, _np.exp(xs_mid))
    S = S + 2.0 * h2 * fm
    return _np.exp(-S)


def _relerr(c, om0, w0, wa, a):
    f = _f(c, om0, w0, wa, a)
    fr = _fref(om0, w0, wa, a)
    D = _D(c, om0, w0, wa, a)
    Dr = _Dref(om0, w0, wa, a)
    e1 = _np.abs(f - fr) / _np.maximum(_np.abs(fr), 1e-12)
    e2 = _np.abs(D - Dr) / _np.maximum(_np.abs(Dr), 1e-12)
    return _np.concatenate([e1, e2])


def _grid():
    om0 = _np.array([0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
    w0 = _np.array([-1.5, -1.2, -1.0, -0.8, -0.5, 0.0, 0.3, 0.5])
    wa = _np.array([-1.0, -0.7, -0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0])
    a = _np.logspace(-3, 0, 60)
    return om0, w0, wa, a


def _objective(c, om0, w0, wa, a, beta=500.0):
    # smooth surrogate for 100 * max relative error (log-sum-exp)
    e = _relerr(c, om0, w0, wa, a)
    e = _np.clip(e, 1e-6, None)
    m = _np.max(e)
    s = _np.log(_np.sum(_np.exp(beta * (e - m)))) / beta + m
    return 100.0 * s


def _construct():
    om0, w0, wa, a = _grid()

    # ---- Stage 1: least-squares fit of gamma to the exact index ----
    # Precompute design matrix across the full parameter grid.
    A = []
    y = []
    for o in om0:
        for w in w0:
            for wa_ in wa:
                om = _Om(a, o, w, wa_)
                q = 1.0 - om
                u = 1.0 + w + wa_ * (1.0 - a)
                g = _np.clip((5.0 / 7.0) * (om / u) ** 0.38, 0.1, 1.2)
                A.append(_np.column_stack([
                    _np.ones_like(a), q, q * q, u, wa_ * a, u * q
                ]))
                y.append(g)
    A = _np.concatenate(A, axis=0)
    y = _np.concatenate(y, axis=0)
    coef, *_ = _np.linalg.lstsq(A, y, rcond=None)
    # Fix the dominant high-z term to the exact 5/7 limit.
    coef[1] = 5.0 / 7.0

    def _score(c):
        return _np.max(_relerr(c, om0, w0, wa, a)) * 100.0

    def _surrogate(c):
        return _objective(c, om0, w0, wa, a)

    # ---- Stage 2: Nelder-Mead on the smooth surrogate ----
    from scipy.optimize import minimize
    c0 = _np.clip(coef, -10.0, 10.0).astype(float)
    c0[1] = 5.0 / 7.0
    best = c0.copy()
    best_s = _score(best)
    try:
        res = minimize(_surrogate, best, method='Nelder-Mead',
                       options={'xatol': 1e-6, 'fatol': 1e-9, 'maxiter': 600,
                                'adaptive': True})
        cand = _np.clip(res.x, -10.0, 10.0)
        cand[1] = 5.0 / 7.0
        s = _score(cand)
        if s < best_s:
            best, best_s = cand, s
    except Exception:
        pass

    # ---- Stage 3: high-precision polish (tighter tolerance) ----
    if _time_left() > 20.0:
        try:
            res2 = minimize(_surrogate, best, method='Nelder-Mead',
                            options={'xatol': 1e-9, 'fatol': 1e-12, 'maxiter': 800,
                                     'adaptive': True})
            cand = _np.clip(res2.x, -10.0, 10.0)
            cand[1] = 5.0 / 7.0
            s = _score(cand)
            if s < best_s:
                best, best_s = cand, s
        except Exception:
            pass

    # ---- Stage 4: light randomized restarts if time remains ----
    if _time_left() > 25.0:
        for _ in range(12):
            if _time_left() < 18.0:
                break
            pert = best + _RNG.normal(0.0, 0.02, size=6)
            pert[1] = 5.0 / 7.0
            pert = _np.clip(pert, -10.0, 10.0)
            try:
                r = minimize(_surrogate, pert, method='Nelder-Mead',
                             options={'xatol': 1e-8, 'fatol': 1e-11, 'maxiter': 250,
                                      'adaptive': True})
                cand = _np.clip(r.x, -10.0, 10.0)
                cand[1] = 5.0 / 7.0
                s = _score(cand)
                if s < best_s:
                    best, best_s = cand, s
            except Exception:
                pass

    # Final sanity: keep gamma in range on the grid.
    gmin = _np.min(_gamma(best, om0, w0, wa, a))
    gmax = _np.max(_gamma(best, om0, w0, wa, a))
    if not (0.1 <= gmin and gmax <= 1.2):
        # fall back to the pure least-squares fit with 5/7
        best = _np.clip(coef, -10.0, 10.0)
        best[1] = 5.0 / 7.0

    return [float(x) for x in best]


_COEF = _construct()


def construct():
    return list(_COEF)
# EVOLVE-BLOCK-END
