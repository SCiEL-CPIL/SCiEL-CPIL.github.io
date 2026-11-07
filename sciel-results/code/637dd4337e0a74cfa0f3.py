import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

_CONST = 137280.0
_BIG = 1e18

def _h_coeffs(p, q_):
    c3 = -(1.0 + 12.0*p + 112.0*q_)/_CONST
    a5 = c3*4096.0
    a4 = -46080.0*c3
    a3 = 256.0*q_ + 205824.0*c3
    a2 = -387072.0*c3
    a1 = 16.0*p + 2688.0*q_ + 322560.0*c3
    a0 = -48.0*p - 1344.0*q_ - 137280.0*c3
    return np.array([a5, a4, a3, a2, a1, a0])

def _largest_pos_sign_root(c):
    amax = np.max(np.abs(c))
    if amax < 1e-300:
        return _BIG
    nz = np.where(np.abs(c) > 1e-12*amax)[0]
    if len(nz) == 0:
        return _BIG
    c = c[nz[0]:]
    if len(c) < 2:
        return _BIG
    c = c/amax
    try:
        rts = np.roots(c)
    except Exception:
        return _BIG
    best = -1.0
    for rt in rts:
        re = rt.real
        if abs(rt.imag) > 1e-8*max(1.0, abs(re)):
            continue
        if re <= 1e-9:
            continue
        d = 1e-5*max(1.0, abs(re))
        try:
            a = np.polyval(c, re - d)
            b = np.polyval(c, re + d)
        except Exception:
            continue
        if a*b < 0.0 and re > best:
            best = re
    return best if best > 0.0 else _BIG

def _objective(x):
    p, q_ = x
    return _largest_pos_sign_root(_h_coeffs(p, q_))

def _seed():
    c0, c1, c2 = -113.0/100.0, 1.0/25.0, 1.0/3240.0
    return np.array([c1/c0, c2/c0])

def construct():
    t0 = time.time()
    deadline = 200.0
    def left():
        return deadline - (time.time() - t0)

    seed = _seed()
    best_x = seed.copy()
    best_val = _objective(best_x)

    def consider(x, v):
        nonlocal best_x, best_val
        if v < best_val:
            best_val = float(v)
            best_x = np.array(x, dtype=float)

    # Phase 1: broad coarse grid
    if left() > 60:
        for (plo, phi, qlo, qhi, n) in [
            (-20.0, 20.0, -2.0, 2.0, 100),
            (-3.0, 3.0, -0.3, 0.3, 150),
        ]:
            if left() < 15:
                break
            ps = np.linspace(plo, phi, n)
            qs = np.linspace(qlo, qhi, n)
            for i in range(n):
                p = ps[i]
                for j in range(n):
                    q = qs[j]
                    consider([p, q], _objective([p, q]))
    # Phase 2: fine grid around best
    if left() > 40:
        hp, hq = best_x
        for (dp, dq, n) in [(0.1, 0.01, 100), (0.02, 0.002, 80)]:
            if left() < 10:
                break
            ps = np.linspace(hp - dp, hp + dp, n)
            qs = np.linspace(hq - dq, hq + dq, n)
            for i in range(n):
                p = ps[i]
                for j in range(n):
                    q = qs[j]
                    consider([p, q], _objective([p, q]))
    # Phase 3: DE (local + broad)
    if left() > 25:
        hp, hq = best_x
        local_bounds = [(hp - 0.5, hp + 0.5), (hq - 0.05, hq + 0.05)]
        broad_bounds = [(-10.0, 10.0), (-1.0, 1.0)]
        for bi, b in enumerate([local_bounds]*3 + [broad_bounds]*2):
            if left() < 8:
                break
            try:
                res = differential_evolution(_objective, b, maxiter=50,
                    popsize=25, tol=1e-10, seed=100+bi, polish=False,
                    init="sobol", updating="deferred")
                consider(res.x, res.fun)
            except Exception:
                pass
    # Phase 4: Nelder-Mead polish (multiple restarts)
    if left() > 15:
        hp, hq = best_x
        starts = [best_x]
        for k in range(4):
            starts.append(best_x + np.array([1e-4*(k+1)*((k%2)*2-1), 1e-5*(k+1)]))
        for si, s in enumerate(starts):
            if left() < 5:
                break
            try:
                res = minimize(_objective, s, method="Nelder-Mead",
                    options={"maxiter": 20000, "xatol": 1e-13, "fatol": 1e-13})
                consider(res.x, res.fun)
            except Exception:
                pass
    # Final safety: ensure we don't return worse than seed
    if _objective(seed) < best_val:
        best_x = seed
    p, q_ = best_x
    return [1.0, float(p), float(q_)]
