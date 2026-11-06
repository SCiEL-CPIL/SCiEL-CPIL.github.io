"""Broadband antireflection coating design via characteristic-matrix + multi-start local optimization."""
import time
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = _T0 + 130.0  # safety margin before the 150 s wall-clock kill
_WLS = None
_AMP = None
_NW = None


def _wavelengths():
    global _WLS, _AMP, _NW
    if _WLS is None:
        _WLS = np.linspace(400.0, 800.0, 401)
        _NW = len(_WLS)
        _AMP = 2.0 * np.pi / _WLS  # 1/nm
    return _WLS, _AMP, _NW


def _reflectance(n, d, n0, ns, amp):
    """Vectorized characteristic-matrix reflectance (normal incidence).
    n, d: (n_layers, NW) arrays; amp: (NW,) = 2*pi/lambda. Returns R (NW,) in [0,1]."""
    M = np.empty((n.shape[0], 2, 2, n.shape[1]), dtype=np.complex128)
    c = np.cos(d * amp)
    s = np.sin(d * amp)
    M[:, 0, 0, :] = c
    M[:, 1, 1, :] = c
    M[:, 0, 1, :] = (1.0j / n) * s
    M[:, 1, 0, :] = 1.0j * n * s
    m00 = np.full(n.shape[1], ns, dtype=np.complex128)
    m01 = np.ones(n.shape[1], dtype=np.complex128)
    m10 = np.full(n.shape[1], 1.0 / ns, dtype=np.complex128)
    m11 = np.full(n.shape[1], ns, dtype=np.complex128)
    for k in range(n.shape[0] - 1, -1, -1):
        a = M[k, 0, 0, :]
        b = M[k, 0, 1, :]
        c2 = M[k, 1, 0, :]
        dd = M[k, 1, 1, :]
        n00 = a * m00 + b * m10
        n01 = a * m01 + b * m11
        n10 = c2 * m00 + dd * m10
        n11 = c2 * m01 + dd * m11
        m00, m01, m10, m11 = n00, n01, n10, n11
    Y = m10 / m00
    r = (n0 - Y) / (n0 + Y)
    R = (r * np.conj(r)).real
    np.clip(R, 0.0, 1.0, out=R)
    return R


def _refine_generic(x, obj, max_layers, max_layer):
    m = int(round(x[-1]))
    m = int(np.clip(m, 1, max_layers))
    d = np.clip(x[:m], 0.0, max_layer)
    cur = list(d) + [float(m)]
    bestv = obj(np.array(cur))
    best = cur
    try:
        res = minimize(
            lambda dd: obj(np.concatenate([np.clip(dd, 0, max_layer), [m]])),
            d.copy(),
            method="Nelder-Mead",
            options={"maxiter": 400, "xatol": 0.5, "fatol": 1e-7, "adaptive": True},
        )
        dd = np.clip(res.x, 0.0, max_layer)
        v = obj(np.concatenate([dd, [m]]))
        if v < bestv:
            bestv = v
            best = list(dd) + [float(m)]
    except Exception:
        pass
    for _ in range(40):
        improved = False
        for i in range(m):
            base = best[i]
            for step in (15.0, 5.0, 1.5, 0.4, 0.1):
                for sign in (1.0, -1.0):
                    cand = best[:]
                    cand[i] = float(np.clip(base + sign * step, 0.0, max_layer))
                    v = obj(np.array(cand))
                    if v < bestv - 1e-10:
                        best = cand
                        bestv = v
                        improved = True
                        break
                if improved:
                    break
    for nm in (m - 1, m + 1):
        if nm < 1 or nm > max_layers:
            continue
        nd = best[:nm] if nm <= m else best[:m] + [100.0]
        try:
            res = minimize(
                lambda dd: obj(np.concatenate([np.clip(dd, 0, max_layer), [nm]])),
                np.clip(nd, 0.0, max_layer).copy(),
                method="Nelder-Mead",
                options={"maxiter": 300, "xatol": 0.5, "fatol": 1e-7, "adaptive": True},
            )
            dd = np.clip(res.x, 0.0, max_layer)
            v = obj(np.concatenate([dd, [nm]]))
            if v < bestv:
                bestv = v
                best = list(dd) + [float(nm)]
        except Exception:
            pass
    return np.array(best)


def _start_families(n0, ns, nL, nH, lam0, max_layers, max_layer, max_total):
    starts = []
    starts.append([lam0 / (4.0 * nL), 1.0])
    starts.append([lam0 / (4.0 * nH), 1.0])
    starts.append([lam0 / (4.0 * nL), lam0 / (4.0 * nH), 2.0])
    starts.append([lam0 / (4.0 * nL), lam0 / (4.0 * nH), lam0 / (4.0 * nL), 3.0])
    starts.append([lam0 / (2.0 * nL), 1.0])
    starts.append([lam0 / (2.0 * nL), lam0 / (2.0 * nH), 2.0])
    for m in (4, 6, 8):
        t = np.linspace(0, 1, m)
        d = np.array([40.0 + 160.0 * t[i] for i in range(m)])
        starts.append(list(d) + [float(m)])
    rng = np.random.default_rng(12345)
    for _ in range(10):
        m = int(rng.integers(2, min(9, max_layers + 1)))
        d = rng.uniform(30.0, 200.0, size=m)
        starts.append(list(d) + [float(m)])
    feasible = []
    for s in starts:
        m = int(round(s[-1]))
        d = np.clip(s[:m], 0.0, max_layer)
        if d.sum() <= max_total:
            feasible.append(s)
    return feasible


def design(problem):
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    mat = problem["materials"]
    nL = mat["L"]
    nH = mat["H"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    lam0 = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])

    def fallback():
        t = min(lam0 / (4.0 * nL), max_layer)
        return ["L"], [t]

    wls, amp, nw = _wavelengths()

    def obj(x):
        m = int(round(x[-1]))
        m = int(np.clip(m, 1, max_layers))
        d = np.clip(x[:m], 0.0, max_layer)
        n = np.empty(m)
        for i in range(m):
            n[i] = nH if i % 2 == 1 else nL
        R = _reflectance(np.tile(n[:, None], (1, nw)), np.tile(d[:, None], (1, nw)), n0, ns, amp)
        return R.mean()

    starts = _start_families(n0, ns, nL, nH, lam0, max_layers, max_layer, max_total)
    if not starts:
        return fallback()

    best = None
    bestv = float("inf")
    for s in starts:
        if time.time() > _DEADLINE:
            break
        x = np.array(s, dtype=float)
        try:
            xr = _refine_generic(x, obj, max_layers, max_layer)
            v = obj(xr)
            if v < bestv:
                bestv = v
                best = xr
        except Exception:
            continue
    if best is None:
        return fallback()

    m = int(round(best[-1]))
    m = int(np.clip(m, 1, max_layers))
    d = np.clip(best[:m], 0.0, max_layer)
    if d.sum() > max_total:
        scale = max_total / d.sum()
        d = d * scale
    labels = []
    for i in range(m):
        labels.append("H" if i % 2 == 1 else "L")
    return labels, [float(t) for t in d]
