"""Fresh design: fit a universal 8-node positive quadrature for the
dimensionless luminosity-distance integrand over a bank of physically
motivated dark-energy models, minimizing worst-case relative error."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, least_squares, differential_evolution

_T0 = time.time()
_DEADLINE = _T0 + 75.0  # safety margin inside the 90 s wall-clock kill

_EPS = 1e-12


def _E_of_a(a, w0, wa):
    a = np.asarray(a, dtype=float)
    return np.sqrt(np.power(a, -3.0) + 1e-12 + 0.5 * np.power(a, -3.0 - wa) *
                   (1.0 - np.power(a, -1.0 - wa)))


def _exact_integral(w0, wa):
    from scipy.integrate import quad
    f = lambda a: 1.0 / _E_of_a(a, w0, wa)
    val, _ = quad(f, 0.0, 1.0, limit=200, epsabs=1e-12, epsrel=1e-12)
    return val


def _build_bank():
    bank = []
    w0s = [-1.5, -1.25, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5,
           -0.3, -0.1, 0.0, 0.1, 0.3]
    was = [-2.0, -1.5, -1.0, -0.7, -0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    for w0 in w0s:
        for wa in was:
            bank.append((w0, wa))
    # a few extra "stiff" extremes
    for w0, wa in [(-1.0, -3.0), (-1.0, 1.0), (-1.2, -2.5), (-0.5, 1.0),
                   (-0.9, -1.5), (-0.9, 1.5), (-1.1, 0.5)]:
        bank.append((w0, wa))
    # de-duplicate
    seen = set()
    uniq = []
    for w0, wa in bank:
        key = (round(w0, 6), round(wa, 6))
        if key not in seen:
            seen.add(key)
            uniq.append((w0, wa))
    return uniq


def _exact_table(bank):
    I = np.empty(len(bank))
    for i, (w0, wa) in enumerate(bank):
        I[i] = _exact_integral(w0, wa)
    return I


def _rel_err(nodes, weights, I):
    a = 1.0 / (1.0 + nodes)
    E = np.array([_E_of_a(a, w0, wa) for (w0, wa) in _BANK])
    num = np.sum(weights[None, :] / E, axis=1)
    return np.abs(num - I) / np.maximum(I, 1e-300)


def _smooth_loss(x):
    nodes, weights = _decode(x)
    r = _rel_err(nodes, weights, _I_TABLE)
    return np.sum(np.maximum(r, _EPS) ** 2)


def _worst(x):
    nodes, weights = _decode(x)
    return float(np.max(_rel_err(nodes, weights, _I_TABLE)))


def _decode(x):
    s = x[:8]
    nodes = 1.0 / (1.0 + np.exp(-s))
    nodes = nodes * (1.0 - 2.0 * _EPS) + _EPS
    w = np.exp(x[8:16])
    w = w / np.sum(w)
    return nodes, w


def _encode(nodes, weights):
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    s = np.log(nodes / (1.0 - nodes))
    w = np.clip(weights, 1e-12, None)
    w = w / np.sum(w)
    return np.concatenate([s, np.log(w)])


def _proj(x):
    nodes, weights = _decode(x)
    return _encode(nodes, weights)


def _lsq(x):
    nodes, weights = _decode(x)
    r = _rel_err(nodes, weights, _I_TABLE)
    return np.maximum(r, 1e-6)


def _refine(x):
    # smooth least-squares polish
    try:
        res = least_squares(_lsq, x, method='trf', max_nfev=400)
        x = _proj(res.x)
    except Exception:
        pass
    # coordinate-descent min-max refiner (handles the non-smooth max)
    best = x
    bestw = _worst(x)
    scale = 0.05
    for _ in range(60):
        if time.time() > _DEADLINE:
            break
        improved = False
        for k in range(16):
            for d in (1.0, -1.0):
                trial = best.copy()
                trial[k] += d * scale
                trial = _proj(trial)
                wv = _worst(trial)
                if wv < bestw - 1e-14:
                    best = trial
                    bestw = wv
                    improved = True
        if not improved:
            scale *= 0.5
            if scale < 1e-4:
                break
    return best


def _seed_grid():
    n = 8
    base = []
    base.append((np.arange(n) + 0.5) / n)                       # midpoint
    base.append(np.linspace(0.0, 1.0, n + 2)[1:-1])              # interior Gauss-ish
    base.append(np.sin((np.arange(n) + 0.5) / n * np.pi / 2))    # cluster near 1
    base.append(1.0 - np.sin((np.arange(n) + 0.5) / n * np.pi / 2))  # cluster near 0
    base.append(np.sqrt((np.arange(n) + 0.5) / n))               # sqrt spacing
    base.append(np.linspace(0.05, 0.95, n))
    base.append(np.linspace(0.0, 1.0, n + 1)[1:-1] + 0.01)
    return base


def construct():
    global _BANK, _I_TABLE
    bank = _build_bank()
    _BANK = bank
    _I_TABLE = _exact_table(bank)

    best = None
    bestw = np.inf
    for g in _seed_grid():
        g = np.clip(g, 1e-6, 1.0 - 1e-6)
        x = _encode(g, np.full(8, 1.0 / 8.0))
        x = _refine(x)
        wv = _worst(x)
        if wv < bestw:
            bestw = wv
            best = x
        if time.time() > _DEADLINE:
            break

    # local polish from the best seed
    try:
        res = minimize(_smooth_loss, best, method='L-BFGS-B',
                       options={'maxiter': 300, 'maxls': 50})
        best = _proj(res.x)
        best = _refine(best)
        bestw = _worst(best)
    except Exception:
        pass

    # global multi-start with a time budget
    nstarts = 0
    while time.time() < _DEADLINE - 3.0:
        x0 = _proj(np.concatenate([
            np.random.uniform(-2.0, 2.0, 8),
            np.random.uniform(-1.0, 1.0, 8)]))
        try:
            res = minimize(_smooth_loss, x0, method='L-BFGS-B',
                           options={'maxiter': 120, 'maxls': 30})
            cand = _proj(res.x)
            cand = _refine(cand)
            wv = _worst(cand)
            if wv < bestw:
                bestw = wv
                best = cand
        except Exception:
            pass
        nstarts += 1
        if nstarts > 60:
            break

    # final polish
    try:
        res = minimize(_smooth_loss, best, method='L-BFGS-B',
                       options={'maxiter': 200, 'maxls': 40})
        best = _proj(res.x)
        best = _refine(best)
    except Exception:
        pass

    nodes, weights = _decode(best)
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    weights = np.clip(weights, 1e-12, None)
    weights = weights / np.sum(weights)
    order = np.argsort(nodes)
    nodes = nodes[order]
    weights = weights[order]
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
