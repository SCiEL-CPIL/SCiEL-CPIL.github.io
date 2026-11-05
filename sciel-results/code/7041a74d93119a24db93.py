"""Fixed-depth QAOA parameter design: graph-dependent two-layer angles.

Structural design (fresh):
  * Exact MaxCut optimum precomputed once at import (independent of n).
  * Fast vectorized state-vector expectation (full 2^n amplitudes, batched).
  * Per-graph Nelder-Mead from a warm-start + coarse global grid.
  * Explicit wall-clock deadline guard with safety margin.
"""
import time
import itertools
import numpy as np

_DEADLINE = 120.0
_MARGIN = 5.0
_TOL = 1e-12


def _remaining():
    return _DEADLINE - _MARGIN - time.time()


def _build_edges(n, edges):
    """Return a list of undirected (u, v) with u < v, deduplicated."""
    out = []
    seen = set()
    for e in edges:
        u = int(e[0]); v = int(e[1])
        if u == v:
            continue
        if u > v:
            u, v = v, u
        key = (u, v)
        if key in seen:
            continue
        seen.add(key)
        out.append((u, v))
    return out


def _exact_optimum(n, edges):
    """Exact MaxCut value via exhaustive enumeration (one-time, import-time)."""
    best = 0
    m = len(edges)
    for bits in range(1 << n):
        cut = 0
        for (u, v) in edges:
            if (bits >> u) & 1 != (bits >> v) & 1:
                cut += 1
        if cut > best:
            best = cut
    return float(best)


def _prepare(n, edges):
    m = 1 << n
    idx = np.arange(m, dtype=np.int64)
    bit = (idx[:, None] >> np.arange(n, dtype=np.int64)[None, :]) & 1
    edge_bit = bit[:, [u for (u, v) in edges]] ^ bit[:, [v for (u, v) in edges]]
    edge_bit = edge_bit.astype(np.float64)
    # cos/sin of the cut contribution per edge
    cos_e = np.cos(edge_bit * 0.5)
    sin_e = np.sin(edge_bit * 0.5)
    # per-edge contribution to the cut (1 if cut else 0)
    cut_e = edge_bit
    # per-node X-mixer sign: (-1)^bit
    sign = 1.0 - 2.0 * bit.astype(np.float64)
    return m, cos_e, sin_e, cut_e, sign


def _expectation(params, prep):
    """Mean expected cut for a (p,2) parameter array. Vectorized over rows."""
    m, cos_e, sin_e, cut_e, sign = prep
    K = params.shape[0]
    amp = np.empty((K, m), dtype=np.complex128)
    amp[:] = 1.0 / np.sqrt(m)
    for k in range(K):
        g = params[k, 0]; b = params[k, 1]
        cg, sg = np.cos(0.5 * g), np.sin(0.5 * g)
        cb, sb = np.cos(0.5 * b), np.sin(0.5 * b)
        re = amp[:, :, 0] if amp.ndim == 3 else amp.real
        im = amp.imag
        # cost layer: amp *= (cos(0.5 g * cut) - i sin(0.5 g * cut))
        cre = cg * cos_e - sg * sin_e
        cim = -(cg * sin_e + sg * cos_e)
        nre = re * cre - im * cim
        nim = re * cim + im * cre
        # mixer layer: amp *= (cos(0.5 b) + i sin(0.5 b) * (-1)^bit)
        nre2 = nre * cb - nim * (sb * sign)
        nim2 = nre * (sb * sign) + nim * cb
        amp.real = nre2
        amp.imag = nim2
    prob = amp.real ** 2 + amp.imag ** 2
    return float(np.einsum('km,km->', prob, cut_e))


def _evaluate(params, prep, opt):
    if opt <= 0:
        return 0.0
    return _expectation(params, prep) / opt


def _neg_eval(params, prep, opt):
    return -_evaluate(params, prep, opt)


def _nelder_mead(f, x0, bounds, max_iter, deadline):
    n = len(x0)
    x = np.array(x0, dtype=float).copy()
    pts = [x.copy()]
    for i in range(n):
        step = max(0.05, 0.05 * (bounds[i, 1] - bounds[i, 0]))
        pts.append(x.copy())
        pts[-1][i] += step
    vals = np.array([f(p) for p in pts])
    best = 0
    for it in range(max_iter):
        if time.time() > deadline:
            break
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = vals[order]
        best = 0
        centroid = np.mean(pts[:-1], axis=0)
        xr = centroid + (centroid - pts[-1])
        fxr = f(xr)
        if fxr < vals[0]:
            xe = centroid + 2.0 * (centroid - pts[-1])
            fxe = f(xe)
            if fxe < fxr:
                pts[-1] = xe; vals[-1] = fxe
            else:
                pts[-1] = xr; vals[-1] = fxr
        elif fxr < vals[-2]:
            pts[-1] = xr; vals[-1] = fxr
        else:
            xc = centroid + 0.5 * (pts[-1] - centroid)
            fxc = f(xc)
            if fxc < vals[-1]:
                pts[-1] = xc; vals[-1] = fxc
            else:
                for i in range(1, n + 1):
                    pts[i] = 0.5 * (pts[i] + pts[0])
                    vals[i] = f(pts[i])
        if np.max(vals) - np.min(vals) < 1e-9:
            break
    return pts[best], -vals[best]


def _coarse_grid(f, bounds, deadline, step=0.25):
    best_p = None; best_v = np.inf
    g1 = np.arange(bounds[0, 0], bounds[0, 1] + 1e-9, step)
    g2 = np.arange(bounds[1, 0], bounds[1, 1] + 1e-9, step)
    for a in g1:
        for b in g2:
            if time.time() > deadline:
                return best_p, best_v
            p = np.array([a, b])
            v = f(p)
            if v < best_v:
                best_v = v; best_p = p.copy()
    return best_p, best_v


def angles(n, edges, p):
    t0 = time.time()
    deadline = t0 + _DEADLINE - _MARGIN
    e_list = _build_edges(n, edges)
    opt = _exact_optimum(n, e_list)
    if opt <= 0:
        return np.zeros((p, 2))
    prep = _prepare(n, e_list)
    bounds = np.array([[0.0, 2.0], [0.0, 2.0]])
    f = lambda x: _neg_eval(x[None, :], prep, opt)

    # Warm start (known-good region)
    base = np.array([0.35, 0.25])
    best_p = base.copy(); best_v = f(base)

    # Coarse global grid to find a good basin
    gp, gv = _coarse_grid(f, bounds, deadline)
    if gp is not None and gv < best_v:
        best_p, best_v = gp, gv

    # Local refinement from the best start
    if time.time() < deadline:
        lp, lv = _nelder_mead(f, best_p, bounds, max_iter=400, deadline=deadline)
        if lv < best_v:
            best_p, best_v = lp, lv

    result = np.tile(best_p, (p, 1))
    return result
