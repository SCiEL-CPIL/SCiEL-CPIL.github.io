# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 600.0  # seconds; evaluator kills at 630s, keep margin


def _make_P(support):
    s = np.asarray(support, dtype=np.float64)
    n = len(s)
    ps = s[:, None] + s[None, :]
    T = 2.0 * s[None, :] - s[:, None]
    I = (ps[None, None, :, :] < T[:, :, None, None])  # (k, l, i, j)
    def P(w):
        w = np.asarray(w, dtype=np.float64)
        tot = w.sum()
        if tot <= 1e-12:
            return 0.0
        w = w / tot
        W2 = w[:, None] * w[None, :]
        G = np.einsum('kl,klij->ij', W2, I)
        return float(np.einsum('ij,ij->', W2, G))
    return P


def _deadline_ok(margin=30.0):
    return (time.time() - _START) < (_DEADLINE - margin)


def _search_3pt_zero(M_grid, u_grid, t_grid, K_refine):
    # support {0, 1, M}; weights: w0 = u*t, w1 = 1-t, wM = (1-u)*t
    best_val = -1.0
    best_cfg = None
    results = []
    for M in M_grid:
        Pfn = _make_P(np.array([0.0, 1.0, M]))
        for u in u_grid:
            for t in t_grid:
                w = np.array([u * t, 1.0 - t, (1.0 - u) * t])
                v = Pfn(w)
                results.append((v, M, u, t))
                if v > best_val:
                    best_val = v
                    best_cfg = (M, u, t)
        if not _deadline_ok():
            break
    # refine top K
    results.sort(key=lambda x: x[0], reverse=True)
    for v, M, u, t in results[:K_refine]:
        if not _deadline_ok():
            break
        x0 = np.array([np.log(M), u, t])
        def obj(x):
            Mv = np.exp(x[0])
            uv = min(max(x[1], 1e-7), 1.0 - 1e-7)
            tv = min(max(x[2], 1e-7), 1.0 - 1e-7)
            Pfn = _make_P(np.array([0.0, 1.0, Mv]))
            w = np.array([uv * tv, 1.0 - tv, (1.0 - uv) * tv])
            return -Pfn(w)
        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True})
            Mv = np.exp(res.x[0])
            uv = min(max(res.x[1], 1e-7), 1.0 - 1e-7)
            tv = min(max(res.x[2], 1e-7), 1.0 - 1e-7)
            Pfn = _make_P(np.array([0.0, 1.0, Mv]))
            w = np.array([uv * tv, 1.0 - tv, (1.0 - uv) * tv])
            v2 = Pfn(w)
            if v2 > best_val:
                best_val = v2
                best_cfg = (Mv, uv, tv)
        except Exception:
            pass
    if best_cfg is None:
        return -1.0, None
    Mv, uv, tv = best_cfg
    s = np.array([0.0, 1.0, Mv])
    w = np.array([uv * tv, 1.0 - tv, (1.0 - uv) * tv])
    return best_val, (s, w)


def _search_3pt_nz(a_grid, b_grid, p1_grid, p2_grid, K_refine):
    # support {a, b, 1}, 0 < a < b < 1; weights (p1, p2, 1-p1-p2)
    best_val = -1.0
    best_cfg = None
    results = []
    for a in a_grid:
        for b in b_grid:
            if b <= a:
                continue
            Pfn = _make_P(np.array([a, b, 1.0]))
            for p1 in p1_grid:
                for p2 in p2_grid:
                    p3 = 1.0 - p1 - p2
                    if p3 <= 1e-9:
                        continue
                    w = np.array([p1, p2, p3])
                    v = Pfn(w)
                    results.append((v, a, b, p1, p2))
                    if v > best_val:
                        best_val = v
                        best_cfg = (a, b, p1, p2)
            if not _deadline_ok():
                break
        if not _deadline_ok():
            break
    results.sort(key=lambda x: x[0], reverse=True)
    for v, a, b, p1, p2 in results[:K_refine]:
        if not _deadline_ok():
            break
        x0 = np.array([np.log(a), np.log(b), p1, p2])
        def obj(x):
            av = min(max(np.exp(x[0]), 1e-6), 0.999)
            bv = min(max(np.exp(x[1]), 1e-6), 0.999)
            if bv <= av:
                bv = av + 1e-6
            av = min(av, bv - 1e-6)
            p1v = min(max(x[2], 1e-6), 1.0 - 1e-6)
            p2v = min(max(x[3], 1e-6), 1.0 - 1e-6)
            p3v = 1.0 - p1v - p2v
            if p3v <= 1e-9:
                return -1e9
            Pfn = _make_P(np.array([av, bv, 1.0]))
            w = np.array([p1v, p2v, p3v])
            return -Pfn(w)
        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True})
            av = min(max(np.exp(res.x[0]), 1e-6), 0.999)
            bv = min(max(np.exp(res.x[1]), 1e-6), 0.999)
            if bv <= av:
                bv = av + 1e-6
            av = min(av, bv - 1e-6)
            p1v = min(max(res.x[2], 1e-6), 1.0 - 1e-6)
            p2v = min(max(res.x[3], 1e-6), 1.0 - 1e-6)
            p3v = 1.0 - p1v - p2v
            if p3v <= 1e-9:
                continue
            Pfn = _make_P(np.array([av, bv, 1.0]))
            w = np.array([p1v, p2v, p3v])
            v2 = Pfn(w)
            if v2 > best_val:
                best_val = v2
                best_cfg = (av, bv, p1v, p2v)
        except Exception:
            pass
    if best_cfg is None:
        return -1.0, None
    a, b, p1, p2 = best_cfg
    s = np.array([a, b, 1.0])
    w = np.array([p1, p2, 1.0 - p1 - p2])
    return best_val, (s, w)


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    global _START
    _START = time.time()
    best_val = -1.0
    best_sw = None

    # Family 1: {0, 1, M}
    if _deadline_ok():
        M_grid = np.logspace(np.log10(0.2), np.log10(300.0), 90)
        u_grid = np.linspace(0.002, 0.998, 34)
        t_grid = np.linspace(0.002, 0.998, 34)
        v, sw = _search_3pt_zero(M_grid, u_grid, t_grid, K_refine=40)
        if sw is not None and v > best_val:
            best_val, best_sw = v, sw

    # Family 2: {a, b, 1} (no zero)
    if _deadline_ok():
        a_grid = np.linspace(0.005, 0.6, 14)
        b_grid = np.linspace(0.1, 0.99, 14)
        p1_grid = np.linspace(0.005, 0.99, 16)
        p2_grid = np.linspace(0.005, 0.99, 16)
        v, sw = _search_3pt_nz(a_grid, b_grid, p1_grid, p2_grid, K_refine=30)
        if sw is not None and v > best_val:
            best_val, best_sw = v, sw

    # Fallback to seed if nothing found
    if best_sw is None:
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        return support, weights

    support, weights = best_sw
    weights = weights / weights.sum()
    support = np.asarray(support, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    return support, weights
# EVOLVE-BLOCK-END
