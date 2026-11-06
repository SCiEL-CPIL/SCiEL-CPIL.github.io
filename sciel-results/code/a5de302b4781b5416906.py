"""Search-portfolio solver for Morse cluster global minima (rewritten)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import concurrent.futures as _cf
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_POOL = None
_T0 = None
_HARD_DEADLINE = None


def _init_timer():
    global _T0, _HARD_DEADLINE
    if _T0 is None:
        _T0 = time.time()
        _HARD_DEADLINE = _T0 + 285.0
    return _T0


def _morse_E(P, rho):
    """Exact energy."""
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    D = P[:, None, :] - P[None, :, :]
    r = np.sqrt((D * D).sum(axis=2))
    np.fill_diagonal(r, np.inf)
    t = np.exp(rho * (1.0 - r))
    return float((t * (t - 2.0)).sum() / 2.0)


def _morse_grad(P, rho):
    """Gradient of Morse energy."""
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    D = P[:, None, :] - P[None, :, :]
    r = np.sqrt((D * D).sum(axis=2))
    np.fill_diagonal(r, np.inf)
    t = np.exp(rho * (1.0 - r))
    c = rho * t * (2.0 * t - 2.0)
    inv_r = np.where(r > 1e-12, 1.0 / r, 0.0)
    W = c * inv_r
    rowsum = W.sum(axis=1)
    grad = -(rowsum[:, None] * P - W @ P)
    return grad


def _enforce_min_dist(P, min_d=0.5):
    P = np.array(P, dtype=float, copy=True)
    n = P.shape[0]
    for _ in range(500):
        d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        bad = (d < min_d)
        np.fill_diagonal(bad, False)
        if not bad.any():
            break
        idx = np.argwhere(bad)
        for i, j in idx[:20]:
            gap = min_d - d[i, j] + 0.02
            v = P[i] - P[j]
            nv = np.linalg.norm(v)
            if nv < 1e-9:
                v = np.array([1.0, 0.0, 0.0]); nv = 1.0
            v /= nv
            P[i] += 0.5 * gap * v
            P[j] -= 0.5 * gap * v
    return P


def _local_opt(P0, rho, max_iter=500):
    P0 = np.asarray(P0, dtype=float)
    n = P0.shape[0]

    def fun(x):
        P = x.reshape(n, 3)
        return _morse_E(P, rho)

    def gfun(x):
        P = x.reshape(n, 3)
        return _morse_grad(P, rho).ravel()

    try:
        res = minimize(fun, P0.ravel(), jac=gfun, method="L-BFGS-B",
                       options={"maxiter": max_iter, "maxfun": 30000,
                                "ftol": 1e-13, "gtol": 1e-9, "iprint": 0})
        P = res.x.reshape(n, 3)
        E = _morse_E(P, rho)
        return E, P
    except Exception:
        return None, None


def _relax_one(pos, rho):
    P = np.asarray(pos, dtype=float).reshape(len(pos), 3)
    best_E, best_P = None, None
    try:
        e, P2 = relax(P, "morse", rho)
        P2 = np.asarray(P2, dtype=float).reshape(len(pos), 3)
        E2 = _morse_E(P2, rho)
        best_E, best_P = E2, P2
    except Exception:
        pass
    start = best_P if best_P is not None else P
    E3, P3 = _local_opt(start, rho)
    if E3 is not None and (best_E is None or E3 < best_E):
        best_E, best_P = E3, P3
    if best_E is None:
        E3, P3 = _local_opt(P, rho)
        if E3 is not None:
            best_E, best_P = E3, P3
    return best_E, best_P


def _get_pool():
    global _POOL
    if _POOL is None:
        _POOL = _cf.ThreadPoolExecutor(max_workers=4)
    return _POOL


def _run_batch(cands, rho, deadline, best):
    pool = _get_pool()
    futs = {}
    for c in cands:
        futs[pool.submit(_relax_one, c, rho)] = c
    while futs:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        done, _ = _cf.wait(list(futs.keys()), timeout=min(remaining, 0.3),
                           return_when=_cf.FIRST_COMPLETED)
        for f in done:
            try:
                E, P = f.result()
            except Exception:
                E, P = None, None
            if E is not None and (best[0] is None or E < best[0]):
                best[0], best[1] = E, P
            futs.pop(f, None)
    return best


def _sunflower(n, R, jitter, rng):
    i = np.arange(n)
    t = (i + 0.5) / n
    r = R * t ** (1.0 / 3.0)
    z = 1.0 - 2.0 * t
    th = 2.0 * np.pi * i * 1.6180339887498949
    xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    P = np.column_stack([r * xy * np.cos(th), r * xy * np.sin(th), r * z])
    if jitter > 0:
        P = P + rng.normal(0.0, jitter, P.shape)
    return P


def _fcc(n, rng, jitter=0.0):
    a = 1.0
    pts = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                pts.append([i * a, j * a, k * a])
                pts.append([(i + 0.5) * a, (j + 0.5) * a, k * a])
                pts.append([(i + 0.5) * a, j * a, (k + 0.5) * a])
                pts.append([i * a, (j + 0.5) * a, (k + 0.5) * a])
    pts = np.array(pts)
    order = np.argsort(np.linalg.norm(pts, axis=1))
    pts = pts[order][:n]
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, pts.shape)
    return pts


def _bcc(n, rng, jitter=0.0):
    a = 1.0
    pts = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                pts.append([i * a, j * a, k * a])
                pts.append([(i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a])
    pts = np.array(pts)
    # Remove duplicates
    if len(pts) > 1:
        keep = [0]
        for idx in range(1, len(pts)):
            if np.all(np.linalg.norm(pts[idx] - pts[keep], axis=1) > 1e-6):
                keep.append(idx)
        pts = pts[keep]
    order = np.argsort(np.linalg.norm(pts, axis=1))
    pts = pts[order][:n]
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, pts.shape)
    return pts


def _hcp(n, rng, jitter=0.0):
    a = 1.0
    c = np.sqrt(8.0 / 3.0) * a
    pts = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = a * (i + 0.5 * (j % 2))
            y = c / 2 * j
            z = 0.0
            pts.append([x, y, z])
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = a * (i + 0.5 * (j % 2)) + a / 2
            y = c / 2 * j
            z = c / 2
            pts.append([x, y, z])
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = a * (i + 0.5 * (j % 2))
            y = c / 2 * j
            z = c
            pts.append([x, y, z])
    pts = np.array(pts)
    order = np.argsort(np.linalg.norm(pts, axis=1))
    pts = pts[order][:n]
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, pts.shape)
    return pts


def _random_pack(n, R, rng, min_d):
    pts = []
    tries = 0
    while len(pts) < n and tries < n * 3000:
        tries += 1
        p = rng.uniform(-R, R, 3)
        if np.linalg.norm(p) > R:
            continue
        if all(np.linalg.norm(p - q) >= min_d for q in pts):
            pts.append(p)
    while len(pts) < n:
        pts.append(rng.uniform(-R, R, 3))
    return np.array(pts)


def _make_seeds(n, rng):
    base = n ** (1.0 / 3.0)
    seeds = []
    # Structured seeds (lattice-based, good for Morse)
    seeds.append(_fcc(n, rng, 0.0))
    seeds.append(_bcc(n, rng, 0.0))
    seeds.append(_hcp(n, rng, 0.0))
    seeds.append(_fcc(n, rng, 0.08))
    seeds.append(_bcc(n, rng, 0.08))
    seeds.append(_hcp(n, rng, 0.08))
    # Sunflower
    for c in (0.85, 1.0, 1.15, 0.7):
        seeds.append(_sunflower(n, base * c, 0.0, rng))
        seeds.append(_sunflower(n, base * c, 0.12, rng))
    # Random
    seeds.append(_random_pack(n, base * 1.15, rng, 0.6))
    seeds.append(_random_pack(n, base * 0.95, rng, 0.55))
    seeds.append(rng.uniform(-base, base, size=(n, 3)))
    return seeds


def construct(instance):
    _init_timer()
    size = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(size * 1000003 + 17)

    call_start = time.time()
    deadline = min(_HARD_DEADLINE, call_start + 40.0)
    best = [None, None]

    # Phase 1: relax diverse seeds in parallel
    _run_batch(_make_seeds(size, rng), rho, deadline, best)

    if best[1] is None:
        P = _fcc(size, rng, 0.0)
        best = [_morse_E(P, rho), P]

    # Phase 2: basin hopping
    kick_scales = [0.3, 0.5, 0.8, 1.2, 0.4, 0.6, 1.0, 1.5, 0.2, 2.0]
    kick_idx = 0
    round_no = 0
    minima = [best[1].copy()]
    while time.time() < deadline - 0.5:
        ks = kick_scales[kick_idx % len(kick_scales)]
        kick_idx += 1
        round_no += 1
        cands = []
        for m in minima[:5]:
            for _k in range(4):
                kick = ks * (0.6 + 0.8 * rng.random())
                cands.append(m + rng.normal(0.0, kick, m.shape))
        if round_no % 2 == 0:
            cands.append(_fcc(size, rng, 0.1))
            cands.append(_bcc(size, rng, 0.1))
            cands.append(_hcp(size, rng, 0.1))
            cands.append(_sunflower(size, size ** (1.0 / 3.0) * (0.8 + 0.4 * rng.random()), 0.15, rng))
            cands.append(_random_pack(size, size ** (1.0 / 3.0) * 1.1, rng, 0.6))
        _run_batch(cands, rho, deadline, best)
        if best[1] is not None:
            if best[1] is not minima[0]:
                minima = [best[1].copy()] + [m for m in minima if m is not best[1]][:4]
                minima = minima[:5]

    # Final polish
    if time.time() < _HARD_DEADLINE - 1.0 and best[1] is not None:
        E, P = _local_opt(best[1], rho, max_iter=800)
        if E is not None and E < best[0]:
            best[0], best[1] = E, P
        for _ in range(8):
            if time.time() > _HARD_DEADLINE - 2.0:
                break
            kick = 0.2 + 0.5 * rng.random()
            Pk = best[1] + rng.normal(0.0, kick, best[1].shape)
            Ek, Pk = _local_opt(Pk, rho, max_iter=500)
            if Ek is not None and Ek < best[0]:
                best[0], best[1] = Ek, Pk

    P = _enforce_min_dist(best[1], 0.5)
    return P
# EVOLVE-BLOCK-END
