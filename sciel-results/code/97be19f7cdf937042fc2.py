"""Lennard-Jones 55-atom cluster: relax many icosahedral seeds, keep the best."""
import time
import numpy as np

N = 55

# ---------- vectorized Lennard-Jones energy / gradient ----------
def _lj(pos):
    d = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", d, d)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** -3
    r12 = r6 * r6
    e = 4.0 * (r12 - r6)
    e.fill_diagonal(0.0)
    return 0.5 * e.sum()


def _lj_grad(pos):
    d = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", d, d)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** -3
    r12 = r6 * r6
    g = 48.0 * r12 - 24.0 * r6
    g = g * d
    g.fill_diagonal(0.0)
    return g.sum(axis=1)


def _valid(pos):
    d = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", d, d)
    np.fill_diagonal(r2, np.inf)
    return bool((r2 < 0.25).any() is False)


def _relax(pos):
    from scipy.optimize import minimize
    try:
        res = minimize(_lj, pos, jac=_lj_grad, method="L-BFGS-B",
                       options=dict(maxiter=3000, maxls=20, gtol=1e-7, ftol=1e-12))
        return np.asarray(res.x, dtype=float)
    except Exception:
        return pos


# ---------- icosahedral geometry ----------
_PHI = (1.0 + np.sqrt(5.0)) / 2.0
_ICS = np.array([
    [0, 1, _PHI], [0, -1, _PHI], [0, 1, -_PHI], [0, -1, -_PHI],
    [1, _PHI, 0], [-1, _PHI, 0], [1, -_PHI, 0], [-1, -_PHI, 0],
    [_PHI, 0, 1], [-_PHI, 0, 1], [_PHI, 0, -1], [-_PHI, 0, -1],
], dtype=float)
_ICS = _ICS / np.linalg.norm(_ICS, axis=1, keepdims=True)


def _edge_midpoints():
    mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            dot = _ICS[i] @ _ICS[j]
            if 0.9 < dot < 0.9999:      # adjacent vertices
                m = (_ICS[i] + _ICS[j]) / 2.0
                mids.append(m)
    mids = np.array(mids)
    mids = mids / np.linalg.norm(mids, axis=1, keepdims=True)
    return mids


_EDGE = _edge_midpoints()  # 30 edge midpoints (unit directions)


def _mackay(ri, ro, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    inner = _ICS * ri
    outer = np.vstack([_ICS, _EDGE]) * ro
    pos = np.vstack([np.zeros(3), inner, outer])
    if jitter > 0:
        pos = pos + rng.normal(0.0, jitter, size=pos.shape)
    return pos


def _cubic(seed):
    rng = np.random.default_rng(seed)
    side = int(np.ceil(N ** (1 / 3)))
    pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    c = (side - 1) / 2.0
    pts.sort(key=lambda p: (p[0] - c) ** 2 + (p[1] - c) ** 2 + (p[2] - c) ** 2)
    p = np.array(pts[:N], dtype=float) * 1.12
    p = p + rng.normal(0.0, 0.05, size=p.shape)
    return p


def construct():
    t0 = time.time()
    deadline = 118.0          # safety margin under the 120 s kill

    # seed set: exact Mackay, jitters, inner/outer radius variations, cubic
    seeds = [
        _mackay(1.122, 2.245, 0.0, 0),
        _mackay(1.122, 2.245, 0.05, 1),
        _mackay(1.122, 2.245, 0.12, 2),
        _mackay(1.122, 2.245, 0.25, 3),
        _mackay(1.10, 2.245, 0.0, 4),
        _mackay(1.15, 2.245, 0.0, 5),
        _mackay(1.122, 2.20, 0.0, 6),
        _mackay(1.122, 2.30, 0.0, 7),
        _mackay(1.122, 2.245, 0.0, 8),
        _cubic(101),
        _cubic(202),
        _cubic(303),
    ]

    best = None
    best_e = np.inf
    for s in seeds:
        if time.time() - t0 > deadline:
            break
        r = _relax(s)
        if _valid(r):
            e = _lj(r)
            if e < best_e:
                best_e = e
                best = r

    if best is None:
        # fallback: relaxed Mackay even if validity check is conservative
        r = _relax(_mackay(1.122, 2.245, 0.0, 0))
        return r
    return best
