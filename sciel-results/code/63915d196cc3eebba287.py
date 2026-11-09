"""Sutton-Chen transition-metal cluster global minima: coarse-to-fine annealing."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_PARAMS = {
    "12-6": (12, 6, 144.41),
    "9-6": (9, 6, 39.432),
    "10-8": (10, 8, 34.408),
}


def _sc_energy(pos, ne, me, c):
    n = pos.shape[0]
    d2 = np.sum(pos * pos, axis=1)
    r2 = d2[:, None] + d2[None, :] - 2.0 * (pos @ pos.T)
    np.maximum(r2, 1e-12, out=r2)
    np.fill_diagonal(r2, 1.0)
    r = np.sqrt(r2)
    rne = r ** (-ne)
    rme = r ** (-me)
    np.fill_diagonal(rne, 0.0)
    np.fill_diagonal(rme, 0.0)
    s = rme.sum(axis=1)
    e = 0.5 * rne.sum() - c * np.sqrt(np.maximum(s, 0.0)).sum()
    return float(e)


def _sc_grad(pos, ne, me, c):
    n = pos.shape[0]
    d2 = np.sum(pos * pos, axis=1)
    r2 = d2[:, None] + d2[None, :] - 2.0 * (pos @ pos.T)
    np.maximum(r2, 1e-12, out=r2)
    np.fill_diagonal(r2, 1.0)
    r = np.sqrt(r2)
    rne = r ** (-ne)
    rme = r ** (-me)
    np.fill_diagonal(rne, 0.0)
    np.fill_diagonal(rme, 0.0)
    s = rme.sum(axis=1)
    s = np.maximum(s, 1e-12)
    dE = -0.5 * ne * rne + 0.5 * c * rme / np.sqrt(s)
    np.fill_diagonal(dE, 0.0)
    grad = -2.0 * (dE * (pos[:, None] - pos[None, :])).sum(axis=0)
    return grad


def _sc_energy_gradient(pos, ne, me, c):
    return _sc_energy(pos, ne, me, c), _sc_grad(pos, ne, me, c)


def _soften_params(ne, me, c, t):
    """Return (ne_t, me_t, c_t) with t in [0,1]; t=0 is softened, t=1 is target."""
    ne_t = 4.0 + (ne - 4.0) * t
    me_t = 4.0 + (me - 4.0) * t
    c_t = c * ((4.0 / ne) / (4.0 / ne_t))
    return ne_t, me_t, c_t


def _valid(pos):
    if not np.all(np.isfinite(pos)):
        return False
    d2 = np.sum(pos * pos, axis=1)
    r2 = d2[:, None] + d2[None, :] - 2.0 * (pos @ pos.T)
    np.maximum(r2, 0.0, out=r2)
    np.fill_diagonal(r2, 1.0)
    return bool(np.sqrt(r2).min() >= 0.35)


def _relax(pos, ne, me, c, maxiter=250):
    try:
        res = minimize(
            _sc_energy_gradient, pos, jac=True, method="L-BFGS-B",
            options={"maxiter": maxiter, "maxls": 40},
        )
        if res.success or np.isfinite(res.fun):
            return float(res.fun), res.x
    except Exception:
        pass
    return None, pos


def _fcc(n, spacing, rng):
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    m = int(np.ceil(n ** (1 / 3))) + 2
    pts = []
    for i in range(-m, m + 1):
        for j in range(-m, m + 1):
            for k in range(-m, m + 1):
                for b in basis:
                    p = np.array([i, j, k]) + b
                    pts.append((np.dot(p, p), p))
    pts.sort(key=lambda x: x[0])
    out = [p * spacing for _, p in pts[:n]]
    if len(out) < n:
        rad = spacing * n ** (1 / 3) * 0.7
        while len(out) < n:
            q = rng.uniform(-rad, rad, 3)
            if all(np.linalg.norm(q - a) >= 0.55 * spacing for a in out):
                out.append(q)
    return np.array(out[:n])


def _hcp(n, spacing, rng):
    a = spacing
    h = np.sqrt(2.0 / 3.0) * a
    pts = []
    m = int(np.ceil(np.sqrt(n / 2))) + 2
    for layer in range(-m, m + 1):
        z = layer * h
        off = 0.5 if (layer % 2) else 0.0
        for i in range(-m, m + 1):
            for j in range(-m, m + 1):
                x = (i + off) * a
                y = (j * np.sqrt(3.0) / 2.0 + (0.5 * np.sqrt(3.0) / 2.0 if (i % 2) else 0.0)) * a
                pts.append((z * z + x * x + y * y, np.array([x, y, z])))
    pts.sort(key=lambda x: x[0])
    out = [p * 1.0 for _, p in pts[:n]]
    if len(out) < n:
        rad = spacing * n ** (1 / 3) * 0.7
        while len(out) < n:
            q = rng.uniform(-rad, rad, 3)
            if all(np.linalg.norm(q - a) >= 0.55 * spacing for a in out):
                out.append(q)
    return np.array(out[:n])


def _random(n, spacing, rng):
    rad = spacing * n ** (1 / 3) * 0.72
    pts = []
    while len(pts) < n:
        p = rng.uniform(-rad, rad, 3)
        if np.linalg.norm(p) <= rad and all(np.linalg.norm(p - q) >= 0.55 * spacing for q in pts):
            pts.append(p)
    if len(pts) < n:
        while len(pts) < n:
            pts.append(rng.uniform(-rad, rad, 3))
    return np.array(pts[:n])


def construct(instance):
    t0 = time.time()
    deadline = 320.0
    size = instance["n"]
    kind = instance.get("kind", "12-6")
    ne, me, c = _PARAMS.get(kind, (12, 6, 144.41))
    ne = int(instance.get("n_exp", ne))
    me = int(instance.get("m_exp", me))
    c = float(instance.get("c", c))
    spacing = 0.72
    rng = np.random.default_rng(size * 1000 + ne * 100 + me)

    starts = [
        _fcc(size, spacing, rng),
        _hcp(size, spacing, rng),
        _random(size, spacing, rng),
        _random(size, spacing, rng),
    ]

    best_pos = None
    best_e = np.inf
    n_steps = 40
    t0temp = 0.06
    tcool = 0.35

    def update(pos, e):
        nonlocal best_pos, best_e
        if _valid(pos) and e < best_e:
            best_e = e
            best_pos = pos.copy()

    for s in starts:
        if time.time() > deadline:
            break
        pos = s.copy()
        # Phase 1: coarse-to-fine annealing with basin hopping on softened potential
        for step in range(n_steps):
            if time.time() > deadline:
                break
            t = step / max(1, n_steps - 1)
            ne_t, me_t, c_t = _soften_params(ne, me, c, t)
            # local relax on current (softened) potential
            e, pos = _relax(pos, ne_t, me_t, c_t, maxiter=200)
            if e is None:
                break
            update(pos, e)
            # basin-hopping move: kick amplitude anneals with temperature
            temp = t0temp * (1.0 - tcool * t)
            amp = 0.12 + 0.18 * (1.0 - t)
            kick = rng.normal(0.0, amp, size=pos.shape)
            cand = pos + kick
            e_c, cand = _relax(cand, ne_t, me_t, c_t, maxiter=200)
            if e_c is None:
                e_c, cand = e, pos
            # Metropolis accept
            if e_c <= e or (np.random.random() < np.exp(-(e_c - e) / max(temp, 1e-9))):
                pos = cand
                e = e_c
        # Phase 2: refine on the true potential
        e, pos = _relax(pos, ne, me, c, maxiter=400)
        if e is not None:
            update(pos, e)

    if best_pos is None:
        # Fallback: direct relax of an FCC start on the true potential
        e, pos = _relax(starts[0], ne, me, c, maxiter=400)
        if e is None or not _valid(pos):
            return starts[0]
        return pos
    return best_pos
# EVOLVE-BLOCK-END
