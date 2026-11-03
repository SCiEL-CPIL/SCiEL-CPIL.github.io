import numpy as np
import time

N = 300


def _hex_init(n=N):
    """Hexagonal lattice clipped to the unit disk (guaranteed valid fallback)."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        keep = rad <= 1.0 - spacing / 2.0
        pts = pts[keep]
        if len(pts) >= n:
            idx = np.argsort(rad, kind="stable")[:n]
            return pts[idx]
        spacing *= 0.98


def _ring_init(n=N, seed=0):
    """Concentric-ring layout: outer rings hold more circles (use the rim)."""
    rng = np.random.default_rng(seed)
    pts = [np.array([0.0, 0.0])]
    placed = 1
    ring = 1
    while placed < n:
        m = min(n - placed, int(6 * ring))
        if m <= 0:
            m = 1
        radii = np.linspace(0.10, 0.97, max(ring, 1))
        r_ring = radii[ring - 1]
        angles = (np.arange(m) + 0.5 / m) * 2.0 * np.pi
        angles += rng.uniform(0, 2.0 * np.pi)
        pts.append(np.column_stack([r_ring * np.cos(angles), r_ring * np.sin(angles)]))
        placed += m
        ring += 1
    pts = np.vstack(pts)
    return pts[:n]


def _score(pts):
    """Packing radius r = min(min_pair_dist/2, min(1-|c_i|))."""
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    min_bound = 1.0 - np.linalg.norm(pts, axis=1).max()
    return min(min_pair, min_bound)


def _neg_score(pts):
    return -_score(pts.reshape(-1, 2))


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before the 150s kill

    # Guaranteed-valid fallback.
    best_pts = _hex_init(N)
    best_r = _score(best_pts)

    # Direct optimization of the packing radius via Nelder-Mead.
    try:
        from scipy.optimize import minimize
    except Exception:
        return best_pts

    for seed in (0, 1, 2, 3, 4, 5):
        if time.time() > deadline - 45.0:
            break
        x0 = _ring_init(N, seed).ravel()
        try:
            res = minimize(_neg_score, x0, method="Nelder-Mead",
                           options={"maxiter": 400, "xatol": 1e-5, "fatol": 1e-8,
                                    "adaptive": True})
        except Exception:
            continue
        cand = res.x.reshape(-1, 2)
        if not np.all(np.isfinite(cand)):
            continue
        r = _score(cand)
        if r > best_r:
            best_r = r
            best_pts = cand

    return best_pts
