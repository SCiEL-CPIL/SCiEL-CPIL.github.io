"""Fifty equal circles in a unit circle: fresh design (force-based + polish)."""
import time
import numpy as np

N = 50
_T0 = time.time()
_DEADLINE = _T0 + 130.0  # safety margin before the 150s kill


def _score(pts):
    """Raw objective: largest common radius for a valid packing."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    d = 1.0 - np.linalg.norm(pts, axis=1)
    r = d.min()
    if r <= 0:
        return 0.0
    # pairwise closest distance
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))
    np.fill_diagonal(dist, np.inf)
    r = min(r, dist.min() / 2.0)
    return float(r)


def _initial():
    """Hexagonal lattice cut to the unit disk, choose N most central points."""
    r = 0.1258254895304
    s = 2.0 * r
    k = int(1.0 / s) + 3
    rows = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            rows.append((x, y))
    pts = np.array(rows, dtype=float)
    rad = np.linalg.norm(pts, axis=1)
    pts = pts[rad <= 1.0 - s / 2.0]
    rad = rad[rad <= 1.0 - s / 2.0]
    order = np.argsort(rad, kind="stable")
    return pts[order[:N]]


def _force_step(pts, r, alpha, beta, max_iter=600):
    """Repulsive-force relaxation. Returns best (score, pts) seen."""
    n = len(pts)
    best_score = _score(pts)
    best = pts.copy()
    for it in range(max_iter):
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt((diff * diff).sum(-1))
        np.fill_diagonal(dist, 1e9)
        # repulsion (strong when close, gentle when far)
        rep = np.where(dist < 1e-6, 0.0, (1.0 / dist) ** 3)
        rep[dist > 2.0] = 0.0
        np.fill_diagonal(rep, 0.0)
        f = (rep[:, :, None] * diff).sum(1)
        # inward containment force
        rad = np.linalg.norm(pts, axis=1, keepdims=True)
        unit = np.where(rad > 1e-9, pts / np.where(rad > 1e-9, rad, 1.0), 0.0)
        f += beta * np.where(rad > 1.0, (rad - 1.0), 0.0) * unit
        pts = pts + alpha * f
        s = _score(pts)
        if s > best_score:
            best_score = s
            best = pts.copy()
        if time.time() > _DEADLINE - 5.0:
            break
    return best_score, best


def _polish(pts):
    """High-precision polish: maximize the min constraint with tiny steps."""
    best_score = _score(pts)
    best = pts.copy()
    n = len(pts)
    # decreasing step sizes
    steps = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    for step in steps:
        for it in range(400):
            if time.time() > _DEADLINE - 5.0:
                return best_score, best
            diff = pts[:, None, :] - pts[None, :, :]
            dist = np.sqrt((diff * diff).sum(-1))
            np.fill_diagonal(dist, 1e9)
            rep = np.where(dist < 1e-6, 0.0, (1.0 / dist) ** 3)
            rep[dist > 2.0] = 0.0
            np.fill_diagonal(rep, 0.0)
            f = (rep[:, :, None] * diff).sum(1)
            rad = np.linalg.norm(pts, axis=1, keepdims=True)
            unit = np.where(rad > 1e-9, pts / np.where(rad > 1e-9, rad, 1.0), 0.0)
            f += 5000.0 * np.where(rad > 1.0, (rad - 1.0), 0.0) * unit
            # adaptive: only move if it improves
            cand = pts + step * f
            s = _score(cand)
            if s > best_score:
                best_score = s
                best = cand
                pts = cand
            else:
                # try a small random nudge to escape local plateaus
                if it % 50 == 0:
                    cand = pts + step * 0.5 * np.random.randn(n, 2)
                    s2 = _score(cand)
                    if s2 > best_score:
                        best_score = s2
                        best = cand
                        pts = cand
    return best_score, best


def construct():
    rng = np.random.default_rng(12345)
    best_score = 0.0
    best_pts = _initial()
    best_score = _score(best_pts)

    # Phase 1: force relaxation from a few different starts
    for restart in range(3):
        if time.time() > _DEADLINE - 20.0:
            break
        if restart == 0:
            pts = _initial()
        else:
            pts = best_pts + 0.05 * rng.standard_normal((N, 2))
        for alpha in [0.05, 0.03, 0.02, 0.01, 0.005]:
            if time.time() > _DEADLINE - 15.0:
                break
            s, pts = _force_step(pts, 0.125, alpha, beta=100.0, max_iter=400)
            if s > best_score:
                best_score = s
                best_pts = pts

    # Phase 2: high-precision polish
    if time.time() < _DEADLINE - 8.0:
        s, pts = _polish(best_pts)
        if s > best_score:
            best_score = s
            best_pts = pts

    # Final safety: ensure validity
    if _score(best_pts) < 0.0:
        fb = _initial()
        return fb
    return best_pts
