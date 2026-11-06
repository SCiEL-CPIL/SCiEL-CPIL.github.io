"""Fresh design: concentric-ring layout + fast deterministic repulsion refinement."""
import numpy as np
import time

N = 50


def _score(pts):
    """Return (valid, r) where r is the max common radius."""
    if pts is None:
        return False, 0.0
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return False, 0.0
    d = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(d >= 1.0):
        return False, 0.0
    r_cont = float(np.min(1.0 - d))
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    r_pair = float(np.min(dist[iu])) / 2.0
    r = min(r_cont, r_pair)
    return True, r


def _ring_layout():
    """Concentric rings: 1 centre + rings of 6,12,18,24 (sums to 50)."""
    pts = [(0.0, 0.0)]
    rings = [6, 12, 18, 24]
    for m in rings:
        r = m / (2.0 * np.pi)
        for k in range(m):
            a = 2.0 * np.pi * k / m
            pts.append((r * np.cos(a), r * np.sin(a)))
    return np.array(pts, dtype=np.float64)


def _repulse(pts, steps, alpha):
    """Deterministic repulsion: push overlapping pairs apart, keep inside disk."""
    pts = pts.copy()
    for s in range(steps):
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_dist = np.min(dist, axis=1)
        # desired gap ~ current min pair distance; use a target
        target = 0.26
        # force: for each point, sum of repulsion from closer-than-target neighbours
        # build per-point displacement
        disp = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = dist[i, j]
                if d < target:
                    # push i away from j
                    w = (target - d) / d
                    disp[i] += w * (pts[i] - pts[j])
        pts += alpha * disp
        # clamp inside disk with margin
        rr = np.hypot(pts[:, 0], pts[:, 1])
        margin = 0.0
        for i in range(N):
            if rr[i] > 1.0 - margin:
                scale = (1.0 - margin) / rr[i]
                pts[i] *= scale
    return pts


def construct():
    t0 = time.time()
    deadline = 110.0  # safety margin under 120s

    best_pts = None
    best_r = 0.0

    # --- Phase 1: structured seed + coarse repulsion ---
    seed = _ring_layout()
    # coarse repulsion to resolve overlaps
    coarse = _repulse(seed, steps=200, alpha=0.5)
    ok, r = _score(coarse)
    if ok:
        best_pts, best_r = coarse.copy(), r

    # --- Phase 2: fine-tune with smaller alpha, more steps ---
    if time.time() - t0 < deadline:
        fine = _repulse(best_pts, steps=400, alpha=0.15)
        ok, r = _score(fine)
        if ok and r > best_r:
            best_pts, best_r = fine.copy(), r

    # --- Phase 3: additional polish passes while time remains ---
    alpha = 0.05
    while time.time() - t0 < deadline:
        polished = _repulse(best_pts, steps=300, alpha=alpha)
        ok, r = _score(polished)
        if ok and r > best_r:
            best_pts, best_r = polished.copy(), r
            alpha = max(0.005, alpha * 0.5)
        else:
            break

    # ensure a valid fallback
    if best_pts is None or not _score(best_pts)[0]:
        best_pts = _ring_layout()

    return best_pts
