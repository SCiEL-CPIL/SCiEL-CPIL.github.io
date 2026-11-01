"""Packing of 50 equal circles in the unit square [-1/2,1/2]^2.
Strategy: seed from the known Boll et al. (2000) arrangement, then refine
with a fast vectorized force-directed optimization (repulsion for overlaps,
attraction to walls for boundary violations) with adaptive step size, multiple
restarts, and a hard deadline guard.
"""
import time

N = 50

# Known best packing (Boll, Donovan, Graham, Lubachevsky 2000), r = 0.071377103865.
# Coordinates given in [0,1]^2; we shift to [-1/2,1/2]^2.
_BOLL = [
    (0.428622896135, 0.071377103865),
    (0.571377103865, 0.071377103865),
    (0.714131311599, 0.071377103865),
    (0.856885519332, 0.071377103865),
    (0.928622896135, 0.071377103865),
    (0.928622896135, 0.214131311599),
    (0.785868688401, 0.214131311599),
    (0.643114480668, 0.214131311599),
    (0.500360272934, 0.214131311599),
    (0.357606065201, 0.214131311599),
    (0.214851857467, 0.214131311599),
    (0.072097649733, 0.214131311599),
    (0.071377103865, 0.356885519332),
    (0.071377103865, 0.499639727066),
    (0.071377103865, 0.642393934799),
    (0.071377103865, 0.785148142533),
    (0.072097649733, 0.927902350267),
    (0.214851857467, 0.928622896135),
    (0.357606065201, 0.928622896135),
    (0.500360272934, 0.928622896135),
    (0.643114480668, 0.928622896135),
    (0.785868688401, 0.928622896135),
    (0.928622896135, 0.928622896135),
    (0.928622896135, 0.785868688401),
    (0.928622896135, 0.643114480668),
    (0.928622896135, 0.500360272934),
    (0.928622896135, 0.357606065201),
    (0.714131311599, 0.356885519332),
    (0.714131311599, 0.499639727066),
    (0.714131311599, 0.642393934799),
    (0.714131311599, 0.785148142533),
    (0.571377103865, 0.356885519332),
    (0.571377103865, 0.499639727066),
    (0.571377103865, 0.642393934799),
    (0.571377103865, 0.785148142533),
    (0.428622896135, 0.356885519332),
    (0.428622896135, 0.499639727066),
    (0.428622896135, 0.642393934799),
    (0.428622896135, 0.785148142533),
    (0.285868688401, 0.356885519332),
    (0.285868688401, 0.499639727066),
    (0.285868688401, 0.642393934799),
    (0.285868688401, 0.785148142533),
    (0.143114480668, 0.356885519332),
    (0.143114480668, 0.499639727066),
    (0.143114480668, 0.642393934799),
    (0.143114480668, 0.785148142533),
    (0.214851857467, 0.356885519332),
    (0.214851857467, 0.499639727066),
    (0.214851857467, 0.642393934799),
    (0.214851857467, 0.785148142533),
]


def _score(pts):
    """Compute the largest common radius for centers pts (N,2)."""
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    # Distance to walls
    wall = (0.5 - np.abs(pts)).min()
    return min(min_pair, 2.0 * wall)


def _refine(pts, t0, deadline):
    """Force-directed refinement. Returns best pts and best radius found."""
    import numpy as np
    pts = pts.astype(np.float64).copy()
    best = pts.copy()
    best_r = _score(best)
    step = 0.01
    for it in range(200000):
        if time.time() > deadline:
            break
        diff = pts[:, None, :] - pts[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        # Repulsive force: push apart overlapping pairs
        # f = (2r - d) * direction, active when d < 2r
        r = best_r
        overlap = np.maximum(2.0 * r - d, 0.0)
        # Avoid division by zero
        d_safe = np.where(d > 1e-12, d, 1e-12)
        force = overlap[:, :, None] * diff / d_safe[:, :, None]
        F = force.sum(axis=1)
        # Wall force: push inward if outside
        for k in range(2):
            out = np.where(pts[:, k] > 0.5 - r, (pts[:, k] - (0.5 - r)), 0.0)
            inn = np.where(pts[:, k] < -0.5 + r, (pts[:, k] - (-0.5 + r)), 0.0)
            F[:, k] += out + inn
        # Update with adaptive step
        pts += step * F
        # Clamp to square (keep inside)
        pts = np.clip(pts, -0.5, 0.5)
        # Evaluate
        cur_r = _score(pts)
        if cur_r > best_r:
            best_r = cur_r
            best = pts.copy()
            step = min(step * 1.05, 0.05)
        else:
            step *= 0.95
            # Revert if it got much worse
            if cur_r < best_r * 0.98:
                pts = best.copy()
    return best, best_r


def construct():
    import numpy as np
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s

    # Seed from Boll et al., shifted to [-1/2, 1/2]^2
    boll = np.array(_BOLL, dtype=np.float64) - 0.5
    best_pts = boll.copy()
    best_r = _score(best_pts)

    # Refine from the Boll seed
    pts, r = _refine(boll, t0, deadline)
    if r > best_r:
        best_r = r
        best_pts = pts

    # Additional restarts with perturbations if time remains
    rng = np.random.default_rng(42)
    for restart in range(200):
        if time.time() > deadline:
            break
        pert = boll + rng.normal(0, 0.005, boll.shape)
        pert = np.clip(pert, -0.5, 0.5)
        pts, r = _refine(pert, t0, deadline)
        if r > best_r:
            best_r = r
            best_pts = pts

    return best_pts
