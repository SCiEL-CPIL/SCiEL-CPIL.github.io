# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 630.0
_SAFE_MARGIN = 10.0


def _count_convex7_vec(points):
    """Count 7-subsets in convex position (all 7 on their convex hull), chunked."""
    from itertools import combinations
    n = len(points)
    idxs = np.array(list(combinations(range(n), 7)), dtype=np.int64)
    M = len(idxs)
    C = 0
    CH = 80000
    eye = np.eye(7, dtype=bool)
    for s in range(0, M, CH):
        e = min(s + CH, M)
        subs = points[idxs[s:e]]
        H = subs.shape[0]
        vecs = subs[:, :, None, :] - subs[:, None, :, :]
        ang = np.arctan2(vecs[:, :, 1], vecs[:, :, 0])
        ang_masked = np.where(eye[None, :, :], np.pi * 4, ang)
        ang_sorted = np.sort(ang_masked, axis=2)
        real = ang_sorted[:, :, :6]
        gaps = np.diff(np.concatenate([real, real[:, :, :1] + 2 * np.pi], axis=2))
        maxgap = gaps.max(axis=2)
        extreme = maxgap > np.pi + 1e-12
        C += int(extreme.all(axis=1).sum())
    return C


def _is_valid(pts):
    if pts.shape != (33, 2):
        return False
    if not np.all(np.isfinite(pts)):
        return False
    if np.any(pts < -1e-6) or np.any(pts > 1 + 1e-6):
        return False
    for i in range(33):
        for j in range(i + 1, 33):
            d2 = (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2
            if d2 < 1e-12:
                return False
    for i in range(33):
        for j in range(i + 1, 33):
            for k in range(j + 1, 33):
                a = pts[i][0] * (pts[j][1] - pts[k][1]) + \
                    pts[j][0] * (pts[k][1] - pts[i][1]) + \
                    pts[k][0] * (pts[i][1] - pts[j][1])
                if abs(a) < 1e-9:
                    return False
    return True


def erdos_szekeres_problem7() -> np.ndarray:
    """Minimize convex 7-gons among 33 points via random restarts + local search."""
    start = time.time()
    rng = np.random.default_rng(17)

    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start)

    best_pts = None
    best_c = None

    # Seed with a few random configs
    for _ in range(6):
        if time_left() < 60:
            break
        pts = rng.random((33, 2)) * 0.9 + 0.05
        if _is_valid(pts):
            c = _count_convex7_vec(pts)
            if best_c is None or c < best_c:
                best_c = c
                best_pts = pts.copy()

    if best_pts is None:
        pts = rng.random((33, 2)) * 0.9 + 0.05
        return np.clip(pts, 0, 1)

    # Local search: perturb one point, accept if C decreases
    while time_left() > 45:
        for _ in range(60):
            if time_left() < 45:
                break
            i = int(rng.integers(0, 33))
            new_pts = best_pts.copy()
            new_pts[i] = rng.random(2) * 0.9 + 0.05
            if not _is_valid(new_pts):
                continue
            c = _count_convex7_vec(new_pts)
            if c < best_c:
                best_c = c
                best_pts = new_pts
                if best_c <= 1:
                    return best_pts
    return best_pts


# EVOLVE-BLOCK-END
