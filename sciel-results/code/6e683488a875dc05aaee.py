# EVOLVE-BLOCK-START
import numpy as np
import time


def _is_valid(pts):
    if pts.shape != (33, 2):
        return False
    if np.any(~np.isfinite(pts)):
        return False
    if np.any(pts < -1e-6) or np.any(pts > 1 + 1e-6):
        return False
    # Pairwise squared distance
    d2 = (pts[:, 0:1] - pts[0:1, 0]) ** 2 + (pts[:, 1:2] - pts[0:1, 1]) ** 2
    np.fill_diagonal(d2, 1.0)
    if d2.min() < 1e-12:
        return False
    # Collinearity
    for i in range(33):
        xi, yi = pts[i]
        for j in range(i + 1, 33):
            xj, yj = pts[j]
            for k in range(j + 1, 33):
                xk, yk = pts[k]
                area2 = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                if area2 < 1e-9:
                    return False
    return True


def _make_concave_arc(m=28, x0=0.02, x1=0.98):
    # Concave upward arc: y = c*(1 - (2t-1)^2) scaled to fit [0,1]
    # t in [x0, x1], y from low at ends to high in middle
    x = np.linspace(x0, x1, m)
    t = (x - 0.5) / 0.5
    y_base = 0.85 * (1.0 - t * t) + 0.05
    return np.column_stack([x, y_base])


def erdos_szekeres_problem7() -> np.ndarray:
    """Minimize convex 7-gons among 33 points in unit square."""
    t0 = time.time()
    deadline = t0 + 540  # safety margin under 600s

    # Build concave arc: 28 points
    pts_arc = _make_concave_arc(28)
    # 5 extra points placed below the arc (inside convex hull)
    # Place them at low y, spread across x, so they lie inside the polygon
    # formed by the arc plus the square boundary
    extra = np.array([
        [0.15, 0.10],
        [0.30, 0.12],
        [0.45, 0.14],
        [0.60, 0.12],
        [0.75, 0.10],
    ])
    pts = np.vstack([pts_arc, extra])

    # Small deterministic jitter to break any accidental collinearity
    rng = np.random.default_rng(12345)
    pts = pts + rng.normal(0, 0.0005, size=(33, 2))
    pts = np.clip(pts, 0.001, 0.999)

    if not _is_valid(pts):
        # Retry with different jitter
        for attempt in range(50):
            pts_try = pts + rng.normal(0, 0.0003, size=(33, 2))
            pts_try = np.clip(pts_try, 0.001, 0.999)
            if _is_valid(pts_try):
                pts = pts_try
                break

    # If still invalid, fall back to a pure random valid set
    if not _is_valid(pts):
        for attempt in range(1000):
            pts_try = rng.random((33, 2)) * 0.98 + 0.01
            if _is_valid(pts_try):
                pts = pts_try
                break

    elapsed = time.time() - t0
    # We have ample time; the construction is deterministic and fast.
    return pts.astype(float)


# EVOLVE-BLOCK-END
