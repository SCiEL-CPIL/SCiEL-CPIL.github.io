"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _circumcenter_robust(a, b, c):
    """Robust circumcenter of triangle on sphere using multiple edge cross products."""
    # Try different edge pairs to handle near-degenerate cases
    for e in ((a, b, c), (b, a, c), (a, c, b)):
        cand = np.cross(e[0], e[1])
        cn = np.linalg.norm(cand)
        if np.dot(cand, e[2]) > 0 and cn > 1e-12:
            return cand / cn
    # Fallback: use centroid direction
    ctr = a + b + c
    cn = np.linalg.norm(ctr)
    if cn < 1e-9:
        return None
    return ctr / cn


def _covering_radius_deg(pts):
    """Compute covering radius in degrees via convex hull facets."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    facets = hull.simplices
    if len(facets) == 0:
        return 180.0

    max_r = 0.0
    for f in facets:
        a, b, c = pts[f[0]], pts[f[1]], pts[f[2]]
        ctr = _circumcenter_robust(a, b, c)
        if ctr is None:
            continue
        # Ensure center points into the triangle interior
        if np.dot(ctr, (a + b + c)) < 0:
            ctr = -ctr
        r = max(
            np.arccos(np.clip(np.dot(ctr, a), -1, 1)),
            np.arccos(np.clip(np.dot(ctr, b), -1, 1)),
            np.arccos(np.clip(np.dot(ctr, c), -1, 1)),
        )
        if r > max_r:
            max_r = r

    return np.degrees(max_r)


def _fibonacci_spiral(n, jitter=0.0, rng=None):
    """Generate Fibonacci spiral points on sphere with optional jitter."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if jitter > 0 and rng is not None:
        noise = rng.normal(0, jitter, size=pts.shape)
        pts = pts + noise
        pts = _normalize(pts)
    return pts


def _optimize(pts, deadline):
    """Steepest descent on the hottest facet with momentum."""
    best_pts = pts.copy()
    best_score = _covering_radius_deg(pts)
    step = 0.004
    mom = np.zeros_like(pts)
    no_improve = 0
    max_no_improve = 300

    while time.time() < deadline:
        try:
            hull = ConvexHull(pts)
        except Exception:
            pts = best_pts.copy()
            break

        facets = hull.simplices
        if len(facets) == 0:
            break

        # Find the hottest facet
        worst_r = -1.0
        hot_f = None
        hot_ctr = None
        for f in facets:
            a, b, c = pts[f[0]], pts[f[1]], pts[f[2]]
            ctr = _circumcenter_robust(a, b, c)
            if ctr is None:
                continue
            if np.dot(ctr, (a + b + c)) < 0:
                ctr = -ctr
            r = max(
                np.arccos(np.clip(np.dot(ctr, a), -1, 1)),
                np.arccos(np.clip(np.dot(ctr, b), -1, 1)),
                np.arccos(np.clip(np.dot(ctr, c), -1, 1)),
            )
            if r > worst_r:
                worst_r = r
                hot_f = f
                hot_ctr = ctr

        if hot_f is None:
            break

        # Build gradient: push hot facet vertices away from circumcenter
        g = np.zeros_like(pts)
        for idx in hot_f:
            g[idx] -= hot_ctr

        # Normalize gradient per point
        gm = np.linalg.norm(g, axis=1, keepdims=True)
        gm = np.where(gm > 0, gm, 1.0)
        g = g / gm

        # Momentum update
        mom = 0.5 * mom + 0.5 * g
        new_pts = pts - step * mom
        new_pts = _normalize(new_pts)

        new_score = _covering_radius_deg(new_pts)
        if new_score < best_score - 1e-12:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts
            no_improve = 0
            step = min(step * 1.02, 0.01)
        else:
            no_improve += 1
            step *= 0.85
            if step < 1e-7:
                step = 0.004
                mom *= 0.2
            if no_improve > max_no_improve:
                # Random perturbation to escape local minimum
                rng = np.random.default_rng(hash(tuple(pts.ravel().round(6))) % 2**32)
                noise = rng.normal(0, 0.05, size=pts.shape)
                pts = _normalize(best_pts + noise)
                step = 0.004
                mom = np.zeros_like(pts)
                no_improve = 0

    return best_pts, best_score


def construct():
    t0 = time.time()
    deadline = 140.0  # safety margin under 150s limit

    # Multiple starts with different jitter levels
    jitter_levels = [0.0, 0.01, 0.02, 0.05, 0.08]
    rng_main = np.random.default_rng(42)

    best_pts = None
    best_score = 180.0

    for jitter in jitter_levels:
        if time.time() > deadline - 5:
            break

        pts = _fibonacci_spiral(N, jitter=jitter, rng=rng_main)
        # Give each start a reasonable time budget
        start_budget = min(25.0, max(8.0, deadline - 5 - (time.time() - t0)))
        pts_opt, score = _optimize(pts, t0 + start_budget)

        if score < best_score:
            best_score = score
            best_pts = pts_opt

    # Final polish on the best configuration if time remains
    if best_pts is not None and time.time() < deadline - 10:
        pts_opt, score = _optimize(best_pts, deadline - 2)
        if score < best_score:
            best_score = score
            best_pts = pts_opt

    if best_pts is None:
        # Fallback: Fibonacci spiral
        best_pts = _fibonacci_spiral(N)

    return best_pts
# EVOLVE-BLOCK-END
