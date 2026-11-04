"""Heilbronn convex regions n=14 — radial ring optimization with angular packing."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 14

def _min_triangle_area(pts):
    """Minimum triangle area over all triples of points."""
    n = len(pts)
    if n < 3:
        return 0.0
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                a = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if a < min_a:
                    min_a = a
    return min_a

def _convex_hull_area(pts):
    """Convex hull area via cross product on sorted angles around centroid."""
    n = len(pts)
    if n < 3:
        return 0.0
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    order = np.argsort(angles)
    hpts = pts[order]
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += hpts[i][0] * hpts[j][1] - hpts[j][0] * hpts[i][1]
    return abs(area) * 0.5

def _score(pts):
    h = _convex_hull_area(pts)
    if h < 1e-12:
        return 0.0
    return _min_triangle_area(pts) / h

def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin before 150s kill

    # --- Strategy 1: All 14 points on a single circle (radial ring) ---
    # Angular packing: minimize minimum triangle area normalized by hull area
    # For points on a unit circle, hull area is fixed (the polygon area).
    # Triangle area for three points at angles a<b<c on unit circle:
    #   0.5 * |sin(b-a) + sin(c-b) - sin(c-a)|
    # We want to maximize the minimum over all triples.
    # This is a 1D optimization over 14 angles.

    def neg_score_unit_circle(angles):
        # Sort angles
        a = np.sort(angles)
        # Normalize to [0, 2*pi)
        a = a - a[0]
        # Points on unit circle
        pts = np.column_stack([np.cos(a), np.sin(a)])
        return -_score(pts)

    def neg_score_radial(R, angles):
        a = np.sort(angles)
        a = a - a[0]
        pts = R * np.column_stack([np.cos(a), np.sin(a)])
        return -_score(pts)

    # Initial guess: evenly spaced angles
    best_score = 0.0
    best_pts = None

    # Try evenly spaced first
    angles_even = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts_even = np.column_stack([np.cos(angles_even), np.sin(angles_even)])
    s = _score(pts_even)
    best_score = s
    best_pts = pts_even.copy()

    # Differential evolution over angles (14-dim)
    bounds = [(0, 2 * np.pi)] * N
    x0 = angles_even.tolist()
    try:
        res = differential_evolution(
            neg_score_unit_circle,
            bounds,
            x0=x0,
            maxiter=80,
            popsize=15,
            tol=1e-6,
            seed=42,
            init='sobol',
            polish=False,
            timeout=60
        )
        if time.time() > deadline:
            return best_pts
        s = -res.fun
        if s > best_score:
            best_score = s
            a = np.sort(res.x)
            best_pts = np.column_stack([np.cos(a), np.sin(a)])
    except Exception:
        pass

    # Local refinement with L-BFGS-B
    if time.time() < deadline - 30:
        try:
            a0 = np.sort(res.x) if 'res' in dir() else angles_even.copy()
            res2 = minimize(
                neg_score_unit_circle,
                a0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            s = -res2.fun
            if s > best_score:
                best_score = s
                a = np.sort(res2.x)
                best_pts = np.column_stack([np.cos(a), np.sin(a)])
        except Exception:
            pass

    # --- Strategy 2: Two concentric rings ---
    if time.time() < deadline - 60:
        # 7 points on inner ring, 7 on outer ring, alternating
        def neg_score_two_ring(params):
            R_in = params[0]
            R_out = params[1]
            angles_in = np.sort(params[2:2+N//2])
            angles_out = np.sort(params[2+N//2:])
            pts_in = R_in * np.column_stack([np.cos(angles_in), np.sin(angles_in)])
            pts_out = R_out * np.column_stack([np.cos(angles_out), np.sin(angles_out)])
            pts = np.vstack([pts_in, pts_out])
            return -_score(pts)

        # Initial: inner radius 0.5, outer 1.0, evenly spaced with offset
        n_in = N // 2
        n_out = N - n_in
        a_in0 = np.linspace(0, 2*np.pi, n_in, endpoint=False)
        a_out0 = np.linspace(np.pi/n_out, 2*np.pi + np.pi/n_out, n_out, endpoint=False)
        x0_two = np.concatenate([[0.5, 1.0], a_in0, a_out0]).tolist()
        bounds_two = [(0.1, 1.5)] + [(0.1, 1.5)] + [(0, 2*np.pi)] * N
        try:
            res3 = differential_evolution(
                neg_score_two_ring,
                bounds_two,
                x0=x0_two,
                maxiter=40,
                popsize=12,
                tol=1e-5,
                seed=123,
                polish=False,
                timeout=50
            )
            if time.time() > deadline:
                return best_pts
            s = -res3.fun
            if s > best_score:
                best_score = s
                R_in = res3.x[0]
                R_out = res3.x[1]
                a_in = np.sort(res3.x[2:2+n_in])
                a_out = np.sort(res3.x[2+n_in:])
                pts_in = R_in * np.column_stack([np.cos(a_in), np.sin(a_in)])
                pts_out = R_out * np.column_stack([np.cos(a_out), np.sin(a_out)])
                best_pts = np.vstack([pts_in, pts_out])
        except Exception:
            pass

    # --- Strategy 3: Free 2D optimization (small n, few iterations) ---
    if time.time() < deadline - 20:
        # Use a regular polygon as starting point and do local search
        def neg_score_free(pts_flat):
            pts = pts_flat.reshape(-1, 2)
            return -_score(pts)

        # Start from best found so far
        x0_free = best_pts.flatten().tolist()
        bounds_free = [(-2, 2)] * (2 * N)
        try:
            res4 = minimize(
                neg_score_free,
                x0_free,
                method='L-BFGS-B',
                bounds=bounds_free,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            s = -res4.fun
            if s > best_score:
                best_score = s
                best_pts = res4.x.reshape(-1, 2)
        except Exception:
            pass

    # Final: verify and return
    if time.time() > deadline:
        return best_pts

    final_score = _score(best_pts)
    if final_score <= 0:
        # Fallback: regular polygon
        a = np.linspace(0, 2*np.pi, N, endpoint=False)
        return np.column_stack([np.cos(a), np.sin(a)])

    return best_pts
