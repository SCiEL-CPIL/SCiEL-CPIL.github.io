import numpy as np
import scipy.optimize
import scipy.spatial
import time

N = 14

def construct():
    start_time = time.time()
    time_limit = 130.0  # Leave 20s buffer for safety

    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts0 = np.vstack([np.cos(angles), np.sin(angles)]).T

    def get_score(pts):
        pts = np.array(pts, dtype=float)
        # Center at origin
        pts -= np.mean(pts, axis=0)
        # Scale to unit RMS radius
        norm = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if norm < 1e-9:
            return 0.0
        pts /= norm

        # Calculate minimum triangle area
        min_area = np.inf
        # Vectorized calculation for all pairs i, j
        # Then for each k > j
        # But N=14 is small, simple loops are fast enough and easier to debug
        for i in range(N):
            for j in range(i + 1, N):
                v1 = pts[j] - pts[i]
                for k in range(j + 1, N):
                    v2 = pts[k] - pts[i]
                    # Cross product magnitude in 2D
                    area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                    if area < min_area:
                        min_area = area

        # Calculate convex hull area
        try:
            hull = scipy.spatial.ConvexHull(pts)
            hull_area = hull.volume
        except Exception:
            return 0.0

        if hull_area < 1e-9:
            return 0.0

        return min_area / hull_area

    def objective(pts):
        return -get_score(pts)

    best_pts = pts0.copy()
    best_score = get_score(best_pts)

    # Optimization using Powell's method
    # Powell is derivative-free and handles non-smooth objectives (min of areas) well
    def callback(xk):
        nonlocal best_pts, best_score
        current_score = get_score(xk)
        if current_score > best_score:
            best_score = current_score
            best_pts = xk.copy()
        if time.time() - start_time > time_limit:
            raise RuntimeError("Time limit exceeded")

    try:
        # maxiter set high, but callback will stop it if time runs out
        res = scipy.optimize.minimize(
            objective, 
            pts0, 
            method='Powell', 
            options={'maxiter': 1000, 'disp': False, 'ftol': 1e-12, 'xtol': 1e-12},
            callback=callback
        )
        # Check result
        final_score = get_score(res.x)
        if final_score > best_score:
            best_score = final_score
            best_pts = res.x.copy()
    except RuntimeError:
        pass # Time limit hit, use best_pts found so far

    return best_pts
