import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    TIME_LIMIT = 600.0
    SAFETY_MARGIN = 25.0
    N = 16

    def get_cost(pts):
        """Calculate (d_max / d_min)^2. Lower is better."""
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_min <= 1e-12 or d_max <= 1e-12:
            return np.inf
        return (d_max / d_min) ** 2

    def get_cost_vec(params):
        pts = params.reshape(16, 2)
        return get_cost(pts)

    def hex_lattice_config(scale=1.0, rotation=0.0):
        """
        Generate 16 points from a truncated hexagonal lattice.
        Uses hexagonal coordinates to get a natural 16-point subset.
        """
        # Hexagonal lattice basis
        a = scale
        # Generate hex lattice points
        hex_points = []
        # Use a range of hex coordinates
        for q in range(-3, 4):
            for r in range(-3, 4):
                x = a * (q + r * 0.5)
                y = a * (r * np.sqrt(3) / 2)
                hex_points.append((x, y))
        hex_points = np.array(hex_points)
        
        # Select 16 points closest to origin (centered hex cluster)
        dists_to_origin = np.sum(hex_points ** 2, axis=1)
        idx = np.argsort(dists_to_origin)[:16]
        pts = hex_points[idx]
        
        # Apply rotation
        if rotation != 0:
            c, s = np.cos(rotation), np.sin(rotation)
            R = np.array([[c, -s], [s, c]])
            pts = pts @ R.T
        
        # Center
        pts = pts - np.mean(pts, axis=0)
        return pts

    def ring_config(n1, n2, r1, r2, phase1, phase2):
        """
        General two-ring configuration with n1 points on inner ring, n2 on outer ring.
        """
        angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False) + phase1
        angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + phase2
        pts1 = r1 * np.column_stack([np.cos(angles1), np.sin(angles1)])
        pts2 = r2 * np.column_stack([np.cos(angles2), np.sin(angles2)])
        pts = np.vstack([pts1, pts2])
        pts = pts - np.mean(pts, axis=0)
        return pts

    best_pts = None
    best_cost = np.inf

    def update_best(pts):
        nonlocal best_pts, best_cost
        c = get_cost(pts)
        if c < best_cost and np.isfinite(c):
            best_cost = c
            best_pts = pts.copy()
        return c

    # ---- Phase 1: Generate diverse initial configurations ----
    # 1a. Hexagonal lattice with various scales and rotations
    for scale in [0.8, 1.0, 1.2]:
        for rot in [0.0, np.pi/6, np.pi/3, np.pi/12]:
            pts = hex_lattice_config(scale, rot)
            update_best(pts)

    # 1b. Two-ring configurations with various parameters
    for n1, n2 in [(4, 12), (6, 10), (8, 8), (5, 11), (7, 9)]:
        for r2 in [1.5, 2.0, 2.5, 3.0]:
            for ph2 in [0.0, np.pi/n2, 2*np.pi/n2]:
                pts = ring_config(n1, n2, 1.0, r2, 0.0, ph2)
                update_best(pts)

    # 1c. Square grid variants
    for s in [0.8, 1.0, 1.2]:
        g = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float) * s
        g = g - np.mean(g, axis=0)
        update_best(g)
    
    # Rotated square grids
    for s in [0.9, 1.0, 1.1]:
        for rot in [0.0, np.pi/8, np.pi/4]:
            g = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float) * s
            g = g - np.mean(g, axis=0)
            c, s_val = np.cos(rot), np.sin(rot)
            R = np.array([[c, -s_val], [s_val, c]])
            g = g @ R.T
            update_best(g)

    # 1d. Fibonacci spiral
    for n_pts in [16]:
        golden = (1 + np.sqrt(5)) / 2
        angles = 2 * np.pi * np.arange(n_pts) / (golden ** 2)
        radii = np.sqrt(np.arange(n_pts) + 0.5)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        pts = pts - np.mean(pts, axis=0)
        update_best(pts)

    # 1e. Regular polygon perturbations
    for n_poly in [8, 12, 16]:
        angles = 2 * np.pi * np.arange(n_poly) / n_poly
        for radius in [1.0, 1.5, 2.0]:
            pts = radius * np.column_stack([np.cos(angles), np.sin(angles)])
            if n_poly < 16:
                # Add extra points at center or between
                extra = np.zeros((16 - n_poly, 2))
                pts = np.vstack([pts, extra])
            pts = pts - np.mean(pts, axis=0)
            update_best(pts)

    # ---- Phase 2: Powell optimization on full 32 parameters ----
    time_elapsed = time.time() - start_time
    if best_pts is not None and (TIME_LIMIT - time_elapsed) > 60:
        try:
            res = minimize(get_cost_vec, best_pts.flatten(), method='Powell',
                           options={'maxiter': 500, 'ftol': 1e-12, 'xtol': 1e-10, 'maxfev': 5000})
            if np.isfinite(res.fun):
                res_pts = res.x.reshape(16, 2)
                c = get_cost(res_pts)
                if c < best_cost:
                    best_cost = c
                    best_pts = res_pts.copy()
        except Exception:
            pass

    # ---- Phase 3: Nelder-Mead refinement ----
    time_elapsed = time.time() - start_time
    if best_pts is not None and (TIME_LIMIT - time_elapsed) > 40:
        try:
            res2 = minimize(get_cost_vec, best_pts.flatten(), method='Nelder-Mead',
                            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if np.isfinite(res2.fun):
                res_pts2 = res2.x.reshape(16, 2)
                c2 = get_cost(res_pts2)
                if c2 < best_cost:
                    best_cost = c2
                    best_pts = res_pts2.copy()
        except Exception:
            pass

    # ---- Phase 4: Another Powell pass with tighter tolerance ----
    time_elapsed = time.time() - start_time
    if best_pts is not None and (TIME_LIMIT - time_elapsed) > 30:
        try:
            res3 = minimize(get_cost_vec, best_pts.flatten(), method='Powell',
                           options={'maxiter': 300, 'ftol': 1e-14, 'xtol': 1e-12, 'maxfev': 3000})
            if np.isfinite(res3.fun):
                res_pts3 = res3.x.reshape(16, 2)
                c3 = get_cost(res_pts3)
                if c3 < best_cost:
                    best_cost = c3
                    best_pts = res_pts3.copy()
        except Exception:
            pass

    # ---- Phase 5: Final Nelder-Mead polish ----
    time_elapsed = time.time() - start_time
    if best_pts is not None and (TIME_LIMIT - time_elapsed) > 20:
        try:
            res4 = minimize(get_cost_vec, best_pts.flatten(), method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if np.isfinite(res4.fun):
                res_pts4 = res4.x.reshape(16, 2)
                c4 = get_cost(res_pts4)
                if c4 < best_cost:
                    best_cost = c4
                    best_pts = res_pts4.copy()
        except Exception:
            pass

    final_pts = best_pts
    if final_pts is None or not np.all(np.isfinite(final_pts)):
        # Fallback: plain 4x4 grid
        g = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        g = g - np.mean(g, axis=0)
        final_pts = g

    # Final safety check
    if not np.all(np.isfinite(final_pts)):
        g = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        g = g - np.mean(g, axis=0)
        final_pts = g

    return final_pts
