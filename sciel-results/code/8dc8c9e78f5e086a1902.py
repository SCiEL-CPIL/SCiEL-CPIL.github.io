import numpy as np
import time
import random
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
    SAFETY_MARGIN = 30.0
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
        """Generate 16 points from a truncated hexagonal lattice."""
        a = scale
        hex_points = []
        for q in range(-3, 4):
            for r in range(-3, 4):
                x = a * (q + r * 0.5)
                y = a * (r * np.sqrt(3) / 2)
                hex_points.append((x, y))
        hex_points = np.array(hex_points)
        dists_to_origin = np.sum(hex_points ** 2, axis=1)
        idx = np.argsort(dists_to_origin)[:16]
        pts = hex_points[idx]
        if rotation != 0:
            c, s = np.cos(rotation), np.sin(rotation)
            R = np.array([[c, -s], [s, c]])
            pts = pts @ R.T
        pts = pts - np.mean(pts, axis=0)
        return pts

    def ring_config(n1, n2, r1, r2, phase1, phase2):
        """General two-ring configuration."""
        angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False) + phase1
        angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + phase2
        pts1 = r1 * np.column_stack([np.cos(angles1), np.sin(angles1)])
        pts2 = r2 * np.column_stack([np.cos(angles2), np.sin(angles2)])
        pts = np.vstack([pts1, pts2])
        pts = pts - np.mean(pts, axis=0)
        return pts

    def random_config():
        """Generate a random configuration centered at origin."""
        pts = np.random.randn(16, 2)
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
    # 1a. Hexagonal lattice
    for scale in [0.8, 1.0, 1.2]:
        for rot in [0.0, np.pi/6, np.pi/3, np.pi/12]:
            pts = hex_lattice_config(scale, rot)
            update_best(pts)

    # 1b. Two-ring configurations
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

    # 1d. Fibonacci spiral
    golden = (1 + np.sqrt(5)) / 2
    angles = 2 * np.pi * np.arange(16) / (golden ** 2)
    radii = np.sqrt(np.arange(16) + 0.5)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    pts = pts - np.mean(pts, axis=0)
    update_best(pts)

    # 1e. Random initial points
    for _ in range(20):
        pts = random_config()
        update_best(pts)

    # ---- Phase 2: Simulated Annealing for global exploration ----
    time_elapsed = time.time() - start_time
    sa_time_budget = min(180.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN)
    if sa_time_budget > 30 and best_pts is not None:
        sa_start = time.time()
        T_init = max(0.1, (best_cost - 1.0) * 0.01) if best_cost > 1.0 else 0.1
        T = T_init
        T_min = 1e-8
        cooling = 0.9995
        current_pts = best_pts.copy()
        current_cost = best_cost

        while time.time() - sa_start < sa_time_budget:
            # Generate perturbation
            step_size = 0.1 * max(0.1, np.sqrt(T))
            perturbation = np.random.randn(16, 2) * step_size
            new_pts = current_pts + perturbation
            new_pts = new_pts - np.mean(new_pts, axis=0)  # Re-center
            new_cost = get_cost(new_pts)

            if not np.isfinite(new_cost):
                continue

            # Acceptance criterion
            if new_cost < current_cost:
                current_pts = new_pts
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_pts = current_pts.copy()
            else:
                # Metropolis acceptance
                delta = new_cost - current_cost
                if np.random.random() < np.exp(-delta / max(T, 1e-10)):
                    current_pts = new_pts
                    current_cost = new_cost

            # Cool down
            T = max(T_min, T * cooling)

            # Check time
            if time.time() - sa_start >= sa_time_budget:
                break

    # ---- Phase 3: Powell optimization ----
    time_elapsed = time.time() - start_time
    if best_pts is not None and (TIME_LIMIT - time_elapsed) > 60:
        try:
            res = minimize(get_cost_vec, best_pts.flatten(), method='Powell',
                           options={'maxiter': 1000, 'ftol': 1e-12, 'xtol': 1e-10, 'maxfev': 10000})
            if np.isfinite(res.fun):
                res_pts = res.x.reshape(16, 2)
                c = get_cost(res_pts)
                if c < best_cost:
                    best_cost = c
                    best_pts = res_pts.copy()
        except Exception:
            pass

    # ---- Phase 4: Second SA round from improved starting point ----
    time_elapsed = time.time() - start_time
    sa2_budget = min(120.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN)
    if sa2_budget > 30 and best_pts is not None:
        sa2_start = time.time()
        T = max(0.05, (best_cost - 1.0) * 0.005) if best_cost > 1.0 else 0.05
        T_min = 1e-9
        cooling = 0.9997
        current_pts = best_pts.copy()
        current_cost = best_cost

        while time.time() - sa2_start < sa2_budget:
            step_size = 0.05 * max(0.1, np.sqrt(T))
            perturbation = np.random.randn(16, 2) * step_size
            new_pts = current_pts + perturbation
            new_pts = new_pts - np.mean(new_pts, axis=0)
            new_cost = get_cost(new_pts)

            if not np.isfinite(new_cost):
                continue

            if new_cost < current_cost:
                current_pts = new_pts
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_pts = current_pts.copy()
            else:
                delta = new_cost - current_cost
                if np.random.random() < np.exp(-delta / max(T, 1e-10)):
                    current_pts = new_pts
                    current_cost = new_cost

            T = max(T_min, T * cooling)

            if time.time() - sa2_start >= sa2_budget:
                break

    # ---- Phase 5: Final Powell refinement ----
    time_elapsed = time.time() - start_time
    if best_pts is not None and (TIME_LIMIT - time_elapsed) > 40:
        try:
            res2 = minimize(get_cost_vec, best_pts.flatten(), method='Powell',
                           options={'maxiter': 500, 'ftol': 1e-14, 'xtol': 1e-12, 'maxfev': 5000})
            if np.isfinite(res2.fun):
                res_pts2 = res2.x.reshape(16, 2)
                c2 = get_cost(res_pts2)
                if c2 < best_cost:
                    best_cost = c2
                    best_pts = res_pts2.copy()
        except Exception:
            pass

    # ---- Phase 6: Nelder-Mead polish ----
    time_elapsed = time.time() - start_time
    if best_pts is not None and (TIME_LIMIT - time_elapsed) > 20:
        try:
            res3 = minimize(get_cost_vec, best_pts.flatten(), method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if np.isfinite(res3.fun):
                res_pts3 = res3.x.reshape(16, 2)
                c3 = get_cost(res_pts3)
                if c3 < best_cost:
                    best_cost = c3
                    best_pts = res_pts3.copy()
        except Exception:
            pass

    final_pts = best_pts
    if final_pts is None or not np.all(np.isfinite(final_pts)):
        g = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        g = g - np.mean(g, axis=0)
        final_pts = g

    if not np.all(np.isfinite(final_pts)):
        g = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        g = g - np.mean(g, axis=0)
        final_pts = g

    return final_pts
