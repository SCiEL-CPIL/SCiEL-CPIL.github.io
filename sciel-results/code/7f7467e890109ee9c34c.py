import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def _all_triangle_areas(points):
    """Vectorized computation of all C(n,3) triangle areas."""
    n = len(points)
    if n < 3:
        return np.array([0.0])
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx, j_idx, k_idx = i_idx[mask], j_idx[mask], k_idx[mask]
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _hull_area(points)
    if hull_area <= 1e-10:
        return 0.0
    return min_area / hull_area

def _soft_min_objective(params, beta=100.0):
    """Negative smooth min area / hull area."""
    points = params.reshape(-1, 2)
    areas = _all_triangle_areas(points)
    
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    lse = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min = -lse / beta
    
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 1e10
    
    return -soft_min / hull_area

def _regular_polygon(n, r=1.0, phase=0.0):
    angles = phase + np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((r * np.cos(angles), r * np.sin(angles)))

def _two_shell_points(r1, r2, p1, p2, n1):
    n1 = int(round(n1))
    n1 = max(1, min(N - 1, n1))
    n2 = N - n1
    pts1 = _regular_polygon(n1, r1, p1)
    pts2 = _regular_polygon(n2, r2, p2)
    return np.vstack([pts1, pts2])

def _three_shell_points(r1, r2, r3, p1, p2, p3, n1, n2):
    n1 = int(round(n1))
    n2 = int(round(n2))
    n1 = max(1, min(N - 2, n1))
    n2 = max(1, min(N - n1 - 1, n2))
    n3 = N - n1 - n2
    if n3 < 1:
        n3 = 1
        n2 = N - n1 - n3
    pts1 = _regular_polygon(n1, r1, p1)
    pts2 = _regular_polygon(n2, r2, p2)
    pts3 = _regular_polygon(n3, r3, p3)
    return np.vstack([pts1, pts2, pts3])

def _normalize_points(points):
    """Center and scale so that the convex hull area is 1."""
    points = points - np.mean(points, axis=0)
    hull_area = _hull_area(points)
    if hull_area > 1e-10:
        points = points / np.sqrt(hull_area)
    return points

def construct():
    n = N
    t0 = time.time()
    deadline = 130.0  # Safety margin before 150s kill

    best_points = None
    best_score = -1e10

    # --- Phase 1: Generate diverse starting configurations ---
    starts = []

    # Regular polygon
    starts.append(_regular_polygon(N))

    # Two-shell configurations with various splits and parameters
    for n1 in range(2, N):
        n2 = N - n1
        for r1 in [0.4, 0.6, 0.8]:
            for r2 in [0.9, 1.0, 1.1]:
                for p1 in [0.0, np.pi / n1]:
                    for p2 in [0.0, np.pi / n2]:
                        pts = _two_shell_points(r1, r2, p1, p2, n1)
                        starts.append(pts)

    # Three-shell configurations
    for n1 in [1, 2, 3, 4]:
        for n2 in [1, 2, 3]:
            n3 = N - n1 - n2
            if n3 < 1:
                continue
            for r1 in [0.3, 0.5]:
                for r2 in [0.7, 0.9]:
                    for r3 in [1.0, 1.1]:
                        pts = _three_shell_points(r1, r2, r3, 0.0, 0.0, 0.0, n1, n2)
                        starts.append(pts)

    # Random configurations
    rng = np.random.default_rng(42)
    for _ in range(30):
        pts = rng.uniform(-1, 1, size=(N, 2))
        # Ensure some spread
        if np.ptp(pts, axis=0).min() < 0.5:
            continue
        starts.append(pts)

    # --- Phase 2: Refine each start using local optimization ---
    for i, start_pts in enumerate(starts):
        if time.time() - t0 > deadline:
            break

        # Normalize the start
        pts = _normalize_points(start_pts.copy())
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

        # Try L-BFGS-B with soft-min objective
        if time.time() - t0 > deadline:
            break
        try:
            res = minimize(_soft_min_objective, pts.flatten(), args=(50.0,),
                           method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            cand = res.x.reshape(-1, 2)
            cand = _normalize_points(cand)
            cs = _score(cand)
            if cs > best_score:
                best_score = cs
                best_points = cand.copy()
        except Exception:
            pass

        # Try Nelder-Mead for final polish
        if time.time() - t0 > deadline:
            break
        try:
            res = minimize(_soft_min_objective, best_points.flatten(), args=(100.0,),
                           method='Nelder-Mead', options={'maxiter': 500, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-10})
            cand = res.x.reshape(-1, 2)
            cand = _normalize_points(cand)
            cs = _score(cand)
            if cs > best_score:
                best_score = cs
                best_points = cand.copy()
        except Exception:
            pass

    # --- Phase 3: Discrete lattice search (simulated annealing) ---
    if time.time() - t0 < deadline:
        # Use a rational grid within a bounded square
        # Try different grid resolutions
        for grid_res in [10, 20, 50]:
            if time.time() - t0 > deadline:
                break
            # Grid points in [-1, 1]
            grid_vals = np.linspace(-1, 1, grid_res)
            grid_pts = np.array([[x, y] for x in grid_vals for y in grid_vals])

            # Start from best continuous solution, snap to grid
            snapped = np.zeros_like(best_points)
            for i in range(N):
                dists = np.linalg.norm(grid_pts - best_points[i], axis=1)
                snapped[i] = grid_pts[np.argmin(dists)]

            # Simulated annealing: swap points on the grid
            current_pts = snapped.copy()
            current_score = _score(current_pts)
            if current_score > best_score:
                best_score = current_score
                best_points = current_pts.copy()

            T = 1.0
            T_min = 0.001
            alpha = 0.995
            steps = 5000

            for step in range(steps):
                if time.time() - t0 > deadline:
                    break
                if T < T_min:
                    break

                # Pick two random points to swap positions on the grid
                i, j = np.random.choice(N, 2, replace=False)
                new_pts = current_pts.copy()
                # Move point i to a random grid point
                new_i = grid_pts[np.random.randint(len(grid_pts))]
                new_pts[i] = new_i

                new_score = _score(new_pts)
                delta = new_score - current_score

                if delta > 0 or np.random.random() < np.exp(delta / T):
                    current_pts = new_pts
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = current_pts.copy()

                T *= alpha

            # Refine the best lattice solution with continuous optimization
            if time.time() - t0 > deadline:
                break
            try:
                res = minimize(_soft_min_objective, best_points.flatten(), args=(100.0,),
                               method='Nelder-Mead', options={'maxiter': 500, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-10})
                cand = res.x.reshape(-1, 2)
                cand = _normalize_points(cand)
                cs = _score(cand)
                if cs > best_score:
                    best_score = cs
                    best_points = cand.copy()
            except Exception:
                pass

    # --- Phase 4: Final high-precision polish ---
    if time.time() - t0 < deadline and best_points is not None:
        try:
            res = minimize(_soft_min_objective, best_points.flatten(), args=(200.0,),
                           method='Nelder-Mead', options={'maxiter': 1000, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-12})
            cand = res.x.reshape(-1, 2)
            cand = _normalize_points(cand)
            cs = _score(cand)
            if cs > best_score:
                best_score = cs
                best_points = cand.copy()
        except Exception:
            pass

    # Ensure we return a valid point set
    if best_points is None:
        best_points = _regular_polygon(N)

    return best_points
