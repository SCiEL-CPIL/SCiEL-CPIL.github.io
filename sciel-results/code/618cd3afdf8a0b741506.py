import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s limit

    # Precompute triangle indices for vectorization
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)

    def compute_score(points):
        """
        Compute min triangle area / convex hull area.
        Returns 0.0 if invalid.
        """
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
            if area_hull < 1e-9:
                return 0.0
            
            x = pts[:, 0]
            y = pts[:, 1]
            v1x = x[tri_j] - x[tri_i]
            v1y = y[tri_j] - y[tri_i]
            v2x = x[tri_k] - x[tri_i]
            v2y = y[tri_k] - y[tri_i]
            cross = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * cross
            min_area = np.min(areas)
            
            if min_area < 0:
                return 0.0
            return min_area / area_hull
        except Exception:
            return 0.0

    def _regular_polygon(n, radius=1.0):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = radius * np.cos(angles)
        ys = radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def points_from_radii(radii):
        """
        Generate points from 13 radii with fixed angles 2*pi*k/13.
        """
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
        return np.column_stack([xs, ys])

    def objective_symmetric(radii):
        """
        Objective function for symmetric optimization.
        Returns negative score to be minimized.
        """
        pts = points_from_radii(radii)
        score = compute_score(pts)
        return -score

    def objective_full(points_flat):
        """
        Objective function for full 26D optimization.
        """
        pts = points_flat.reshape(-1, 2)
        score = compute_score(pts)
        return -score

    best_score = -float('inf')
    best_points = None

    def try_update(pts):
        nonlocal best_score, best_points
        score = compute_score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # Initialize with regular polygon
    init_pts = _regular_polygon(N, 1.0)
    try_update(init_pts)

    # Phase 1: Optimize in the 13-dimensional symmetric subspace
    # This is much faster than full 26D and captures the main structure
    initializations = [
        np.ones(N),
        np.ones(N) + 0.1 * np.sin(2 * np.pi * np.arange(N) / N),
        np.ones(N) + 0.1 * np.sin(4 * np.pi * np.arange(N) / N),
        np.ones(N) + 0.05 * np.cos(3 * np.pi * np.arange(N) / N),
    ]
    
    for init_rad in initializations:
        if time.time() > start_time + time_limit - 30:
            break
            
        try:
            res = opt.minimize(
                objective_symmetric,
                init_rad,
                method='Powell',
                options={'maxiter': 300, 'xtol': 1e-8, 'ftol': 1e-10}
            )
            candidate_pts = points_from_radii(res.x)
            try_update(candidate_pts)
        except Exception:
            continue

    # Phase 2: Simulated Annealing in full Cartesian space
    # This allows breaking radial symmetry and escaping local minima
    if best_points is not None:
        if time.time() > start_time + time_limit - 20:
            return best_points
        
        current_pts = best_points.copy()
        current_score = best_score
        T = 0.01
        T_min = 1e-6
        cooling = 0.995
        step_size = 0.05
        rng = np.random.RandomState(42)
        
        sa_start = time.time()
        sa_limit = 15.0  # Max 15 seconds for SA
        
        while time.time() - sa_start < sa_limit:
            # Generate neighbor
            noise = rng.normal(0, step_size, size=(N, 2))
            neighbor = current_pts + noise
            
            neighbor_score = compute_score(neighbor)
            
            # Accept if better or with probability exp(-delta/T)
            if neighbor_score > current_score:
                current_pts = neighbor
                current_score = neighbor_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_pts.copy()
            else:
                delta = current_score - neighbor_score
                if delta < 0:
                    delta = 0
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_pts = neighbor
                    current_score = neighbor_score
            
            T *= cooling
            if T < T_min:
                break

    # Phase 3: Polish in full 26-dimensional space
    if best_points is not None:
        if time.time() > start_time + time_limit - 10:
            return best_points
            
        try:
            res = opt.minimize(
                objective_full,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            candidate_pts = res.x.reshape(-1, 2)
            try_update(candidate_pts)
        except Exception:
            pass

    # Phase 4: Random restarts in Cartesian space with local optimization
    if time.time() < start_time + time_limit - 5:
        rng = np.random.RandomState(123)
        for _ in range(3):
            if time.time() > start_time + time_limit - 2:
                break
            
            # Generate random points in a convex position
            theta = rng.uniform(0, 2 * np.pi, N)
            theta.sort()
            r = rng.uniform(0.8, 1.2, N)
            pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            
            try:
                res = opt.minimize(
                    objective_full,
                    pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                candidate_pts = res.x.reshape(-1, 2)
                try_update(candidate_pts)
            except Exception:
                continue

    if best_points is None:
        best_points = init_pts

    return best_points
