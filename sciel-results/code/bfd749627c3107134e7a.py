import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
DEADLINE = 110.0  # Safety margin below 120s limit

def _compute_all_areas(points):
    """Compute all triangle areas for n points (vectorized)."""
    n = len(points)
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i < j) & (j < k)
    i = i[mask]
    j = j[mask]
    k = k[mask]
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    return np.min(_compute_all_areas(points))

def _hull_area(points):
    """Compute the convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # Fallback: if hull fails, return a small value to penalize
        return 1e-10

def _score(points):
    """Score = min triangle area / convex hull area."""
    min_ta = _min_triangle_area(points)
    hull_a = _hull_area(points)
    if hull_a <= 1e-12:
        return 0.0
    return min_ta / hull_a

def _soft_min(areas, tau):
    """Soft minimum with log-sum-exp stability."""
    min_val = np.min(areas)
    shifted = areas - min_val
    # Clip to avoid overflow in exp
    shifted = np.maximum(shifted, -50.0 / tau)
    log_sum = np.log(np.sum(np.exp(-tau * shifted)))
    return min_val - (1.0 / tau) * log_sum

def construct():
    start_time = time.time()
    best_score = -1.0
    best_points = None
    rng = np.random.RandomState(42)

    def time_left():
        return (start_time + DEADLINE) - time.time()

    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        try:
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass

    # Normalization: fix p0=[0,0], p1=[1,0] -> 22 free params
    def params_to_points(params):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(N - 2, 2)
        return pts

    def points_to_params(pts):
        """Normalize: translate p0 to origin, rotate p1 to x-axis, scale |p1-p0|=1."""
        p0 = pts[0]
        p1 = pts[1]
        d = p1 - p0
        dist = np.linalg.norm(d)
        if dist < 1e-10:
            return np.zeros(2 * (N - 2))
        pts_t = pts - p0
        angle = np.arctan2(d[1], d[0])
        c, s = np.cos(-angle), np.sin(-angle)
        R = np.array([[c, -s], [s, c]])
        pts_r = pts_t @ R.T
        pts_s = pts_r / dist
        return pts_s[2:].flatten()

    def smooth_objective(params, tau):
        """-soft_min(areas) / hull_area in normalized 22-param space."""
        try:
            pts = params_to_points(params)
            # Check for degenerate points (all same)
            if np.allclose(pts, pts[0]):
                return 1e6
            areas = _compute_all_areas(pts)
            sm = _soft_min(areas, tau)
            hull_a = _hull_area(pts)
            if hull_a < 1e-10:
                return 1e6
            score = sm / hull_a
            return -score
        except Exception:
            return 1e6

    def hard_negative_score(params):
        """Negative of hard score for Nelder-Mead polish."""
        try:
            pts = params_to_points(params)
            s = _score(pts)
            return -s
        except Exception:
            return 1e6

    # Generate diverse initial configurations
    initial_configs = []

    def make_regular_ngon(n, center=None, radius=1.0, phase=0.0):
        angles = np.linspace(phase, phase + 2 * np.pi, n, endpoint=False)
        if center is None:
            center = np.array([0.0, 0.0])
        return np.column_stack((center[0] + radius * np.cos(angles),
                                center[1] + radius * np.sin(angles)))

    # 1. Regular 13-gon
    initial_configs.append(make_regular_ngon(13, radius=1.0))

    # 2. 12-gon ring + 1 center point
    ring12 = make_regular_ngon(12, radius=1.0)
    initial_configs.append(np.vstack([ring12, np.array([[0.0, 0.0]])]))

    # 3. 11-gon ring + 2 points (center and offset)
    ring11 = make_regular_ngon(11, radius=1.0)
    interior2 = np.array([[0.0, 0.0], [0.3, 0.0]])
    initial_configs.append(np.vstack([ring11, interior2]))

    # 4. 10-gon ring + 3 inner (triangle pattern)
    ring10 = make_regular_ngon(10, radius=1.0)
    inner3 = make_regular_ngon(3, radius=0.45, phase=np.pi/6)
    initial_configs.append(np.vstack([ring10, inner3]))

    # 5. 9-gon ring + 4 inner (square pattern)
    ring9 = make_regular_ngon(9, radius=1.0)
    inner4 = make_regular_ngon(4, radius=0.45, phase=np.pi/4)
    initial_configs.append(np.vstack([ring9, inner4]))

    # 6. 8-gon ring + 5 inner (pentagon pattern)
    ring8 = make_regular_ngon(8, radius=1.0)
    inner5 = make_regular_ngon(5, radius=0.5, phase=np.pi/5)
    initial_configs.append(np.vstack([ring8, inner5]))

    # 7. 7-gon ring + 6 inner (hexagon pattern)
    ring7 = make_regular_ngon(7, radius=1.0)
    inner6 = make_regular_ngon(6, radius=0.5, phase=0.0)
    initial_configs.append(np.vstack([ring7, inner6]))

    # 8. Two concentric rings: 7 outer + 6 inner
    outer7 = make_regular_ngon(7, radius=1.0)
    inner6 = make_regular_ngon(6, radius=0.55, phase=np.pi/6)
    initial_configs.append(np.vstack([outer7, inner6]))

    # 9. Three layers: 6 outer + 5 middle + 2 inner
    outer6 = make_regular_ngon(6, radius=1.0)
    middle5 = make_regular_ngon(5, radius=0.65, phase=np.pi/5)
    inner2 = np.array([[0.0, 0.0], [0.2, 0.0]])
    initial_configs.append(np.vstack([outer6, middle5, inner2]))

    # 10. Perturbed regular 13-gon
    base = make_regular_ngon(13, radius=1.0)
    perturbed = base + rng.normal(0, 0.03, base.shape)
    initial_configs.append(perturbed)

    # 11. Slightly elliptical 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    ellipse = np.column_stack((1.05 * np.cos(angles), 0.95 * np.sin(angles)))
    initial_configs.append(ellipse)

    # 12. 12-gon with center and one vertex slightly inward
    ring12c = make_regular_ngon(12, radius=1.0)
    ring12c[0] = ring12c[0] * 0.92
    initial_configs.append(np.vstack([ring12c, np.array([[0.0, 0.0]])]))

    # 13. Hexagonal lattice subset
    hex_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i + 0.5 * (j % 2)
            y = j * 0.866
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    hex_pts -= np.mean(hex_pts, axis=0)
    if len(hex_pts) >= 13:
        dists = np.linalg.norm(hex_pts, axis=1)
        idx_sorted = np.argsort(dists)
        lattice_init = hex_pts[idx_sorted[:13]]
        initial_configs.append(lattice_init)

    # 14. Spiral pattern (Archimedean spiral)
    t = np.linspace(0.3, 2.0, N)
    theta = 2.5 * t
    spiral = np.column_stack((t * np.cos(theta), t * np.sin(theta)))
    # Normalize to have similar scale
    spiral = spiral / np.max(np.linalg.norm(spiral, axis=1))
    initial_configs.append(spiral)

    # 15. Three-ring structure: 5 outer + 5 middle + 3 inner
    outer5 = make_regular_ngon(5, radius=1.0)
    middle5 = make_regular_ngon(5, radius=0.65, phase=np.pi/5)
    inner3 = make_regular_ngon(3, radius=0.35, phase=0.0)
    initial_configs.append(np.vstack([outer5, middle5, inner3]))

    # 16. Four-ring structure: 4 outer + 4 middle + 3 inner + 2 center
    outer4 = make_regular_ngon(4, radius=1.0)
    middle4 = make_regular_ngon(4, radius=0.7, phase=np.pi/4)
    inner3 = make_regular_ngon(3, radius=0.4, phase=np.pi/6)
    center2 = np.array([[0.0, 0.0], [0.15, 0.0]])
    initial_configs.append(np.vstack([outer4, middle4, inner3, center2]))

    # 17. Random perturbed multi-ring
    base_ring = make_regular_ngon(13, radius=1.0)
    random_perturb = base_ring + rng.normal(0, 0.05, base_ring.shape)
    initial_configs.append(random_perturb)

    # 18. Square grid subset
    grid_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            grid_pts.append([i * 0.5, j * 0.5])
    grid_pts = np.array(grid_pts)
    grid_pts -= np.mean(grid_pts, axis=0)
    if len(grid_pts) >= 13:
        dists = np.linalg.norm(grid_pts, axis=1)
        idx_sorted = np.argsort(dists)
        grid_init = grid_pts[idx_sorted[:13]]
        initial_configs.append(grid_init)

    # 19. Fibonacci spiral
    golden_angle = np.pi * (3 - np.sqrt(5))
    fib_t = np.linspace(0.2, 1.0, N)
    fib_theta = golden_angle * np.arange(N)
    fib_pts = np.column_stack((fib_t * np.cos(fib_theta), fib_t * np.sin(fib_theta)))
    fib_pts = fib_pts / np.max(np.linalg.norm(fib_pts, axis=1))
    initial_configs.append(fib_pts)

    # 20. Clustered points with one outlier
    cluster = make_regular_ngon(12, radius=0.8)
    outlier = np.array([[1.2, 0.0]])
    initial_configs.append(np.vstack([cluster, outlier]))

    # Optimization loop with adaptive tau
    # Phase 1: Broad exploration with lower tau
    # Phase 2: Refinement with higher tau
    # Phase 3: High-precision polish

    tau_schedule = [
        (20.0, 200),   # (tau, maxiter) - broad exploration
        (50.0, 300),   # medium refinement
        (100.0, 400),  # higher precision
        (200.0, 500),  # high precision
        (500.0, 600),  # very high precision
    ]

    for init_idx, init_pts in enumerate(initial_configs):
        if time_left() < 15.0:
            break

        try:
            params = points_to_params(init_pts)

            # Try multiple tau values, starting from lower
            current_params = params
            for tau, maxiter in tau_schedule:
                if time_left() < 10.0:
                    break

                # Smooth optimization with L-BFGS-B
                res = minimize(
                    smooth_objective,
                    current_params,
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8}
                )

                if res.success or res.status == 0:
                    current_params = res.x
                    candidate_pts = params_to_points(current_params)

                    # Check validity
                    if _hull_area(candidate_pts) < 1e-10:
                        continue

                    # Update best if improved
                    update_best(candidate_pts)

            # Final high-precision polish with Nelder-Mead on hard objective
            if time_left() > 5.0:
                res_nm = minimize(
                    hard_negative_score,
                    current_params,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )

                final_params = res_nm.x
                final_pts = params_to_points(final_params)
                update_best(final_pts)

        except Exception as e:
            # If optimization fails, try the raw initial config
            update_best(init_pts)
            continue

    # If no best found, return a regular polygon
    if best_points is None:
        best_points = make_regular_ngon(13, radius=1.0)

    return best_points
