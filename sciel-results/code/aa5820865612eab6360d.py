"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull


def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        if len(points) < 3:
            return 0.0
        c = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - c[1], points[:, 0] - c[0])
        order = np.argsort(angles)
        pts = points[order]
        n = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        return abs(area) * 0.5


def _score(points):
    """Score = minimum triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _regular_polygon(n, radius, phase=0.0):
    """Generate vertices of a regular n-gon."""
    angles = np.linspace(phase, 2 * np.pi + phase, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack((x, y))


def _build_points_from_params(params):
    """
    Build 13 points from parameters:
    params = [r0, r1, r2, phase0, phase1, phase2]
    where r0, r1, r2 are radii of three concentric regular polygons,
    and phase0, phase1, phase2 are their rotation angles.
    
    We distribute 13 points as: 5 + 4 + 4 (or other partitions)
    Actually, let's use a more flexible partition:
    We'll try partitions like (5,4,4), (6,4,3), (7,3,3), etc.
    But to keep it simple, let's fix the partition as (5, 4, 4) for now,
    and also try (6, 4, 3) and (7, 3, 3) as separate candidates.
    
    For this function, we assume partition (5, 4, 4).
    """
    r0, r1, r2, p0, p1, p2 = params
    pts0 = _regular_polygon(5, r0, p0)
    pts1 = _regular_polygon(4, r1, p1)
    pts2 = _regular_polygon(4, r2, p2)
    return np.vstack([pts0, pts1, pts2])


def _build_points_partition(partition, params):
    """
    Build points from a given partition and parameters.
    partition: list of integers summing to 13
    params: [r0, r1, ..., r_{m-1}, p0, p1, ..., p_{m-1}]
    """
    m = len(partition)
    radii = params[:m]
    phases = params[m:]
    pts_list = []
    for i, (n_i, r_i, p_i) in enumerate(zip(partition, radii, phases)):
        pts_list.append(_regular_polygon(n_i, r_i, p_i))
    return np.vstack(pts_list)


def _objective_partition(partition, x):
    """Objective function for a given partition: minimize negative score."""
    pts = _build_points_partition(partition, x)
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    min_area = _min_triangle_area(pts)
    return -min_area / hull_area


def _regular_polygon_13():
    """Generate vertices of a regular 13-gon on the unit circle."""
    return _regular_polygon(13, 1.0, 0.0)


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    # Baseline: regular 13-gon
    best_points = _regular_polygon_13()
    best_score = _score(best_points)

    # Define several partitions of 13 into 2 or 3 parts
    # Each partition is a list of polygon sizes
    partitions = [
        (5, 4, 4),
        (6, 4, 3),
        (7, 3, 3),
        (5, 5, 3),
        (6, 5, 2),  # 2-gon is just a line segment, but let's see
        (4, 4, 5),
        (7, 6),
        (8, 5),
        (9, 4),
        (10, 3),
        (11, 2),
        (12, 1),  # 1-point "polygon" is just a point
    ]

    # Filter out partitions with parts < 2 (a 1-gon or 2-gon is degenerate for our purposes,
    # but a 2-gon is just two points, which is fine. A 1-gon is one point, also fine.)
    # Actually, regular_polygon(1, r, p) returns one point. regular_polygon(2, r, p) returns two points.
    # Let's keep them all.

    for partition in partitions:
        if time.time() > deadline:
            break

        m = len(partition)
        # Parameters: m radii, m phases
        # Radii bounds: [0.01, 1.5] (allow outer radius > 1 for normalization flexibility, 
        # but since we normalize by hull area, scale doesn't matter, only relative)
        # Actually, since score is scale-invariant, we can fix one radius or use relative radii.
        # Let's fix r0 = 1.0 and optimize r1/r0, r2/r0, etc.
        # Or just let DE optimize all radii in [0.01, 2.0].
        
        bounds = []
        for i in range(m):
            bounds.append((0.01, 2.0))  # radius
        for i in range(m):
            bounds.append((0.0, 2 * np.pi))  # phase

        # Initial guess: equal spacing
        x0 = []
        for i in range(m):
            x0.append(1.0)  # radius
        for i in range(m):
            x0.append(0.0)  # phase

        try:
            result = differential_evolution(
                _objective_partition,
                bounds=bounds,
                args=(partition,),
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                init='latinhypercube',
                polish=True,
                update_interval=1,
                callback=None
            )
            
            if time.time() > deadline:
                break

            candidate = _build_points_partition(partition, result.x)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate

        except Exception:
            continue

    # Also try a 2-shell structure with optimized radii and phase
    # This is already covered by partitions like (7,6), (8,5), etc.

    # Final polish with Nelder-Mead on the best configuration found
    if time.time() < deadline - 10:
        # We need to map the best points back to parameters for polishing
        # This is tricky because we don't know which partition produced the best.
        # Instead, let's do a local search in coordinate space on the best points.
        
        def objective_coords(x):
            pts = x.reshape(-1, 2)
            hull_area = _convex_hull_area(pts)
            if hull_area < 1e-10:
                return 1e10
            min_area = _min_triangle_area(pts)
            return -min_area / hull_area

        try:
            res = minimize(
                objective_coords,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            candidate = res.x.reshape(-1, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate
        except Exception:
            pass

    return best_points
# EVOLVE-BLOCK-END
