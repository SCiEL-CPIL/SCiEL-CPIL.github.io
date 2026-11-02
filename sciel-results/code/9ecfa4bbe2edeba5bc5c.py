import numpy as np
from scipy.spatial import ConvexHull

N = 13

def _score_configuration(points):
    """
    Calculate the score for a given set of points.
    Score = min_triangle_area / convex_hull_area
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return 0.0

    # Ensure all points are distinct
    # Use a small tolerance
    unique_points = []
    for p in points:
        is_dup = False
        for q in unique_points:
            if np.linalg.norm(p - q) < 1e-9:
                is_dup = True
                break
        if not is_dup:
            unique_points.append(p)
    points = np.array(unique_points)
    
    if len(points) < 3:
        return 0.0

    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0

    if hull_area <= 1e-10:
        return 0.0

    # Calculate min triangle area using vectorized approach
    n = len(points)
    if n < 3:
        return 0.0

    # Get all unique triples
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx, j_idx, k_idx = i_idx[mask], j_idx[mask], k_idx[mask]

    p1 = points[i_idx]
    p2 = points[j_idx]
    p3 = points[k_idx]

    # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    areas = 0.5 * np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                         (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1]))

    min_area = np.min(areas)
    return min_area / hull_area


def _build_points(r, phi):
    """Build 13 points: center + 6 outer hexagon + 6 inner hexagon rotated by phi at radius r."""
    outer_angles = np.arange(6) * (np.pi / 3)
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner_angles = outer_angles + phi
    inner_pts = r * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    center = np.array([[0.0, 0.0]])
    return np.vstack([center, outer_pts, inner_pts])


def construct():
    best_score = 0.0
    best_points = None

    # Global grid search over r and phi
    # r: inner hexagon radius (0 to 1)
    # phi: angular offset of inner hexagon relative to outer (0 to pi/3)
    # By symmetry, phi in [0, pi/3] covers all distinct configurations

    # Phase 1: Coarse grid
    r_vals = np.linspace(0.05, 1.0, 100)
    phi_vals = np.linspace(0.0, np.pi / 3, 60)

    for r in r_vals:
        for phi in phi_vals:
            points = _build_points(r, phi)
            score = _score_configuration(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()

    if best_points is None:
        # Fallback: regular hexagon + center + tiny inner hexagon
        return _build_points(0.1, np.pi / 6)

    # Phase 2: Local refinement around best (r, phi)
    # Find the (r, phi) that gave best_score
    best_r = None
    best_phi = None
    for r in r_vals:
        for phi in phi_vals:
            points = _build_points(r, phi)
            score = _score_configuration(points)
            if abs(score - best_score) < 1e-12:
                best_r = r
                best_phi = phi
                break
        if best_r is not None:
            break

    # Fine grid around best_r, best_phi
    r_step = (r_vals[1] - r_vals[0]) if len(r_vals) > 1 else 0.01
    phi_step = (phi_vals[1] - phi_vals[0]) if len(phi_vals) > 1 else 0.01

    r_fine = np.linspace(max(0.01, best_r - 5 * r_step), min(1.0, best_r + 5 * r_step), 50)
    phi_fine = np.linspace(max(0.0, best_phi - 5 * phi_step), min(np.pi / 3, best_phi + 5 * phi_step), 50)

    for r in r_fine:
        for phi in phi_fine:
            points = _build_points(r, phi)
            score = _score_configuration(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()

    # Phase 3: Even finer local refinement
    # Find new best r, phi
    best_r2 = None
    best_phi2 = None
    for r in r_fine:
        for phi in phi_fine:
            points = _build_points(r, phi)
            score = _score_configuration(points)
            if abs(score - best_score) < 1e-12:
                best_r2 = r
                best_phi2 = phi
                break
        if best_r2 is not None:
            break

    r_fine2 = np.linspace(max(0.01, best_r2 - 2 * r_step / 5), min(1.0, best_r2 + 2 * r_step / 5), 30)
    phi_fine2 = np.linspace(max(0.0, best_phi2 - 2 * phi_step / 5), min(np.pi / 3, best_phi2 + 2 * phi_step / 5), 30)

    for r in r_fine2:
        for phi in phi_fine2:
            points = _build_points(r, phi)
            score = _score_configuration(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()

    return best_points
