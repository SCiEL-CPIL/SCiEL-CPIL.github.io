"""Radial-ring decomposition for packing 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _build_points(radii, phases, ring_counts):
    """
    Build point array from ring radii, phases, and counts.
    Each ring i has count_i points at radius radii[i] with angular offset phases[i].
    """
    points = []
    for i, (r, phase, count) in enumerate(zip(radii, phases, ring_counts)):
        if count == 0:
            continue
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + phase
        xs = r * np.cos(angles)
        ys = r * np.sin(angles)
        points.append(np.column_stack([xs, ys]))
    
    if not points:
        return np.zeros((0, 2))
    
    return np.vstack(points)


def _compute_radius(points):
    """Compute the largest common radius for the given points."""
    if len(points) == 0:
        return 0.0
    if len(points) == 1:
        return 1.0 - np.hypot(*points[0])
    
    # Distance to boundary
    dists_boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(dists_boundary)
    
    # Pairwise distances
    dist_matrix = cdist(points, points)
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = np.min(dist_matrix) / 2.0
    
    return min(min_boundary, min_pair)


def _min_ring_radius_for_circles(n, r):
    """Minimum ring radius to fit n circles of radius r on the ring."""
    if n <= 1:
        return 0.0
    # Chord distance between adjacent circles must be >= 2r
    # chord = 2 * R * sin(pi/n) >= 2r  =>  R >= r / sin(pi/n)
    return r / np.sin(np.pi / n)


def _min_radial_gap(r):
    """Minimum radial gap between adjacent rings to avoid overlap."""
    # For circles of radius r on adjacent rings, the minimum radial gap
    # depends on angular alignment. With optimal staggering, the gap can be
    # smaller than 2r. For now, use 2r as a conservative bound.
    return 2.0 * r


def _compute_ring_radii(r, ring_counts):
    """
    Compute the minimum ring radii for a given circle radius r.
    Returns array of ring radii or None if infeasible.
    """
    k = len(ring_counts)
    radii = np.zeros(k)
    
    # Ring 0: if count is 1, it's at center (radius 0)
    # Otherwise, it needs radius >= r / sin(pi/n_0)
    radii[0] = _min_ring_radius_for_circles(ring_counts[0], r)
    
    # For each subsequent ring, ensure:
    # 1. Ring radius >= r / sin(pi/n_i)
    # 2. Radial gap from previous ring >= 2r (conservative)
    for i in range(1, k):
        min_from_circ = _min_ring_radius_for_circles(ring_counts[i], r)
        min_from_gap = radii[i-1] + _min_radial_gap(r)
        radii[i] = max(min_from_circ, min_from_gap)
    
    # Check boundary: outermost ring + r <= 1
    if radii[-1] + r > 1.0 + 1e-12:
        return None
    
    return radii


def _is_feasible(r, ring_counts):
    """Check if circle radius r is feasible with the given ring structure."""
    radii = _compute_ring_radii(r, ring_counts)
    return radii is not None


def _binary_search_radius(ring_counts, r_low=1e-6, r_high=0.1, tol=1e-10, max_iter=100):
    """
    Binary search for the maximum circle radius r such that the ring
    structure is feasible.
    """
    for _ in range(max_iter):
        r_mid = (r_low + r_high) / 2.0
        if _is_feasible(r_mid, ring_counts):
            r_low = r_mid
        else:
            r_high = r_mid
        if r_high - r_low < tol:
            break
    
    return r_low


def _build_initial_config(ring_counts, r):
    """
    Build initial configuration with optimal ring radii and staggered phases.
    """
    radii = _compute_ring_radii(r, ring_counts)
    if radii is None:
        return None
    
    k = len(ring_counts)
    phases = np.zeros(k)
    
    # Stagger phases: offset each ring by half the angular spacing of the previous
    for i in range(1, k):
        if ring_counts[i-1] > 1:
            phases[i] = np.pi / ring_counts[i-1]
        else:
            phases[i] = 0.0
    
    points = _build_points(radii, phases, ring_counts)
    return points


def _objective_refine(params, ring_counts):
    """
    Objective for local refinement: negative of packing radius.
    params = [r0, r1, ..., r_{k-1}, phi0, phi1, ..., phi_{k-1}]
    """
    k = len(ring_counts)
    radii = params[:k]
    phases = params[k:]
    
    radii = np.maximum(radii, 0.0)
    points = _build_points(radii, phases, ring_counts)
    r = _compute_radius(points)
    return -r


def construct():
    """Construct the packing configuration."""
    
    # Define ring structure: hexagonal-like pattern
    # 1 + 6 + 12 + 18 + 24 + 30 + 36 + 42 + 48 + 54 = 271
    # Add 29 more to the outermost ring: 54 + 29 = 83? No, that's too many.
    # Let's use: 1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29 = 300
    ring_counts = np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29])
    num_rings = len(ring_counts)
    
    # Binary search for maximum feasible radius
    r_max = _binary_search_radius(ring_counts, r_low=0.01, r_high=0.08, tol=1e-12, max_iter=80)
    
    # Build initial configuration
    points = _build_initial_config(ring_counts, r_max)
    
    if points is None or len(points) != N:
        # Fallback: try slightly smaller radius
        for scale in [0.99, 0.98, 0.97, 0.96, 0.95]:
            r_try = r_max * scale
            points = _build_initial_config(ring_counts, r_try)
            if points is not None and len(points) == N:
                break
        if points is None:
            # Hexagonal lattice fallback
            spacing = 2.0 / (np.sqrt(N) + 1.0)
            while True:
                k = int(1.0 / spacing) + 2
                rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                        for i in range(-k, k + 1) for j in range(-k, k + 1)]
                pts = np.array(rows)
                radius = np.hypot(pts[:, 0], pts[:, 1])
                pts, radius = pts[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
                if len(pts) >= N:
                    return pts[np.argsort(radius, kind="stable")[:N]]
                spacing *= 0.98
    
    # Local refinement using Nelder-Mead
    radii = np.zeros(num_rings)
    phases = np.zeros(num_rings)
    
    # Extract current radii and phases from points
    idx = 0
    for i, count in enumerate(ring_counts):
        if count == 0:
            continue
        ring_pts = points[idx:idx+count]
        if count == 1:
            radii[i] = np.hypot(ring_pts[0, 0], ring_pts[0, 1])
            phases[i] = np.arctan2(ring_pts[0, 1], ring_pts[0, 0])
        else:
            radii[i] = np.mean(np.hypot(ring_pts[:, 0], ring_pts[:, 1]))
            # Estimate phase from first point
            phases[i] = np.arctan2(ring_pts[0, 1], ring_pts[0, 0])
        idx += count
    
    x0 = np.concatenate([radii, phases])
    bounds = [(0.0, 1.0)] * num_rings + [(0.0, 2 * np.pi)] * num_rings
    
    result = minimize(
        _objective_refine, x0, args=(ring_counts,),
        method='Nelder-Mead',
        options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
    )
    
    radii_refined = np.maximum(result.x[:num_rings], 0.0)
    phases_refined = result.x[num_rings:]
    
    points_refined = _build_points(radii_refined, phases_refined, ring_counts)
    
    if len(points_refined) != N:
        return points
    
    r_initial = _compute_radius(points)
    r_refined = _compute_radius(points_refined)
    
    if r_refined > r_initial:
        return points_refined
    
    return points

# EVOLVE-BLOCK-END
