"""Radial-ring decomposition for packing 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist


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


def _build_points(radii, phases, ring_counts):
    """
    Build point array from ring radii, phases, and counts.
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


def _objective(params, ring_counts):
    """
    Objective function: negative of the packing radius.
    """
    k = len(ring_counts)
    radii = params[:k]
    phases = params[k:]
    
    # Ensure radii are non-negative
    radii = np.maximum(radii, 0.0)
    
    points = _build_points(radii, phases, ring_counts)
    r = _compute_radius(points)
    return -r


def _ring_structure_bounds(ring_counts):
    """
    Compute reasonable bounds for ring radii and phases.
    """
    k = len(ring_counts)
    bounds = []
    for i, n in enumerate(ring_counts):
        # Ring radius bounds
        if n == 1:
            r_min = 0.0
        else:
            # For n circles on a ring, minimum radius to avoid overlap
            # r_i >= r / sin(pi/n) where r is the circle radius
            # But we don't know r yet, so use a conservative bound
            r_min = 0.0
        r_max = 1.0
        bounds.append((r_min, r_max))
        # Phase bounds
        bounds.append((0.0, 2 * np.pi))
    
    return bounds


def _optimize_ring_structure(ring_counts, n_initial=10, max_iter=3000):
    """
    Optimize a given ring structure using differential evolution
    followed by local refinement.
    """
    k = len(ring_counts)
    bounds = _ring_structure_bounds(ring_counts)
    
    best_r = 0.0
    best_params = None
    
    for seed in range(n_initial):
        rng = np.random.default_rng(seed)
        
        # Initial guess: staggered rings
        x0 = np.zeros(2 * k)
        for i in range(k):
            n = ring_counts[i]
            if n == 1:
                x0[i] = 0.0  # Center
            else:
                # Estimate ring radius based on position
                # Use a heuristic: rings are roughly evenly spaced
                x0[i] = (i + 1) / (k + 1) * 0.9
            # Phase: stagger adjacent rings
            if i > 0 and ring_counts[i-1] > 1:
                x0[i + k] = np.pi / ring_counts[i-1]
            else:
                x0[i + k] = 0.0
        
        # Add some randomness
        x0 += rng.uniform(-0.05, 0.05, size=2 * k)
        x0[:k] = np.maximum(x0[:k], 0.0)
        x0[k:] = np.mod(x0[k:], 2 * np.pi)
        
        # Local optimization from this initial guess
        result = minimize(
            _objective, x0, args=(ring_counts,),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        
        radii = np.maximum(result.x[:k], 0.0)
        phases = np.mod(result.x[k:], 2 * np.pi)
        
        points = _build_points(radii, phases, ring_counts)
        
        if len(points) != N:
            continue
        
        r = _compute_radius(points)
        if r > best_r:
            best_r = r
            best_params = (radii, phases)
    
    return best_r, best_params


def _try_ring_structures():
    """
    Try multiple ring structures and return the best configuration.
    """
    # Define various ring structures that sum to 300
    structures = [
        # Classic hexagonal-like: 1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29
        np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29]),
        # More uniform rings
        np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 36, 33]),
        # Fewer, larger rings
        np.array([1, 12, 24, 36, 48, 60, 72, 47]),
        # Dense inner, sparse outer
        np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 42, 36, 30, 15]),
        # Very dense center
        np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 19]),
        # Balanced
        np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 48, 21]),
        # Alternate pattern
        np.array([1, 8, 16, 24, 32, 40, 48, 44, 40, 36, 32, 29]),
        # Sparse outer
        np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 19]),
    ]
    
    best_r = 0.0
    best_points = None
    best_structure = None
    
    for ring_counts in structures:
        if np.sum(ring_counts) != N:
            continue
        
        r, params = _optimize_ring_structure(ring_counts, n_initial=8, max_iter=2000)
        
        if r > best_r:
            best_r = r
            radii, phases = params
            best_points = _build_points(radii, phases, ring_counts)
            best_structure = ring_counts
    
    return best_r, best_points


def _local_refinement(points, max_iter=5000):
    """
    Perform local refinement of the packing by moving individual points.
    """
    best_points = points.copy()
    best_r = _compute_radius(points)
    
    for iteration in range(50):
        improved = False
        for i in range(N):
            # Try moving point i in small increments
            for dx in [-0.005, 0.0, 0.005]:
                for dy in [-0.005, 0.0, 0.005]:
                    new_points = best_points.copy()
                    new_points[i] = best_points[i] + np.array([dx, dy])
                    
                    # Check if still inside container
                    if np.hypot(*new_points[i]) >= 1.0:
                        continue
                    
                    r = _compute_radius(new_points)
                    if r > best_r + 1e-12:
                        best_r = r
                        best_points = new_points
                        improved = True
        
        if not improved:
            break
    
    return best_points, best_r


def _fallback_hexagonal():
    """Fallback hexagonal lattice packing."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        pts = np.array(rows)
        radius = np.hypot(pts[:, 0], pts[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        pts = pts[mask]
        if len(pts) >= N:
            return pts[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98


def construct():
    """Construct the packing configuration."""
    
    # First, try ring structure optimization
    best_r, best_points = _try_ring_structures()
    
    # If we got a valid result, try local refinement
    if best_points is not None and len(best_points) == N:
        refined_points, refined_r = _local_refinement(best_points)
        if refined_r > best_r:
            best_r = refined_r
            best_points = refined_points
    
    # If ring structure didn't work well, use fallback
    if best_points is None or len(best_points) != N:
        best_points = _fallback_hexagonal()
    
    return best_points

# EVOLVE-BLOCK-END
