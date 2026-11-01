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


def _objective(params, ring_counts):
    """
    Objective function: negative of the packing radius.
    params = [r0, r1, ..., r_{k-1}, phi0, phi1, ..., phi_{k-1}]
    First half are radii, second half are angular phases.
    """
    k = len(ring_counts)
    radii = params[:k]
    phases = params[k:]
    
    # Ensure radii are non-negative
    radii = np.maximum(radii, 0.0)
    
    points = _build_points(radii, phases, ring_counts)
    r = _compute_radius(points)
    return -r


def _initial_guess(ring_counts, jitter_seed=0, r_target=0.053):
    """Generate initial guess for radii and phases."""
    rng = np.random.default_rng(jitter_seed)
    
    k = len(ring_counts)
    
    # Compute constraint-based radii
    radii = np.zeros(k)
    for i, n in enumerate(ring_counts):
        if n > 1:
            r_min_ring = r_target / np.sin(np.pi / n)
        else:
            r_min_ring = 0.0
        r_min_adj = 2.0 * i * r_target
        radii[i] = max(r_min_ring, r_min_adj)
    
    # Scale so outermost ring + r_target <= 1
    if radii[-1] > 0:
        scale = (1.0 - r_target) / radii[-1]
        if scale < 1.0:
            radii *= scale
    
    # Add small jitter
    radii = radii + rng.uniform(-0.005, 0.005, size=k)
    radii = np.maximum(radii, 0.0)
    
    # Phases: stagger adjacent rings for better packing
    phases = np.zeros(k)
    for i in range(1, k):
        if ring_counts[i] > 0 and ring_counts[i-1] > 0:
            # Offset by half the angular spacing of the previous ring
            phases[i] = np.pi / ring_counts[i-1] + rng.uniform(-0.05, 0.05)
    
    return np.concatenate([radii, phases])


def construct():
    """Construct the packing configuration."""
    
    best_points = None
    best_r = 0.0
    
    # Try multiple ring structures
    ring_structures = [
        np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29]),
        np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 53, 30]),
        np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 52, 31]),
        np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 51, 32]),
        np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 50, 33]),
    ]
    
    for ring_counts in ring_structures:
        if len(ring_counts) == 0 or np.sum(ring_counts) != N:
            continue
        
        num_rings = len(ring_counts)
        
        # Try multiple initial guesses with different jitters and r_targets
        for seed in range(8):
            for r_target in [0.052, 0.053, 0.054]:
                x0 = _initial_guess(ring_counts, jitter_seed=seed, r_target=r_target)
                
                # Bounds: radii in [0, 1], phases in [0, 2*pi]
                bounds = [(0.0, 1.0)] * num_rings + [(0.0, 2 * np.pi)] * num_rings
                
                # Use Nelder-Mead for robustness with non-smooth objective
                try:
                    result = minimize(
                        _objective, x0, args=(ring_counts,),
                        method='Nelder-Mead',
                        options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                    )
                    
                    radii = np.maximum(result.x[:num_rings], 0.0)
                    phases = result.x[num_rings:]
                    
                    points = _build_points(radii, phases, ring_counts)
                    
                    if len(points) != N:
                        continue
                    
                    r = _compute_radius(points)
                    if r > best_r:
                        best_r = r
                        best_points = points.copy()
                except Exception:
                    continue
    
    # If we found a good solution, return it
    if best_points is not None and best_r > 0.05:
        return best_points
    
    # Fallback: try a simpler grid-based approach
    # Hexagonal lattice fallback
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while spacing > 0.01:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        pts = np.array(rows)
        radius = np.hypot(pts[:, 0], pts[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        pts, radius = pts[mask], radius[mask]
        if len(pts) >= N:
            # Sort by radius and take first N
            idx = np.argsort(radius, kind="stable")[:N]
            return pts[idx]
        spacing *= 0.97
    
    # Ultimate fallback: concentric rings with simple spacing
    ring_counts = np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29])
    num_rings = len(ring_counts)
    r_target = 0.05
    radii = np.zeros(num_rings)
    for i, n in enumerate(ring_counts):
        if n > 1:
            radii[i] = r_target / np.sin(np.pi / n)
        else:
            radii[i] = 0.0
    scale = (1.0 - r_target) / radii[-1] if radii[-1] > 0 else 1.0
    radii *= scale
    phases = np.zeros(num_rings)
    for i in range(1, num_rings):
        phases[i] = np.pi / ring_counts[i-1]
    
    points = _build_points(radii, phases, ring_counts)
    if len(points) == N:
        return points
    
    # Last resort: uniform grid
    pts = []
    for i in range(20):
        for j in range(20):
            x = -1 + 2 * i / 19
            y = -1 + 2 * j / 19
            if x*x + y*y <= 0.9:
                pts.append([x, y])
    pts = np.array(pts[:N])
    return pts

# EVOLVE-BLOCK-END
