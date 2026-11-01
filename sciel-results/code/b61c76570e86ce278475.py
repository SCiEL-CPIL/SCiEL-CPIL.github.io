"""Radial-ring decomposition for packing 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
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


# Define ring structure: counts per ring summing to 300
# Use a hexagonal-like progression: 1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29
# Sum: 1+6+12+18+24+30+36+42+48+54+29 = 300
ring_counts = np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29])
num_rings = len(ring_counts)


def _initial_guess():
    """Generate initial guess for radii and phases."""
    # Radii: evenly spaced from 0 to ~0.95
    radii = np.linspace(0.0, 0.95, num_rings)
    # Phases: start at 0
    phases = np.zeros(num_rings)
    return np.concatenate([radii, phases])


def _refine_with_nelder_mead(x0, ring_counts):
    """Refine a starting point using Nelder-Mead."""
    bounds = [(0.0, 1.0)] * num_rings + [(0.0, 2 * np.pi)] * num_rings
    result = minimize(
        _objective, x0, args=(ring_counts,),
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
    )
    return result.x


def construct():
    """Construct the packing configuration."""
    
    # Phase 1: Global optimization with differential evolution
    bounds_de = [(0.0, 1.0)] * num_rings + [(0.0, 2 * np.pi)] * num_rings
    de_result = differential_evolution(
        _objective, bounds_de, args=(ring_counts,),
        seed=42, maxiter=200, popsize=15, tol=1e-6,
        mutation=(0.3, 1.0), recombination=0.7, polish=True
    )
    x_de = de_result.x
    
    # Phase 2: Local refinement with Nelder-Mead from best DE result
    x_refined = _refine_with_nelder_mead(x_de, ring_counts)
    
    # Phase 3: Additional local refinement from a few different starting points
    best_params = x_refined
    best_obj = _objective(x_refined, ring_counts)
    
    # Try a few different starting points
    for trial in range(5):
        np.random.seed(trial)
        radii_init = np.random.uniform(0.0, 0.95, num_rings)
        phases_init = np.random.uniform(0.0, 2 * np.pi, num_rings)
        x_init = np.concatenate([radii_init, phases_init])
        x_trial = _refine_with_nelder_mead(x_init, ring_counts)
        obj_trial = _objective(x_trial, ring_counts)
        if obj_trial < best_obj:
            best_obj = obj_trial
            best_params = x_trial
    
    radii = np.maximum(best_params[:num_rings], 0.0)
    phases = best_params[num_rings:]
    
    points = _build_points(radii, phases, ring_counts)
    
    # Verify it's valid
    if len(points) != N:
        # Fallback to hexagonal lattice
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
    
    return points

# EVOLVE-BLOCK-END
