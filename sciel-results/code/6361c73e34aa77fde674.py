"""Radial-ring decomposition for packing 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
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
    
    # Ensure radii are non-negative and sorted
    radii = np.maximum(radii, 0.0)
    
    points = _build_points(radii, phases, ring_counts)
    r = _compute_radius(points)
    return -r


# Define ring structure: counts per ring summing to 300
# Use a hexagonal-like progression
ring_counts = np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29])
num_rings = len(ring_counts)


def _analytic_initial_radii(ring_counts, target_r=None):
    """
    Compute initial radii based on the constraint that adjacent rings
    don't overlap. For two rings with counts n1, n2 at radii r1, r2,
    the minimum distance between points is at least 2*target_r.
    
    The angular separation between adjacent points on a ring with count n
    is 2*pi/n, so the chord length is 2*r*sin(pi/n).
    For adjacent rings, we need the distance between closest points >= 2*r.
    
    A simple heuristic: space rings so that the radial gap plus the 
    minimum chord on each ring provides enough separation.
    """
    if target_r is None:
        # Estimate target radius
        target_r = 0.053
    
    radii = np.zeros(len(ring_counts))
    if len(ring_counts) > 0:
        radii[0] = 0.0  # Center
    
    for i in range(1, len(ring_counts)):
        n_prev = ring_counts[i-1]
        n_curr = ring_counts[i]
        
        # Minimum chord length on each ring
        chord_prev = 2 * radii[i-1] * np.sin(np.pi / n_prev) if n_prev > 1 else 0
        chord_curr = 2 * radii[i-1] * np.sin(np.pi / n_curr)  # estimate
        
        # We need radial gap such that the distance between rings >= 2*target_r
        # Approximate: sqrt(dr^2 + (chord/2)^2) >= 2*target_r
        # So dr >= sqrt((2*target_r)^2 - (chord/2)^2)
        chord_est = 2 * radii[i-1] * np.sin(np.pi / max(n_prev, 2))
        min_dr = np.sqrt(max((2*target_r)**2 - (chord_est/2)**2, (target_r)**2))
        
        radii[i] = radii[i-1] + min_dr
    
    # Scale to fit in unit circle
    max_r = radii[-1]
    if max_r > 0:
        scale = (1.0 - target_r) / max_r
        radii *= scale
    
    return radii


def _initial_guess(seed=0):
    """Generate initial guess for radii and phases with some randomness."""
    rng = np.random.RandomState(seed)
    
    # Use analytic initial radii
    radii = _analytic_initial_radii(ring_counts)
    
    # Add small random perturbation
    radii = radii + rng.uniform(-0.02, 0.02, size=num_rings)
    radii = np.maximum(radii, 0.0)
    radii = np.sort(radii)  # Ensure sorted
    
    # Phases: random offsets
    phases = rng.uniform(0, 2 * np.pi, size=num_rings)
    
    return np.concatenate([radii, phases])


def construct():
    """Construct the packing configuration."""
    
    best_points = None
    best_radius = 0.0
    
    # Try multiple initial guesses with Nelder-Mead
    for seed in range(5):
        x0 = _initial_guess(seed)
        
        # Bounds: radii in [0, 1], phases in [0, 2*pi]
        bounds = [(0.0, 1.0)] * num_rings + [(0.0, 2 * np.pi)] * num_rings
        
        # Use Nelder-Mead for robustness with non-smooth objective
        result = minimize(
            _objective, x0, args=(ring_counts,),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
        )
        
        best_params = result.x
        radii = np.maximum(best_params[:num_rings], 0.0)
        phases = best_params[num_rings:]
        
        points = _build_points(radii, phases, ring_counts)
        
        if len(points) == N:
            r = _compute_radius(points)
            if r > best_radius:
                best_radius = r
                best_points = points
    
    # If Nelder-Mead didn't work well, try differential evolution on a subset
    if best_radius < 0.05:
        # Use differential evolution for global optimization
        bounds_de = [(0.0, 1.0)] * num_rings + [(0.0, 2 * np.pi)] * num_rings
        
        result_de = differential_evolution(
            _objective, bounds_de, args=(ring_counts,),
            seed=42, maxiter=200, tol=1e-8, popsize=15,
            mutation=(0.5, 1.5), recombination=0.7
        )
        
        best_params = result_de.x
        radii = np.maximum(best_params[:num_rings], 0.0)
        phases = best_params[num_rings:]
        
        points = _build_points(radii, phases, ring_counts)
        
        if len(points) == N:
            r = _compute_radius(points)
            if r > best_radius:
                best_radius = r
                best_points = points
    
    # Final local refinement with L-BFGS-B
    if best_points is not None and best_radius > 0:
        # Extract current parameters
        radii = np.maximum(best_params[:num_rings], 0.0)
        phases = best_params[num_rings:]
        x0 = np.concatenate([radii, phases])
        
        bounds_final = [(0.0, 1.0)] * num_rings + [(0.0, 2 * np.pi)] * num_rings
        
        result_final = minimize(
            _objective, x0, args=(ring_counts,),
            method='L-BFGS-B',
            bounds=bounds_final,
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        best_params = result_final.x
        radii = np.maximum(best_params[:num_rings], 0.0)
        phases = best_params[num_rings:]
        
        points = _build_points(radii, phases, ring_counts)
        
        if len(points) == N:
            r = _compute_radius(points)
            if r > best_radius:
                best_radius = r
                best_points = points
    
    # Fallback to hexagonal lattice if all else fails
    if best_points is None or len(best_points) != N:
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
    
    return best_points

# EVOLVE-BLOCK-END
