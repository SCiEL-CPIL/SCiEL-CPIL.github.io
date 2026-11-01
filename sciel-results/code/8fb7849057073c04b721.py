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
    params = [R0, delta1, ..., delta_{k-1}, phi0, phi1, ..., phi_{k-1}]
    Radii are constructed as R_i = R0 + sum(deltas[:i+1])
    """
    k = len(ring_counts)
    R0 = params[0]
    deltas = params[1:k]
    phases = params[k:]
    
    # Construct radii
    radii = np.zeros(k)
    radii[0] = R0
    for i in range(1, k):
        radii[i] = radii[i-1] + deltas[i-1]
    
    # Ensure non-negative radii (deltas can be negative in NM, so we clip or penalize)
    # To keep it simple for NM, we allow negative deltas but the resulting radii must be positive.
    # If any radius is negative, return a large penalty.
    if np.any(radii < 0):
        return 1e6
    
    points = _build_points(radii, phases, ring_counts)
    r = _compute_radius(points)
    return -r


def _initial_guess(ring_counts, jitter_seed=0):
    """Generate initial guess for R0, deltas, and phases."""
    rng = np.random.default_rng(jitter_seed)
    k = len(ring_counts)
    
    # Estimate target radius
    r_target = 0.052
    
    # Estimate radii based on constraints
    # R_i >= r_target / sin(pi/n_i)
    est_radii = np.zeros(k)
    for i, n in enumerate(ring_counts):
        if n > 1:
            est_radii[i] = r_target / np.sin(np.pi / n)
        else:
            est_radii[i] = 0.0
        
    # Ensure increasing radii
    for i in range(1, k):
        if est_radii[i] <= est_radii[i-1]:
            est_radii[i] = est_radii[i-1] + 2 * r_target
            
    # Scale to fit in unit circle
    if est_radii[-1] + r_target > 1.0:
        scale = (1.0 - r_target) / est_radii[-1]
        est_radii *= scale
        
    # Convert to R0 and deltas
    R0 = est_radii[0]
    deltas = np.diff(est_radii)
    
    # Add jitter
    R0 += rng.uniform(-0.01, 0.01)
    deltas += rng.uniform(-0.01, 0.01, size=k-1)
    
    # Phases
    phases = np.zeros(k)
    for i in range(1, k):
        if ring_counts[i] > 0 and ring_counts[i-1] > 0:
            phases[i] = np.pi / ring_counts[i-1] + rng.uniform(-0.1, 0.1)
            
    return np.concatenate([[R0], deltas, phases])


def construct():
    """Construct the packing configuration."""
    
    best_points = None
    best_r = 0.0
    
    # Define candidate ring structures
    # Base hexagonal layers: 1, 6, 12, 18, 24, 30, 36, 42, 48, 54
    # Sum = 271. Need 29 more.
    base_counts = [1, 6, 12, 18, 24, 30, 36, 42, 48, 54]
    remaining = 29
    
    candidates = []
    
    # Variation 1: Current
    candidates.append(np.array(base_counts + [remaining]))
    
    # Variation 2: Distribute remaining differently
    # Try moving some from last ring to previous
    for shift in range(0, 15):
        if shift == 0: continue
        new_counts = base_counts.copy()
        new_counts[-1] = remaining - shift
        new_counts[-2] += shift
        if new_counts[-1] > 0:
            candidates.append(np.array(new_counts))
            
    # Variation 3: Adjust earlier rings
    # Try reducing ring 8 (48) and increasing ring 9 (54)
    for shift in range(1, 10):
        new_counts = base_counts.copy()
        new_counts[8] = 48 - shift
        new_counts[9] = 54 + shift
        # Adjust last ring to keep sum 300
        # Sum so far: 271 - shift + shift = 271. Last is 29.
        candidates.append(np.array(new_counts + [29]))
        
    # Variation 4: Different inner structure?
    # Maybe 1, 6, 12, 18, 24, 30, 36, 42, 48, 53, 30
    candidates.append(np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 53, 30]))
    candidates.append(np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 52, 31]))
    candidates.append(np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 51, 32]))
    
    # Remove duplicates
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
            
    for ring_counts in unique_candidates:
        if sum(ring_counts) != N:
            continue
            
        num_rings = len(ring_counts)
        
        # Try multiple initial guesses
        for seed in range(10):
            x0 = _initial_guess(ring_counts, jitter_seed=seed)
            
            # Bounds: R0 > 0, deltas > -R_prev (hard to bound easily without knowing prev), phases [0, 2pi]
            # NM doesn't use bounds effectively, so we rely on penalty for negative radii.
            
            result = minimize(
                _objective, x0, args=(ring_counts,),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            R0 = result.x[0]
            deltas = result.x[1:num_rings]
            phases = result.x[num_rings:]
            
            # Reconstruct radii
            radii = np.zeros(num_rings)
            radii[0] = R0
            for i in range(1, num_rings):
                radii[i] = radii[i-1] + deltas[i-1]
                
            if np.any(radii < 0):
                continue
                
            points = _build_points(radii, phases, ring_counts)
            
            if len(points) != N:
                continue
                
            r = _compute_radius(points)
            if r > best_r:
                best_r = r
                best_points = points
    
    # Fallback
    if best_points is None:
        # Simple grid
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            pts = np.array(rows)
            radius = np.hypot(pts[:, 0], pts[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            pts, radius = pts[mask], radius[mask]
            if len(pts) >= N:
                return pts[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98
            
    return best_points

# EVOLVE-BLOCK-END
