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


def _objective_radii(params, ring_counts, phases):
    """
    Objective function: negative of the packing radius.
    params are ring radii.
    phases are fixed.
    """
    radii = np.maximum(params, 0.0)
    points = _build_points(radii, phases, ring_counts)
    if len(points) != N:
        return -1e-9 # Invalid
    r = _compute_radius(points)
    return -r


def _get_staggered_phases(ring_counts):
    """
    Generate phases based on hexagonal staggering.
    Phase of ring i is offset from ring i-1 by half the angular step of ring i-1.
    """
    k = len(ring_counts)
    phases = np.zeros(k)
    for i in range(1, k):
        if ring_counts[i-1] > 0:
            # Offset by half the spacing of the previous ring
            phases[i] = phases[i-1] + np.pi / ring_counts[i-1]
        else:
            phases[i] = phases[i-1]
    return phases


def _generate_structures():
    """
    Generate a list of candidate ring structures.
    """
    structures = []
    
    # Base hexagonal: 1, 6, 12, 18, 24, 30, 36, 42, 48, 54
    base = [1, 6, 12, 18, 24, 30, 36, 42, 48, 54]
    sum_base = sum(base)
    rem = N - sum_base # 29
    
    # Structure 1: Last ring takes remainder
    structures.append(np.array(base + [rem]))
    
    # Structure 2: Split remainder into last two rings
    # e.g., 54 -> 40, rem+14?
    # Let's try moving some from the 54 ring to a new last ring
    for split in range(1, 55):
        if split < 54:
            s = base[:-1] + [54 - split, split + rem]
            # Ensure counts are positive and reasonable
            if all(x > 0 for x in s):
                structures.append(np.array(s))
                
    # Structure 3: Distribute remainder over last 3 rings
    # 48, 54 -> 48, 40, rem+14
    for split1 in range(10, 50, 5):
        for split2 in range(10, 50, 5):
            if split1 + split2 < 54 + 48: # rough check
                # This is getting messy, let's just use a few hand-picked ones
                pass
                
    # Hand-picked variations
    structures.append(np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 40, 43]))
    structures.append(np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 30, 53]))
    structures.append(np.array([1, 6, 12, 18, 24, 30, 36, 42, 42, 48, 41]))
    structures.append(np.array([1, 6, 12, 18, 24, 30, 36, 36, 48, 54, 35]))
    
    # Remove duplicates
    unique_structures = []
    for s in structures:
        if s not in unique_structures:
            unique_structures.append(s)
            
    return unique_structures


def _optimize_structure(ring_counts):
    """
    Optimize radii for a given ring structure with fixed staggered phases.
    """
    k = len(ring_counts)
    phases = _get_staggered_phases(ring_counts)
    
    # Initial guess for radii
    # Estimate r ~ 0.05
    r_est = 0.05
    radii_guess = np.zeros(k)
    for i, n in enumerate(ring_counts):
        if n > 1:
            radii_guess[i] = r_est / np.sin(np.pi / n)
        else:
            radii_guess[i] = 0.0
        # Add spacing
        radii_guess[i] += 2.0 * i * r_est * 0.5 # heuristic
    
    # Ensure monotonic increase
    for i in range(1, k):
        if radii_guess[i] < radii_guess[i-1]:
            radii_guess[i] = radii_guess[i-1] + 0.01
    
    # Bounds
    bounds = [(0.0, 1.0)] * k
    
    # Run optimization
    best_r = 0.0
    best_x = radii_guess
    
    # Try a few different starting points
    for seed in range(3):
        rng = np.random.default_rng(seed)
        x0 = radii_guess + rng.uniform(-0.05, 0.05, size=k)
        x0 = np.maximum(x0, 0.0)
        
        res = minimize(
            _objective_radii, x0, args=(ring_counts, phases),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        
        radii_opt = np.maximum(res.x, 0.0)
        points = _build_points(radii_opt, phases, ring_counts)
        if len(points) == N:
            r = _compute_radius(points)
            if r > best_r:
                best_r = r
                best_x = radii_opt
    
    return _build_points(best_x, phases, ring_counts), best_r


def construct():
    """Construct the packing configuration."""
    
    best_points = None
    best_r = 0.0
    
    structures = _generate_structures()
    
    for counts in structures:
        if sum(counts) != N:
            continue
        points, r = _optimize_structure(counts)
        if r > best_r:
            best_r = r
            best_points = points
            
    # Fallback if no valid configuration found
    if best_points is None:
        # Simple grid fallback
        spacing = 0.1
        pts = []
        for y in np.arange(-1, 1, spacing):
            for x in np.arange(-1, 1, spacing):
                if x**2 + y**2 <= (1 - spacing/2)**2:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            return pts[:N]
        else:
            # Random fallback
            return np.random.rand(N, 2) * 2 - 1 # Will be invalid, but prevents crash
            
    return best_points

# EVOLVE-BLOCK-END
