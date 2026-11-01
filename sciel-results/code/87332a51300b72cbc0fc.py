"""Radial-ring decomposition for packing 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _ring_constraint(r, n):
    """
    For a ring of radius r with n circles, the minimum adjacent distance is r*sin(pi/n).
    The circle radius must be <= r*sin(pi/n)/2.
    Also, the ring must fit inside the container: r + circle_radius <= 1.
    """
    if n <= 0:
        return 0.0
    if n == 1:
        # Single circle at center or on ring
        return min(r, 1.0 - r) if r > 0 else 1.0
    # Adjacent distance constraint
    adj_dist = r * np.sin(np.pi / n)
    return adj_dist / 2.0


def _inter_ring_constraint(r_i, r_j, n_i, n_j, phase_diff):
    """
    Minimum distance between circles on ring i and ring j.
    This is complex; we approximate using the closest approach.
    """
    if n_i == 0 or n_j == 0:
        return np.inf
    if n_i == 1 and n_j == 1:
        # Both are single points (center and one other, or two centers - shouldn't happen)
        return abs(r_i - r_j)
    
    # For rings with multiple circles, the minimum inter-ring distance depends on angular alignment
    # Approximate: the minimum distance is at least |r_i - r_j| but can be less due to angular offset
    # A better lower bound: consider the closest angular positions
    # For simplicity, use |r_i - r_j| as a conservative estimate, but we'll refine in optimization
    
    # Actually, the minimum distance between two circles on different rings:
    # If both rings have many circles, the minimum distance is approximately sqrt(r_i^2 + r_j^2 - 2*r_i*r_j*cos(delta))
    # where delta is the minimum angular difference between any circle on ring i and any on ring j
    # The minimum delta is 0 if the phases align, or pi/max(n_i,n_j) if they're offset
    
    # For a conservative estimate, assume worst case (closest approach)
    # The minimum possible distance is |r_i - r_j| (when aligned radially)
    # But with angular offsets, it can be larger
    
    # Let's compute the actual minimum distance for given phases
    if n_i == 1:
        # One point on ring i
        angles_i = np.array([0.0])  # phase doesn't matter for single point
        angles_j = np.linspace(0, 2*np.pi, n_j, endpoint=False)
        # Distance from point at (r_i, 0) to each point on ring j
        dists = np.sqrt(r_i**2 + r_j**2 - 2*r_i*r_j*np.cos(angles_j))
        return np.min(dists) / 2.0
    
    if n_j == 1:
        angles_i = np.linspace(0, 2*np.pi, n_i, endpoint=False)
        dists = np.sqrt(r_i**2 + r_j**2 - 2*r_i*r_j*np.cos(angles_i))
        return np.min(dists) / 2.0
    
    # Both have multiple circles
    # The minimum distance occurs when circles are angularly aligned
    # The angular spacing on ring i is 2*pi/n_i, on ring j is 2*pi/n_j
    # The minimum angular difference between any pair is 0 (if we can align them)
    # So the minimum distance is |r_i - r_j|
    return abs(r_i - r_j) / 2.0


def _build_points(radii, phases, ring_counts):
    """
    Build point array from ring radii, phases, and counts.
    Each ring i has count_i points at radius radii[i] with angular offset phases[i].
    """
    points = []
    for i, (r, phase, count) in enumerate(zip(radii, phases, ring_counts)):
        if count == 0:
            continue
        if count == 1:
            # Single point: place at angle = phase
            x = r * np.cos(phase)
            y = r * np.sin(phase)
            points.append(np.array([[x, y]]))
        else:
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
    
    # Pairwise distances - use efficient computation
    # For 300 points, this is manageable
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(points, points)
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = np.min(dist_matrix) / 2.0
    
    return min(min_boundary, min_pair)


def _packing_radius_from_params(radii, phases, ring_counts):
    """
    Compute the maximum common circle radius given ring parameters.
    This is the minimum of:
    - For each ring: the adjacent circle distance / 2
    - For each pair of rings: the minimum inter-ring distance / 2
    - Boundary distance for each ring
    """
    n_rings = len(ring_counts)
    min_r = np.inf
    
    # Intra-ring constraints
    for i in range(n_rings):
        r = radii[i]
        n = ring_counts[i]
        if n == 0:
            continue
        if n == 1:
            # Single circle: must fit inside container
            min_r = min(min_r, 1.0 - r)
        else:
            # Adjacent distance constraint
            adj_r = r * np.sin(np.pi / n) / 2.0
            min_r = min(min_r, adj_r)
            # Also boundary
            min_r = min(min_r, 1.0 - r)
    
    # Inter-ring constraints (approximate)
    for i in range(n_rings):
        for j in range(i+1, n_rings):
            if ring_counts[i] == 0 or ring_counts[j] == 0:
                continue
            ri = radii[i]
            rj = radii[j]
            ni = ring_counts[i]
            nj = ring_counts[j]
            
            # Compute minimum distance between rings
            if ni == 1 and nj == 1:
                # Both single points
                # Distance between two points at (ri, phase_i) and (rj, phase_j)
                di = np.sqrt(ri**2 + rj**2 - 2*ri*rj*np.cos(phases[i] - phases[j]))
                min_r = min(min_r, di / 2.0)
            elif ni == 1:
                # One single point, one ring
                # Minimum distance from point to any point on ring j
                # This is min over angles of sqrt(ri^2 + rj^2 - 2*ri*rj*cos(angle_diff))
                # The minimum occurs when angle_diff = 0, giving |ri - rj|
                # But we need to check if there's a circle on ring j at the aligned angle
                # For now, use |ri - rj| as lower bound
                min_r = min(min_r, abs(ri - rj) / 2.0)
            elif nj == 1:
                min_r = min(min_r, abs(ri - rj) / 2.0)
            else:
                # Both rings have multiple circles
                # The minimum distance is approximately |ri - rj| if they can align
                # But with different counts, they may not perfectly align
                # Conservative estimate: |ri - rj|
                min_r = min(min_r, abs(ri - rj) / 2.0)
    
    # Ensure non-negative
    return max(0.0, min_r)


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
    
    r = _packing_radius_from_params(radii, phases, ring_counts)
    return -r


# Define ring structure: counts per ring summing to 300
# Use a progression that balances ring sizes
# 1 + 6 + 12 + 18 + 24 + 30 + 36 + 42 + 48 + 54 + 29 = 300
ring_counts = np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29])
num_rings = len(ring_counts)


def _initial_guess():
    """Generate initial guess for radii and phases."""
    # Radii: start with evenly spaced, but ensure first ring is at center
    radii = np.zeros(num_rings)
    radii[0] = 0.0  # Center
    if num_rings > 1:
        # Spread remaining rings from small to near boundary
        radii[1:] = np.linspace(0.15, 0.92, num_rings - 1)
    
    # Phases: stagger alternate rings for better packing
    phases = np.zeros(num_rings)
    for i in range(1, num_rings):
        if i % 2 == 0:
            phases[i] = np.pi / ring_counts[i] if ring_counts[i] > 1 else 0.0
    
    return np.concatenate([radii, phases])


def construct():
    """Construct the packing configuration."""
    
    # Initial guess
    x0 = _initial_guess()
    
    # Bounds: radii in [0, 1], phases in [0, 2*pi]
    bounds = [(0.0, 1.0)] * num_rings + [(0.0, 2 * np.pi)] * num_rings
    
    # Use Nelder-Mead for robustness with non-smooth objective
    result = minimize(
        _objective, x0, args=(ring_counts,),
        method='Nelder-Mead',
        options={'maxiter': 15000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True, 'maxfev': 30000}
    )
    
    best_params = result.x
    radii = np.maximum(best_params[:num_rings], 0.0)
    phases = best_params[num_rings:]
    
    points = _build_points(radii, phases, ring_counts)
    
    # Verify it's valid
    if len(points) != N:
        # Fallback: generate a valid packing using hexagonal lattice
        return _fallback_hexagonal()
    
    # Verify all points are inside the unit circle
    norms = np.hypot(points[:, 0], points[:, 1])
    if np.any(norms > 1.0 + 1e-9):
        return _fallback_hexagonal()
    
    return points


def _fallback_hexagonal():
    """Fallback: generate a valid packing using hexagonal lattice."""
    # Start with a reasonable spacing and adjust
    best_pts = None
    best_r = 0.0
    
    for spacing in np.linspace(0.05, 0.08, 60):
        # Generate hexagonal lattice points
        k = int(1.0 / spacing) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        
        pts = np.array(rows)
        radius = np.hypot(pts[:, 0], pts[:, 1])
        
        # Keep points that fit with margin
        margin = spacing / 2.0
        mask = radius <= 1.0 - margin
        pts = pts[mask]
        
        if len(pts) >= N:
            # Select N points closest to center
            idx = np.argsort(radius, kind='stable')[:N]
            pts = pts[idx]
            
            # Compute the actual packing radius
            if len(pts) == 1:
                r = 1.0 - np.hypot(*pts[0])
            else:
                from scipy.spatial.distance import cdist
                dist_matrix = cdist(pts, pts)
                np.fill_diagonal(dist_matrix, np.inf)
                min_pair = np.min(dist_matrix) / 2.0
                min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
                r = min(min_pair, min_boundary)
            
            if r > best_r:
                best_r = r
                best_pts = pts
    
    if best_pts is not None:
        return best_pts
    
    # Ultimate fallback: grid
    n = int(np.sqrt(N))
    spacing = 2.0 / (n + 1)
    xs, ys = np.meshgrid(np.linspace(-1 + spacing, 1 - spacing, n),
                          np.linspace(-1 + spacing, 1 - spacing, n))
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    # Keep only those inside
    radius = np.hypot(pts[:, 0], pts[:, 1])
    mask = radius <= 1.0 - spacing / 2
    pts = pts[mask]
    if len(pts) >= N:
        return pts[:N]
    
    # If still not enough, return a simple ring
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    return np.column_stack([0.5*np.cos(angles), 0.5*np.sin(angles)])

# EVOLVE-BLOCK-END
