"""Radial-ring decomposition for packing 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar


def _build_ring_configuration(rings):
    """
    Given a list of (radius, count) pairs, build the point array.
    For each ring, place 'count' points evenly spaced in angle, with an offset
    to maximize minimum angular separation (which is always 2*pi/count for even spacing).
    We can rotate each ring independently to potentially improve inter-ring distances,
    but for simplicity and symmetry we use fixed offsets. Actually, let's just use
    even spacing with a small phase shift per ring to break symmetry if needed.
    For now, even spacing with phase = 0 for all rings, or alternating.
    """
    points = []
    for i, (r, count) in enumerate(rings):
        if count == 0:
            continue
        # Evenly spaced angles
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        # Add a small phase offset for alternating rings to potentially improve packing
        # Actually, let's just use phase = 0 for simplicity first
        phase = 0.0
        xs = r * np.cos(angles + phase)
        ys = r * np.sin(angles + phase)
        points.extend(zip(xs, ys))
    
    if not points:
        return np.zeros((0, 2))
    
    return np.array(points)


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
    # For N=300, this is O(N^2) = 90000 pairs, which is fine
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(points, points)
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = np.min(dist_matrix) / 2.0
    
    return min(min_boundary, min_pair)


def _score_ring_configuration(ring_params):
    """
    ring_params: array of radial distances for each ring.
    The number of points in each ring is fixed.
    Returns the negative of the computed radius (for minimization).
    """
    rings = []
    total = 0
    for i, (r, count) in enumerate(zip(ring_params, ring_counts)):
        if count > 0:
            rings.append((r, count))
            total += count
    
    if total != N:
        return 1e10  # Invalid
    
    points = _build_ring_configuration(rings)
    r = _compute_radius(points)
    return -r


# Define ring structure: number of rings and points per ring
# We'll use a heuristic: start with 1 center, then rings with increasing counts
# Typical ring structures for circular packings: 1, 6, 12, 18, 24, ... (hexagonal-like)
# But for optimal packing, the counts per ring may vary.

# Let's define a reasonable ring structure
# For N=300, let's try:
# Ring 0: 1 point (center)
# Ring 1: 6 points
# Ring 2: 12 points
# Ring 3: 18 points
# Ring 4: 24 points
# Ring 5: 30 points
# Ring 6: 36 points
# Ring 7: 42 points
# Ring 8: 48 points
# Ring 9: 54 points
# Ring 10: 60 points
# Sum: 1+6+12+18+24+30+36+42+48+54+60 = 331, too many.

# Let's try:
# Ring 0: 1
# Ring 1: 6
# Ring 2: 12
# Ring 3: 18
# Ring 4: 24
# Ring 5: 30
# Ring 6: 36
# Ring 7: 42
# Ring 8: 48
# Ring 9: 54
# Sum so far: 1+6+12+18+24+30+36+42+48+54 = 271
# Need 29 more. Let's add a 10th ring with 29 points? That's awkward.

# Alternative: adjust the counts to sum to exactly 300.
# Let's use a more flexible approach.

# Ring counts that sum to 300:
# Let's try: 1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29
# Sum: 1+6+12+18+24+30+36+42+48+54+29 = 300. Yes!

ring_counts = [1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29]
num_rings = len(ring_counts)

# Initial guess for radial positions
# For ring i with count c_i, the minimum radius to fit c_i points on a circle of radius r
# such that adjacent points are at least 2*radius apart is:
# 2*r*sin(pi/c_i) >= 2*radius => r >= radius/sin(pi/c_i)
# Also, r must be <= 1 - radius.
# Let's start with r_i = (i+1) * 0.05 as a rough guess, then optimize.

r0 = np.array([0.0] + [0.05 * (i + 1) for i in range(num_rings - 1)])


def construct():
    """Construct the packing configuration."""
    
    # Optimize radial positions
    # Bounds: each radius must be >= 0 and <= 1
    # Also, for ring i, r_i must be large enough to accommodate the points
    # and small enough that points are inside the unit circle.
    
    # Let's do a simple coordinate descent or use scipy.optimize.minimize
    # with bounds.
    
    from scipy.optimize import minimize
    
    def objective(params):
        rings = []
        for i, (r, count) in enumerate(zip(params, ring_counts)):
            if count > 0:
                rings.append((r, count))
        points = _build_ring_configuration(rings)
        r = _compute_radius(points)
        return -r
    
    # Bounds: r_i in [0, 1]
    bounds = [(0.0, 1.0)] * num_rings
    
    # Initial guess
    x0 = r0.copy()
    
    # Use Nelder-Mead or L-BFGS-B
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
    
    best_params = result.x
    
    # Build final configuration
    rings = []
    for i, (r, count) in enumerate(zip(best_params, ring_counts)):
        if count > 0:
            rings.append((r, count))
    
    points = _build_ring_configuration(rings)
    
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
