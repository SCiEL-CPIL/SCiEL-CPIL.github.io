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


def _ring_constraint_radius(ring_counts):
    """
    For a given ring structure, compute the maximum achievable radius
    by solving for ring radii such that:
    1. Each ring's circumference can fit its circles: 2*pi*r_i >= n_i * 2*r
       => r_i >= n_i * r / pi
    2. Adjacent rings don't overlap: r_{i+1} - r_i >= 2*r (approx, for tangent rings)
       More precisely, the minimum distance between circles on adjacent rings >= 2*r
    3. Outermost ring: r_{k-1} + r <= 1
    
    This gives a lower bound on what's achievable. We use this to guide
    initial guesses and to validate.
    """
    k = len(ring_counts)
    # For ring i with n_i circles at radius r_i:
    #   chord distance between adjacent circles = 2 * r_i * sin(pi/n_i)
    #   This must be >= 2*r  =>  r_i >= r / sin(pi/n_i)
    # For adjacent rings i, i+1:
    #   minimum distance between a circle on ring i and ring i+1 >= 2*r
    #   The closest approach is approximately |r_{i+1} - r_i| (radial)
    #   but actually depends on angular alignment. Worst case: radial gap >= 2*r
    #   So: r_{i+1} >= r_i + 2*r
    # Boundary: r_{k-1} + r <= 1
    
    # From ring i constraint: r_i >= r / sin(pi/n_i)
    # From adjacency: r_i >= r_0 + 2*i*r  (for i=0,1,...)
    # Combined: r_i >= max(r / sin(pi/n_i), r_0 + 2*i*r)
    # And r_0 >= r / sin(pi/n_0) if n_0 > 1, or r_0 >= 0 if n_0 == 1 (center)
    
    # Simplest: assume r_0 = 0 if first ring is center (n_0=1)
    # Then r_i >= max(r / sin(pi/n_i), 2*i*r)
    # And r_{k-1} + r <= 1
    
    # So: max over i of [max(r/sin(pi/n_i), 2*i*r)] + r <= 1
    # => r * max_i [max(1/sin(pi/n_i), 2*i)] + r <= 1
    # => r <= 1 / (1 + max_i [max(1/sin(pi/n_i), 2*i)])
    
    max_term = 0.0
    for i, n in enumerate(ring_counts):
        if n > 1:
            term1 = 1.0 / np.sin(np.pi / n)
        else:
            term1 = 0.0
        term2 = 2.0 * i
        max_term = max(max_term, term1, term2)
    
    return 1.0 / (1.0 + max_term)


# Define ring structure: counts per ring summing to 300
# Hexagonal-like rings: 1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29
# Sum: 1+6+12+18+24+30+36+42+48+54+29 = 300
ring_counts = np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29])
num_rings = len(ring_counts)

# Alternative ring structures to try
# More balanced: 1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29 (current)
# Try: 1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29
# Or: 1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29
# Or try a different distribution: 1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29
# Let's try: 1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29

# Compute the theoretical max radius for a given ring structure
def _theoretical_max_radius(ring_counts):
    """
    Compute the theoretical maximum circle radius for a given ring structure.
    Uses the constraint that:
    - Ring i with n_i circles needs radius r_i >= r / sin(pi/n_i) for n_i > 1
    - Adjacent rings need radial gap >= 2*r (worst case)
    - Outermost ring: r_{k-1} + r <= 1
    
    For a ring at radius R with n circles, the minimum distance between adjacent
    circles on the same ring is 2*R*sin(pi/n). For non-overlapping circles of radius r:
    2*R*sin(pi/n) >= 2*r  =>  R >= r/sin(pi/n)
    
    For adjacent rings at radii R_i and R_{i+1}, the minimum distance between circles
    depends on angular alignment. Worst case (aligned): |R_{i+1} - R_i| >= 2*r
    Best case (staggered): the distance can be larger.
    
    We use the worst-case radial constraint for a lower bound estimate.
    """
    k = len(ring_counts)
    # For each ring i, the required radius is at least r / sin(pi/n_i) if n_i > 1
    # And at least 2*i*r from the center (if we start at r_0 = 0 for center)
    
    # Actually, let's think more carefully. If ring 0 is at the center (n_0=1),
    # then r_0 = 0. Ring 1 must be at least 2*r from center (to not overlap center circle).
    # Ring 2 must be at least 2*r from ring 1, so r_2 >= 4*r, etc.
    # But also ring i must satisfy r_i >= r/sin(pi/n_i).
    
    # So r_i >= max(r/sin(pi/n_i), 2*i*r) for i >= 1
    # And r_0 = 0 (center)
    # Constraint: r_{k-1} + r <= 1
    
    # This gives: max_i [max(r/sin(pi/n_i), 2*i*r)] + r <= 1
    # => r * max_i [max(1/sin(pi/n_i), 2*i)] + r <= 1
    # => r <= 1 / (1 + max_i [max(1/sin(pi/n_i), 2*i)])
    
    max_term = 0.0
    for i, n in enumerate(ring_counts):
        if n > 1:
            term1 = 1.0 / np.sin(np.pi / n)
        else:
            term1 = 0.0
        term2 = 2.0 * i
        max_term = max(max_term, term1, term2)
    
    return 1.0 / (1.0 + max_term)


def _initial_guess(ring_counts, jitter_seed=0):
    """Generate initial guess for radii and phases using constraint-based calculation."""
    rng = np.random.default_rng(jitter_seed)
    
    k = len(ring_counts)
    
    # Use the theoretical max radius as target
    r_target = _theoretical_max_radius(ring_counts)
    # Add a small safety margin
    r_target *= 0.98
    
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


def _local_refine(points, max_iter=100, tol=1e-10):
    """
    Local refinement: iteratively push circles apart to increase the minimum distance.
    This is a simple gradient-free approach that works on the point positions directly.
    """
    points = points.copy()
    n = len(points)
    
    for iteration in range(max_iter):
        # Compute all pairwise distances
        dist_matrix = cdist(points, points)
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Find the minimum distance pair
        min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        min_dist = dist_matrix[min_idx]
        
        # Also check boundary distances
        boundary_dists = 1.0 - np.hypot(points[:, 0], points[:, 1])
        min_boundary = np.min(boundary_dists)
        
        # The limiting constraint
        r = min(min_dist / 2.0, min_boundary)
        
        # If we've converged, stop
        if min_dist / 2.0 - min_boundary < tol and min_dist > 0:
            break
        
        # Find all pairs that are close (within some threshold)
        threshold = min_dist * 1.01
        close_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if dist_matrix[i, j] < threshold:
                    close_pairs.append((i, j, dist_matrix[i, j]))
        
        if not close_pairs:
            break
        
        # For each close pair, push them apart slightly
        step_size = 0.001
        for i, j, d in close_pairs:
            if d < 1e-10:
                # Coincident points - add random jitter
                dx = rng_uniform(-step_size, step_size)
                dy = rng_uniform(-step_size, step_size)
                points[i, 0] += dx
                points[i, 1] += dy
                points[j, 0] -= dx
                points[j, 1] -= dy
            else:
                # Push apart along the line connecting them
                dx = points[j, 0] - points[i, 0]
                dy = points[j, 1] - points[i, 1]
                norm = np.hypot(dx, dy)
                if norm > 1e-10:
                    push = step_size / norm
                    points[i, 0] -= dx * push
                    points[i, 1] -= dy * push
                    points[j, 0] += dx * push
                    points[j, 1] += dy * push
        
        # Push circles that are too close to boundary inward
        for i in range(n):
            dist_to_center = np.hypot(points[i, 0], points[i, 1])
            if dist_to_center > 1.0 - r:
                # Push inward
                if dist_to_center > 1e-10:
                    scale = (1.0 - r) / dist_to_center
                    points[i, 0] *= scale
                    points[i, 1] *= scale
    
    return points


def rng_uniform(low, high):
    """Simple random uniform in [low, high] without needing a global RNG."""
    return low + (high - low) * np.random.random()


def construct():
    """Construct the packing configuration."""
    
    best_points = None
    best_r = 0.0
    
    # Try multiple initial guesses with different jitters
    for seed in range(8):
        x0 = _initial_guess(ring_counts, jitter_seed=seed)
        
        # Bounds: radii in [0, 1], phases in [0, 2*pi]
        bounds = [(0.0, 1.0)] * num_rings + [(0.0, 2 * np.pi)] * num_rings
        
        # Use Nelder-Mead for robustness with non-smooth objective
        result = minimize(
            _objective, x0, args=(ring_counts,),
            method='Nelder-Mead',
            options={'maxiter': 8000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
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
    
    # Local refinement on the best configuration
    if best_points is not None:
        refined = _local_refine(best_points, max_iter=200, tol=1e-12)
        r_refined = _compute_radius(refined)
        if r_refined > best_r:
            best_r = r_refined
            best_points = refined
    
    # Fallback if no valid configuration found
    if best_points is None:
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
    
    return best_points

# EVOLVE-BLOCK-END
