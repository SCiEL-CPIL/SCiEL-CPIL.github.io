"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution


def _evaluate_ring_config(rings, n_circles):
    """
    rings: list of (radius, count) tuples, ordered from center outward.
    Returns the maximum possible common circle radius for this ring configuration.
    """
    if len(rings) == 0:
        return 0.0
    
    # Sort by radius
    rings = sorted(rings, key=lambda x: x[0])
    
    r_max = np.inf
    
    # For each ring, the circles on that ring must not overlap with each other
    for ring_idx, (ring_r, count) in enumerate(rings):
        if count == 1:
            # Single circle on this ring: only constraint is distance to other rings
            continue
        # Angular spacing on this ring
        angular_spacing = 2.0 * np.pi / count
        # Chord length between adjacent circles on this ring
        chord = 2.0 * ring_r * np.sin(angular_spacing / 2.0)
        r_max = min(r_max, chord / 2.0)
    
    # Constraint: circles must be inside the container
    # For each ring, the outermost point of a circle is ring_r + r
    for ring_r, count in rings:
        r_max = min(r_max, 1.0 - ring_r)
    
    # Constraint: circles on adjacent rings must not overlap
    # For two rings with radii r1 < r2 and counts c1, c2,
    # we need to check the minimum distance between any circle on ring 1 and any on ring 2.
    # The worst case depends on angular alignment. To be safe, we consider the worst-case
    # angular offset. Actually, for a given configuration, we can optimize angular offsets.
    # But for a quick estimate, let's use a conservative bound.
    # A better approach: for each pair of adjacent rings, the minimum distance between
    # circles is determined by the radial difference and angular offset.
    # Let's compute the exact minimum distance for given angular offsets.
    # Since we can choose angular offsets, we want to maximize the minimum distance.
    # For two rings with radii r1, r2 and counts c1, c2, the optimal angular offset
    # depends on c1 and c2. This is complex. Let's use a simpler constraint:
    # The minimum distance between a circle on ring i and ring j (i < j) is at least
    # |r_j - r_i| if the angular positions align radially, but can be larger with offset.
    # A conservative lower bound on the min distance is:
    #   min over all pairs of circles on different rings of distance
    # Let's compute this properly by iterating over all pairs of circles.
    # But that's O(N^2). For N=300, that's 90000 pairs, which is fine.
    
    # Actually, let's restructure: we'll place circles and compute the actual min distance.
    # But this function is called many times during optimization, so it needs to be fast.
    # Let's use the fact that for concentric rings, the minimum distance between circles
    # on different rings can be computed analytically.
    
    # For now, let's just return the constraint from same-ring and container.
    # The cross-ring constraint will be handled by the optimizer checking validity.
    
    return r_max


def _build_config_from_params(params, n_circles):
    """
    params: array of parameters defining the ring configuration.
    We use: [r0, c0, r1, c1, ..., rk, ck] where r_i is ring radius, c_i is count.
    The first ring (center) has radius 0 if c0 > 0.
    """
    rings = []
    i = 0
    while i + 1 < len(params):
        r = params[i]
        c = int(round(params[i + 1]))
        if c > 0 and r >= 0:
            rings.append((r, c))
        i += 2
    return rings


def _total_circles(rings):
    return sum(c for _, c in rings)


def construct():
    """
    Place N equal circles in the unit disk using a radial-ring decomposition.
    """
    n = N
    
    # Strategy: use a center circle plus concentric rings.
    # We'll optimize the ring radii and counts.
    
    # Let's start with a heuristic: estimate the number of rings and circles per ring.
    # For a hexagonal packing, the number of circles in a ring of radius r is roughly
    # proportional to r. Let's use that as a starting point.
    
    # Try different numbers of rings and optimize.
    
    best_r = 0.0
    best_points = None
    
    # We'll try configurations with 1 to 15 rings (including center).
    # For each, we'll optimize the ring radii.
    
    for n_rings in range(1, 16):
        # Decide how many circles in each ring.
        # Center ring: 1 circle if n_rings > 0.
        # Remaining rings: distribute n - 1 circles.
        
        if n_rings == 1:
            # All circles in one ring (or center)
            if n == 1:
                return np.array([[0.0, 0.0]])
            # All on one ring: not good for n=300.
            continue
        
        # Center: 1 circle
        remaining = n - 1
        n_outer = n_rings - 1
        
        # Distribute remaining circles among outer rings.
        # Use proportional distribution based on expected circumference.
        # Let's try a simple distribution: more circles in outer rings.
        
        # We'll optimize the counts as well, but start with a heuristic.
        # For now, use a fixed distribution and optimize radii.
        
        # Let's try several count distributions.
        for count_dist in range(3):
            if count_dist == 0:
                # Equal distribution
                counts = [remaining // n_outer] * n_outer
                counts[-1] += remaining - sum(counts)
            elif count_dist == 1:
                # Linear increase
                weights = np.arange(1, n_outer + 1, dtype=float)
                weights /= weights.sum()
                counts = np.round(weights * remaining).astype(int)
                diff = remaining - counts.sum()
                counts[-1] += diff
            else:
                # Quadratic increase
                weights = np.arange(1, n_outer + 1, dtype=float) ** 2
                weights /= weights.sum()
                counts = np.round(weights * remaining).astype(int)
                diff = remaining - counts.sum()
                counts[-1] += diff
            
            if any(c < 0 for c in counts):
                continue
            
            # Now optimize ring radii.
            # Parameters: r1, r2, ..., r_{n_outer} (center is at 0)
            
            def objective(radii):
                rings = [(0.0, 1)] + [(radii[i], counts[i]) for i in range(n_outer)]
                total = _total_circles(rings)
                if total != n:
                    return 1e10
                
                # Check that radii are sorted and positive
                for i in range(len(radii)):
                    if radii[i] < 0:
                        return 1e10
                    if i > 0 and radii[i] <= radii[i-1]:
                        return 1e10
                
                # Compute max radius
                r_max = np.inf
                
                # Same-ring constraints
                for ring_r, count in rings:
                    if count > 1:
                        angular_spacing = 2.0 * np.pi / count
                        chord = 2.0 * ring_r * np.sin(angular_spacing / 2.0)
                        r_max = min(r_max, chord / 2.0)
                    # Container constraint
                    r_max = min(r_max, 1.0 - ring_r)
                
                # Cross-ring constraints: compute actual min distance
                # Place circles and compute min pairwise distance
                points = []
                for ring_idx, (ring_r, count) in enumerate(rings):
                    if ring_r == 0 and count == 1:
                        points.append([0.0, 0.0])
                    else:
                        # Optimal angular offset: stagger rings
                        offset = (ring_idx % 2) * np.pi / count if count > 1 else 0.0
                        for k in range(count):
                            angle = offset + 2.0 * np.pi * k / count
                            points.append([ring_r * np.cos(angle), ring_r * np.sin(angle)])
                
                points = np.array(points)
                if len(points) != n:
                    return 1e10
                
                # Compute min pairwise distance
                min_dist = np.inf
                for i in range(len(points)):
                    for j in range(i+1, len(points)):
                        d = np.linalg.norm(points[i] - points[j])
                        if d < min_dist:
                            min_dist = d
                
                r_max = min(r_max, min_dist / 2.0)
                
                if r_max <= 0:
                    return 1e10
                
                return -r_max  # Minimize negative radius = maximize radius
            
            # Bounds for radii: 0 < r1 < r2 < ... < 1
            bounds = [(0.01, 0.95)] * n_outer
            
            # Initial guess: evenly spaced
            r_init = np.linspace(0.1, 0.9, n_outer)
            
            try:
                result = differential_evolution(
                    objective, bounds, seed=42, maxiter=50, popsize=10,
                    tol=1e-4, polish=True
                )
                if result.fun < 1e9:
                    r_candidate = -result.fun
                    if r_candidate > best_r:
                        # Reconstruct points
                        radii = result.x
                        rings = [(0.0, 1)] + [(radii[i], counts[i]) for i in range(n_outer)]
                        points = []
                        for ring_idx, (ring_r, count) in enumerate(rings):
                            if ring_r == 0 and count == 1:
                                points.append([0.0, 0.0])
                            else:
                                offset = (ring_idx % 2) * np.pi / count if count > 1 else 0.0
                                for k in range(count):
                                    angle = offset + 2.0 * np.pi * k / count
                                    points.append([ring_r * np.cos(angle), ring_r * np.sin(angle)])
                        points = np.array(points)
                        if len(points) == n:
                            best_r = r_candidate
                            best_points = points
            except Exception:
                pass
    
    if best_points is not None:
        return best_points
    
    # Fallback: hexagonal lattice
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= n:
            return points[np.argsort(radius, kind="stable")[:n]]
        spacing *= 0.98

# EVOLVE-BLOCK-END
