"""Radial-ring decomposition for packing 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time


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


def _local_search(points, ring_counts, radii, phases, iterations=500, rng=None):
    """
    Perform a local search by perturbing radii and phases.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    current_r = _compute_radius(points)
    best_r = current_r
    best_radii = radii.copy()
    best_phases = phases.copy()
    best_points = points.copy()
    
    k = len(ring_counts)
    step_sizes = np.ones(k) * 0.01
    
    for _ in range(iterations):
        # Pick a random ring to perturb
        i = rng.integers(0, k)
        if ring_counts[i] == 0:
            continue
            
        # Perturb radius
        dr = rng.normal(0, step_sizes[i])
        new_radii = radii.copy()
        new_radii[i] += dr
        
        # Ensure radii are ordered and positive
        valid = True
        for j in range(k):
            if new_radii[j] < 0:
                valid = False
                break
            if j > 0 and new_radii[j] < new_radii[j-1]:
                valid = False
                break
        
        if not valid:
            continue
            
        # Perturb phase
        dphi = rng.normal(0, 0.1)
        new_phases = phases.copy()
        new_phases[i] += dphi
        
        new_points = _build_points(new_radii, new_phases, ring_counts)
        new_r = _compute_radius(new_points)
        
        if new_r > best_r:
            best_r = new_r
            best_radii = new_radii
            best_phases = new_phases
            best_points = new_points
            radii = new_radii
            phases = new_phases
            # Reduce step size slightly on success
            step_sizes[i] *= 0.9
        else:
            # Increase step size slightly on failure (simulated annealing style)
            step_sizes[i] *= 1.05
            if step_sizes[i] > 0.1:
                step_sizes[i] = 0.1

    return best_points, best_r


def _generate_ring_candidates():
    """Generate diverse ring count candidates."""
    candidates = []
    
    # Base hexagonal
    base = [1, 6, 12, 18, 24, 30, 36, 42, 48, 54]
    sum_base = sum(base)
    remaining = N - sum_base # 29
    
    # 1. Append remaining
    candidates.append(np.array(base + [remaining]))
    
    # 2. Distribute remaining among last few rings
    for split1 in range(0, remaining + 1, 3):
        for split2 in range(0, remaining - split1 + 1, 3):
            c1 = remaining - split1 - split2
            c2 = split1
            c3 = split2
            if c1 < 0 or c2 < 0 or c3 < 0:
                continue
            # Merge with last rings
            new_counts = base[:-2] + [base[-2] + c3, base[-1] + c2, c1]
            # Clean up zeros and ensure sum is N
            new_counts = [c for c in new_counts if c > 0]
            if sum(new_counts) == N:
                candidates.append(np.array(new_counts))
                
    # 3. Compress inner rings?
    # Try reducing ring 1, 2, 3 counts to allow more outer rings?
    # This is complex. Let's stick to variations of the outer part.
    
    # 4. Specific known good structures for N=300?
    # Try shifting counts from ring 9 to ring 10, 11
    for shift in range(0, 20, 2):
        if shift == 0: continue
        new_counts = base.copy()
        new_counts[-1] = remaining + shift
        new_counts[-2] = 54 - shift
        if new_counts[-2] > 0:
             candidates.append(np.array(new_counts))
             
    # Remove duplicates
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
            
    return unique_candidates


def construct():
    """Construct the packing configuration."""
    
    best_points = None
    best_r = 0.0
    start_time = time.time()
    timeout = 110.0 # Leave 10s buffer
    
    candidates = _generate_ring_candidates()
    
    # Shuffle candidates to explore different structures early
    rng_global = np.random.default_rng(42)
    rng_global.shuffle(candidates)
    
    for ring_counts in candidates:
        if time.time() - start_time > timeout:
            break
            
        if sum(ring_counts) != N:
            continue
            
        num_rings = len(ring_counts)
        
        # Initialize
        # Estimate radii based on packing density
        # r_target ~ 0.053
        # R_i ~ r_target / sin(pi/n_i)
        est_radii = np.zeros(num_rings)
        for i, n in enumerate(ring_counts):
            if n > 1:
                est_radii[i] = 0.053 / np.sin(np.pi / n)
            else:
                est_radii[i] = 0.0
                
        # Ensure increasing
        for i in range(1, num_rings):
            if est_radii[i] <= est_radii[i-1]:
                est_radii[i] = est_radii[i-1] + 0.1
                
        # Scale to fit
        if est_radii[-1] + 0.053 > 1.0:
            scale = (1.0 - 0.053) / est_radii[-1]
            est_radii *= scale
            
        # Staggered phases
        phases = np.zeros(num_rings)
        for i in range(1, num_rings):
            if ring_counts[i] > 0 and ring_counts[i-1] > 0:
                phases[i] = np.pi / ring_counts[i-1]
        
        # Try a few local searches with different random seeds
        for seed in range(3):
            if time.time() - start_time > timeout:
                break
                
            rng = np.random.default_rng(seed + hash(tuple(ring_counts)) % 1000)
            
            # Add small jitter to initial radii and phases
            jitter_radii = est_radii + rng.normal(0, 0.01, num_rings)
            jitter_phases = phases + rng.normal(0, 0.1, num_rings)
            
            # Ensure radii order
            for i in range(1, num_rings):
                if jitter_radii[i] < jitter_radii[i-1]:
                    jitter_radii[i] = jitter_radii[i-1] + 0.01
                    
            if np.any(jitter_radii < 0):
                continue
                
            points = _build_points(jitter_radii, jitter_phases, ring_counts)
            
            if len(points) != N:
                continue
                
            # Run local search
            new_points, new_r = _local_search(points, ring_counts, jitter_radii, jitter_phases, iterations=200, rng=rng)
            
            if new_r > best_r:
                best_r = new_r
                best_points = new_points
    
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
