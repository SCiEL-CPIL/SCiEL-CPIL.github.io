"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial.distance import pdist

N = 300

# EVOLVE-BLOCK-START

def _compute_radius(points):
    """
    Compute the feasible common radius for a set of points.
    Radius is limited by:
    1. Distance to boundary: min(1 - |p_i|)
    2. Distance to other circles: min(|p_i - p_j|) / 2
    """
    if points is None or len(points) == 0:
        return 0.0
    
    # 1. Boundary constraint
    dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = float(np.min(dists_to_boundary))
    
    # 2. Pairwise distance constraint
    # pdist computes condensed pairwise distances
    d = pdist(points)
    if len(d) == 0:
        min_pair = np.inf
    else:
        min_pair = float(np.min(d)) / 2.0
        
    return min(min_boundary, min_pair)

def _hexagonal_lattice(spacing):
    """
    Generate hexagonal lattice points inside the unit disk.
    Returns None if not enough points are generated.
    """
    k = int(1.0 / spacing) + 3
    pts = []
    sqrt3 = np.sqrt(3.0)
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * sqrt3 / 2.0
            # Ensure points are strictly inside with some margin for safety in initial construction
            # The refinement step will handle the exact boundary
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                pts.append((x, y))
    
    if len(pts) < N:
        return None
        
    pts = np.array(pts)
    # Sort by distance from origin to pick the central cluster
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]

def _square_grid(spacing):
    """
    Generate square grid points inside the unit disk.
    """
    k = int(1.0 / spacing) + 3
    pts = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = i * spacing
            y = j * spacing
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                pts.append((x, y))
                
    if len(pts) < N:
        return None
        
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]

def _concentric_rings(n_rings):
    """
    Generate points on concentric rings.
    """
    if n_rings <= 0:
        return None
        
    pts = []
    # Center point
    pts.append([0.0, 0.0])
    
    remaining = N - 1
    if n_rings == 1:
        # Just one ring? Actually if n_rings==1, we put all in one ring?
        # The logic below assumes ring 0 is center.
        # Let's adjust: if n_rings is small, distribute evenly.
        pass
        
    ring_counts = [1] # Center
    if n_rings > 1:
        # Distribute remaining N-1 points among n_rings-1 rings
        # Outer rings can hold more points
        # Simple heuristic: proportional to radius
        radii = np.linspace(0.3, 0.95, n_rings - 1)
        weights = radii
        weights /= np.sum(weights)
        counts = np.round(weights * remaining).astype(int)
        
        # Adjust for rounding errors
        diff = remaining - np.sum(counts)
        if diff != 0:
            # Add/subtract from the ring with largest count (usually outer)
            idx_max = np.argmax(counts)
            counts[idx_max] += diff
            
        ring_counts.extend(counts.tolist())
        
    for ring_idx in range(1, n_rings):
        count = ring_counts[ring_idx]
        if count <= 0:
            continue
        ring_radius = 0.95 * ring_idx / n_rings
        for i in range(count):
            angle = 2 * np.pi * i / count
            x = ring_radius * np.cos(angle)
            y = ring_radius * np.sin(angle)
            pts.append([x, y])
            
    if len(pts) != N:
        return None
        
    return np.array(pts)

def _refine_local(points, deadline, max_iters=5000, step_size=0.05):
    """
    Hill-climbing local refinement.
    Tries to move individual points to increase the minimum radius.
    """
    points = points.copy()
    best_r = _compute_radius(points)
    best_points = points.copy()
    
    rng = np.random.default_rng(42)
    
    # Pre-compute some constants for speed if needed, but pdist is fast enough for N=300
    # N=300 -> ~45k pairs. pdist is vectorized C.
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        # Pick a random point
        idx = rng.integers(0, N)
        orig_x, orig_y = points[idx]
        
        # Try a few random perturbations
        best_local_r = best_r
        best_local_pt = (orig_x, orig_y)
        
        for _ in range(10): # 10 attempts per iteration
            if time.time() > deadline:
                break
            # Random step
            angle = rng.uniform(0, 2 * np.pi)
            # Step size can be adaptive, but fixed small step is safe
            step = step_size * rng.uniform(0.1, 1.0)
            dx = step * np.cos(angle)
            dy = step * np.sin(angle)
            
            new_x = orig_x + dx
            new_y = orig_y + dy
            
            # Quick check: is it inside the unit circle?
            # If not, the radius will be negative or very small, but we can skip
            # to save pdist calls if it's clearly bad.
            if new_x*new_x + new_y*new_y > 1.0:
                continue
                
            points[idx] = [new_x, new_y]
            r = _compute_radius(points)
            
            if r > best_local_r:
                best_local_r = r
                best_local_pt = (new_x, new_y)
                
        # Commit the best local move
        points[idx] = best_local_pt
        
        if best_local_r > best_r:
            best_r = best_local_r
            best_points = points.copy()
            
    return best_points, best_r

def construct():
    start_time = time.time()
    # Safety margin: 150s limit. Let's use 140s to be safe.
    deadline = start_time + 140.0
    
    candidates = []
    
    # 1. Hexagonal Lattice Candidates
    # Optimal spacing for hex packing in a circle is roughly related to N.
    # Area of circle ~ 1. Area of N circles ~ N * pi * r^2.
    # r ~ 1/sqrt(N) ~ 0.057.
    # Spacing d = 2r ~ 0.114.
    # Let's try a range of spacings.
    spacings = np.linspace(0.10, 0.13, 20)
    for s in spacings:
        pts = _hexagonal_lattice(s)
        if pts is not None:
            r = _compute_radius(pts)
            candidates.append((pts, r))
            
    # 2. Square Grid Candidates
    spacings_sq = np.linspace(0.10, 0.13, 10)
    for s in spacings_sq:
        pts = _square_grid(s)
        if pts is not None:
            r = _compute_radius(pts)
            candidates.append((pts, r))
            
    # 3. Concentric Rings Candidates
    for n_rings in range(5, 20):
        pts = _concentric_rings(n_rings)
        if pts is not None:
            r = _compute_radius(pts)
            candidates.append((pts, r))
            
    if not candidates:
        # Fallback: random points (very unlikely to be called)
        pts = np.random.randn(N, 2) * 0.5
        return pts

    # Sort by radius, best first
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    best_points = candidates[0][0]
    best_r = candidates[0][1]
    
    # 4. Refinement
    # Refine the top 3 candidates
    num_to_refine = min(3, len(candidates))
    for i in range(num_to_refine):
        if time.time() > deadline - 5.0:
            break
            
        cand_pts, cand_r = candidates[i]
        # Allocate remaining time proportionally or just run until deadline
        # We pass the global deadline to the refiner
        refined_pts, refined_r = _refine_local(cand_pts, deadline, max_iters=10000, step_size=0.05)
        
        if refined_r > best_r:
            best_r = refined_r
            best_points = refined_pts
            
    return best_points

# EVOLVE-BLOCK-END
