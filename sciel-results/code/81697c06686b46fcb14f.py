"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

def _compute_radius(points):
    """Compute the feasible radius for a set of points."""
    if points is None or len(points) == 0:
        return 0.0
    # Distance to boundary
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dist_boundary)
    
    # Pairwise distances
    d = pdist(points)
    min_dist = np.min(d)
    min_pair = min_dist / 2.0
    
    return min(min_boundary, min_pair)

def _construct_concentric_rings():
    """Construct a concentric rings packing."""
    best_r = 0.0
    best_points = None
    
    # Try different ring configurations
    for n_rings in range(1, 20):
        # Distribute points among rings
        # Ring 0 (center) can have 1 point, others have proportional counts
        if n_rings == 1:
            ring_counts = [N]
        else:
            # First ring has 1 point (center), rest distributed
            remaining = N - 1
            # Distribute remaining points among n_rings-1 rings
            base = remaining // (n_rings - 1)
            extra = remaining % (n_rings - 1)
            ring_counts = [1] + [base + (1 if i < extra else 0) for i in range(n_rings - 1)]
        
        points = []
        
        # Center point
        if ring_counts[0] > 0:
            points.append([0.0, 0.0])
        
        for ring_idx in range(1, n_rings):
            count = ring_counts[ring_idx]
            if count <= 0:
                continue
            
            # Radius for this ring - optimize later
            # Start with uniform spacing
            ring_radius = ring_idx / n_rings * 0.95
            
            for i in range(count):
                angle = 2 * np.pi * i / count
                x = ring_radius * np.cos(angle)
                y = ring_radius * np.sin(angle)
                points.append([x, y])
        
        if len(points) != N:
            continue
        
        points = np.array(points)
        r = _compute_radius(points)
        
        if r > best_r:
            best_r = r
            best_points = points.copy()
    
    return best_points, best_r

def _construct_hexagonal():
    """Construct a hexagonal lattice packing."""
    best_r = 0.0
    best_points = None
    
    # Try different spacings
    for trial in range(50):
        spacing = 0.1 + trial * 0.01
        
        # Generate hexagonal grid points
        points = []
        k = int(1.0 / spacing) + 3
        
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    points.append([x, y])
        
        if len(points) < N:
            continue
        
        points = np.array(points)
        
        # Select N points closest to center
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists, kind='stable')[:N]
        points = points[idx]
        
        r = _compute_radius(points)
        
        if r > best_r:
            best_r = r
            best_points = points.copy()
    
    return best_points, best_r

def _construct_grid():
    """Construct a square grid packing."""
    best_r = 0.0
    best_points = None
    
    for trial in range(50):
        spacing = 0.1 + trial * 0.01
        
        points = []
        k = int(1.0 / spacing) + 3
        
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = i * spacing
                y = j * spacing
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    points.append([x, y])
        
        if len(points) < N:
            continue
        
        points = np.array(points)
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists, kind='stable')[:N]
        points = points[idx]
        
        r = _compute_radius(points)
        
        if r > best_r:
            best_r = r
            best_points = points.copy()
    
    return best_points, best_r

def _refine_local(points, max_iters=1000, step_size=0.01):
    """Local refinement using gradient-free optimization."""
    if points is None:
        return None, 0.0
    
    best_points = points.copy()
    best_r = _compute_radius(points)
    
    # Simple hill climbing: try small random perturbations
    np.random.seed(42)
    
    for iteration in range(max_iters):
        # Pick a random point and perturb it
        idx = np.random.randint(len(points))
        original = points[idx].copy()
        
        # Try several perturbations
        for _ in range(10):
            dx = np.random.uniform(-step_size, step_size)
            dy = np.random.uniform(-step_size, step_size)
            points[idx] = original + [dx, dy]
            
            r = _compute_radius(points)
            
            if r > best_r:
                best_r = r
                best_points = points.copy()
            else:
                points[idx] = original
    
    return best_points, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Generate initial candidates from different constructions
    candidates = []
    
    # Hexagonal lattice
    pts, r = _construct_hexagonal()
    if pts is not None:
        candidates.append((pts, r))
    
    # Concentric rings
    pts, r = _construct_concentric_rings()
    if pts is not None:
        candidates.append((pts, r))
    
    # Grid
    pts, r = _construct_grid()
    if pts is not None:
        candidates.append((pts, r))
    
    if not candidates:
        # Fallback: simple ring
        points = []
        for i in range(N):
            angle = 2 * np.pi * i / N
            points.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
        return np.array(points)
    
    # Select best initial candidate
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_points, best_r = candidates[0]
    
    # Refine the best candidate
    if time.time() < deadline - 20.0:
        refined_pts, refined_r = _refine_local(best_points, max_iters=500, step_size=0.02)
        if refined_pts is not None and refined_r > best_r:
            best_points, best_r = refined_pts, refined_r
    
    # Try refining second-best if time permits
    if len(candidates) > 1 and time.time() < deadline - 10.0:
        second_pts, second_r = candidates[1]
        refined_pts, refined_r = _refine_local(second_pts, max_iters=300, step_size=0.02)
        if refined_pts is not None and refined_r > best_r:
            best_points, best_r = refined_pts, refined_r
    
    return best_points
# EVOLVE-BLOCK-END
