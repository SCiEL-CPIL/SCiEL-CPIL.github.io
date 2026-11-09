"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def _compute_radius(points):
    """Compute the feasible radius for a set of points."""
    n = len(points)
    if n == 0:
        return 0.0
    # Distance to boundary
    r_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(r_boundary)
    # Nearest neighbor distances
    tree = cKDTree(points)
    pairs = tree.query_pairs(2.0)  # generous threshold
    min_dist = np.inf
    for i, j in pairs:
        d = np.linalg.norm(points[i] - points[j])
        if d < min_dist:
            min_dist = d
    if min_dist == np.inf:
        min_dist = 2.0 * min_boundary
    return min(min_dist / 2.0, min_boundary)

def _hexagonal_lattice():
    """Generate a hexagonal lattice of points within the unit circle."""
    best_points = None
    best_r = 0.0
    
    # Try a range of spacings
    for s in np.linspace(0.05, 0.12, 50):
        k = int(1.0 / s) + 2
        rows = [((i + 0.5 * (j % 2)) * s, j * s * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - s / 2.0]
        if len(points) >= N:
            # Pick the N points closest to center
            r = np.hypot(points[:, 0], points[:, 1])
            points = points[np.argsort(r, kind="stable")[:N]]
            r_val = _compute_radius(points)
            if r_val > best_r:
                best_r = r_val
                best_points = points.copy()
    
    return best_points

def _concentric_rings():
    """Generate points on concentric rings."""
    best_points = None
    best_r = 0.0
    
    # Try different ring configurations
    for n_rings in range(5, 15):
        for angle_offset in np.linspace(0, np.pi / 6, 5):
            points = [np.array([0.0, 0.0])]
            for ring in range(1, n_rings + 1):
                # Number of points on this ring
                n_pts = max(6, int(2 * np.pi * ring / 0.06))
                angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + angle_offset
                r = 1.0 - 0.055  # approximate radius
                # Adjust radius to fit
                for a in angles:
                    pts = np.array([r * np.cos(a), r * np.sin(a)])
                    # Check if this point would cause overlap
                    valid = True
                    for p in points:
                        if np.linalg.norm(pts - p) < 0.106:
                            valid = False
                            break
                    if valid:
                        points.append(pts)
            if len(points) >= N:
                points = np.array(points[:N])
                r_val = _compute_radius(points)
                if r_val > best_r:
                    best_r = r_val
                    best_points = points.copy()
    
    return best_points

def _square_grid():
    """Generate a square grid of points."""
    best_points = None
    best_r = 0.0
    
    for s in np.linspace(0.05, 0.12, 50):
        k = int(1.0 / s) + 2
        points = np.array([[i * s, j * s] for i in range(-k, k + 1) for j in range(-k, k + 1)])
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - s / 2.0]
        if len(points) >= N:
            r = np.hypot(points[:, 0], points[:, 1])
            points = points[np.argsort(r, kind="stable")[:N]]
            r_val = _compute_radius(points)
            if r_val > best_r:
                best_r = r_val
                best_points = points.copy()
    
    return best_points

def _refine_local(points, max_time=10.0):
    """Local refinement using gradient-free optimization."""
    start_time = time.time()
    current_points = points.copy()
    current_r = _compute_radius(current_points)
    
    # Simple hill climbing with random perturbations
    step_size = 0.005
    for iteration in range(100000):
        if time.time() - start_time > max_time:
            break
        
        # Try moving each point slightly
        idx = np.random.randint(len(current_points))
        original = current_points[idx].copy()
        
        # Random direction
        angle = np.random.uniform(0, 2 * np.pi)
        current_points[idx] = original + step_size * np.array([np.cos(angle), np.sin(angle)])
        
        # Check constraints
        new_r = _compute_radius(current_points)
        if new_r > current_r:
            current_r = new_r
        else:
            current_points[idx] = original
        
        # Gradually decrease step size
        if iteration % 1000 == 0:
            step_size *= 0.99
    
    return current_points, current_r

def construct():
    """Main function to construct the best packing found."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Generate multiple candidates
    candidates = []
    
    # Hexagonal lattice
    hex_pts = _hexagonal_lattice()
    if hex_pts is not None:
        candidates.append(hex_pts)
    
    # Concentric rings
    ring_pts = _concentric_rings()
    if ring_pts is not None:
        candidates.append(ring_pts)
    
    # Square grid
    sq_pts = _square_grid()
    if sq_pts is not None:
        candidates.append(sq_pts)
    
    # Select the best candidate
    best_points = None
    best_r = 0.0
    for pts in candidates:
        if len(pts) == N:
            r_val = _compute_radius(pts)
            if r_val > best_r:
                best_r = r_val
                best_points = pts.copy()
    
    if best_points is None:
        # Fallback: return a simple grid
        s = 0.06
        k = int(1.0 / s) + 2
        points = np.array([[i * s, j * s] for i in range(-k, k + 1) for j in range(-k, k + 1)])
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - s / 2.0]
        r = np.hypot(points[:, 0], points[:, 1])
        best_points = points[np.argsort(r, kind="stable")[:N]]
        best_r = _compute_radius(best_points)
    
    # Refine the best candidate if time allows
    if time.time() < deadline - 5.0:
        refined_points, refined_r = _refine_local(best_points, max_time=min(20.0, deadline - time.time() - 2.0))
        if refined_r > best_r:
            best_points = refined_points
            best_r = refined_r
    
    return best_points
# EVOLVE-BLOCK-END
