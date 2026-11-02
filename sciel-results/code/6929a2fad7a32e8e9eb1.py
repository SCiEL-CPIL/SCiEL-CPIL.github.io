"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2  # area of equilateral triangle with side 1 = sqrt(3)/4
DEADLINE = 145  # seconds, safety margin from 150s limit


def _min_area_idx(points):
    """Compute minimum triangle area and return (min_area, (i,j,k))."""
    n = len(points)
    min_a = float('inf')
    best_ijk = (0, 1, 2)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = points[j, 0] - points[i, 0]
                dy1 = points[j, 1] - points[i, 1]
                dx2 = points[k, 0] - points[i, 0]
                dy2 = points[k, 1] - points[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2) * 0.5
                if cross < min_a:
                    min_a = cross
                    best_ijk = (i, j, k)
    return min_a, best_ijk


def _min_area_fast(points):
    """Compute minimum triangle area using vectorized approach."""
    n = len(points)
    min_a = float('inf')
    for i in range(n - 2):
        # Vectorize over j, k pairs for fixed i
        pi = points[i]
        for j in range(i + 1, n - 1):
            pj = points[j]
            # All k > j
            pk = points[j + 1:]
            dx1 = pj[0] - pi[0]
            dy1 = pj[1] - pi[1]
            dx2 = pk[:, 0] - pi[0]
            dy2 = pk[:, 1] - pi[1]
            cross = np.abs(dx1 * dy2 - dy1 * dx2) * 0.5
            m = np.min(cross)
            if m < min_a:
                min_a = m
    return min_a


def _is_inside_triangle(p, margin=0.001):
    """Check if point is inside the equilateral triangle with margin."""
    x, y = p
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Check barycentric coordinates
    # Edge 1: from (0,0) to (1,0) - y >= 0
    if y < -margin:
        return False
    # Edge 2: from (1,0) to (0.5, sqrt(3)/2)
    # Line: y = -sqrt(3)*(x - 1) => sqrt(3)*x + y - sqrt(3) <= 0
    if np.sqrt(3) * x + y - np.sqrt(3) > margin:
        return False
    # Edge 3: from (0.5, sqrt(3)/2) to (0,0)
    # Line: y = sqrt(3)*x => -sqrt(3)*x + y <= 0
    if -np.sqrt(3) * x + y > margin:
        return False
    return True


def _clamp_to_triangle(p, margin=0.002):
    """Clamp point to be inside the triangle with margin."""
    x, y = float(p[0]), float(p[1])
    
    # Simple iterative projection
    for _ in range(50):
        # Check if inside
        if y < margin:
            y = margin
        if np.sqrt(3) * x + y - np.sqrt(3) > margin:
            # Project onto this edge
            # The edge line: sqrt(3)*x + y = sqrt(3)
            # Normal direction: (sqrt(3), 1) / 2
            # Move point inward
            dist = (np.sqrt(3) * x + y - np.sqrt(3)) / 2.0
            x -= dist * np.sqrt(3)
            y -= dist
        if -np.sqrt(3) * x + y > margin:
            dist = (-np.sqrt(3) * x + y) / 2.0
            x += dist * np.sqrt(3)
            y -= dist
        if x < margin:
            x = margin
        if x > 1 - margin:
            x = 1 - margin
        if y > S3H - margin:
            y = S3H - margin
    
    return np.array([x, y])


def _initial_config():
    """Create a good initial configuration with symmetry."""
    # Start with a grid-like arrangement inside the triangle
    # Use barycentric coordinates for better distribution
    points = []
    
    # Corners (slightly inset)
    m = 0.05
    points.append([m, m])
    points.append([1 - m, m])
    points.append([0.5, S3H - m])
    
    # Edge midpoints
    points.append([0.5, m])
    points.append([1 - 0.5*m, S3H * 0.5])
    points.append([0.5*m, S3H * 0.5])
    
    # Interior points - use centroid and symmetric positions
    # Centroid
    points.append([0.5, S3H / 3])
    
    # More interior points for symmetry
    # Create a smaller equilateral triangle inside
    scale = 0.3
    cx, cy = 0.5, S3H / 3
    r = scale
    for angle in [0, 2*np.pi/3, 4*np.pi/3]:
        points.append([cx + r * np.cos(angle), cy + r * np.sin(angle)])
    
    # Clamp all points
    points = [_clamp_to_triangle(p) for p in points]
    return np.array(points)


def _local_optimize(points, max_iter=5000, step=0.01):
    """Iteratively push the closest triple apart."""
    points = points.copy()
    n = len(points)
    
    for it in range(max_iter):
        min_a, (i, j, k) = _min_area_idx(points)
        
        # Compute centroid of the triple
        centroid = (points[i] + points[j] + points[k]) / 3.0
        
        # For each point in the triple, push it away from the other two
        for idx in [i, j, k]:
            others = [x for x in [i, j, k] if x != idx]
            # Direction away from the other two points
            dir_vec = np.zeros(2)
            for o in others:
                d = points[idx] - points[o]
                dist = np.linalg.norm(d)
                if dist > 1e-10:
                    dir_vec += d / dist
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm > 1e-10:
                dir_vec /= dir_norm
            
            # Try to move point in that direction
            new_pt = points[idx] + step * dir_vec
            if _is_inside_triangle(new_pt, margin=0.001):
                points[idx] = new_pt
            else:
                # Try smaller step or clamp
                new_pt = _clamp_to_triangle(points[idx] + step * 0.5 * dir_vec)
                points[idx] = new_pt
        
        # Also try to push all points slightly apart from nearest neighbor
        # to avoid getting stuck
        if it % 10 == 0:
            for idx in range(n):
                # Find nearest neighbor
                dists = np.linalg.norm(points - points[idx], axis=1)
                dists[idx] = np.inf
                nearest = np.argmin(dists)
                d = points[idx] - points[nearest]
                dist = np.linalg.norm(d)
                if dist > 1e-10:
                    d /= dist
                    new_pt = points[idx] + step * 0.1 * d
                    if _is_inside_triangle(new_pt, margin=0.001):
                        points[idx] = new_pt
    
    return points


def construct():
    start_time = time.time()
    
    # Initial configuration
    points = _initial_config()
    current_best = _min_area_fast(points)
    
    # Local optimization
    points = _local_optimize(points, max_iter=3000, step=0.005)
    val = _min_area_fast(points)
    if val > current_best:
        current_best = val
    
    # Check deadline
    if time.time() - start_time > DEADLINE - 10:
        return points
    
    # Try multiple restarts with perturbations
    rng = np.random.default_rng(42)
    for restart in range(20):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Perturb the best solution
        perturbed = points.copy()
        noise = rng.normal(0, 0.01, size=perturbed.shape)
        perturbed += noise
        perturbed = np.array([_clamp_to_triangle(p) for p in perturbed])
        
        # Optimize
        opt_pts = _local_optimize(perturbed, max_iter=1000, step=0.003)
        val = _min_area_fast(opt_pts)
        
        if val > current_best:
            current_best = val
            points = opt_pts
    
    return points
# EVOLVE-BLOCK-END
