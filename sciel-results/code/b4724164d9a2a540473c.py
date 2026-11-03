import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50
DEADLINE = 145.0 # 5s safety margin

def compute_covering_radius(points):
    # points: (N, 3)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.inf
    
    max_rad = 0.0
    for simplex in hull.simplices:
        a, b, c = points[simplex]
        # Normal to the plane of the triangle
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        m = n / norm_n
        
        # Check if m is on the correct side (facing outwards from origin? 
        # The circumcenter of the spherical triangle is the pole.
        # The dot product m.a should be positive if the triangle is small (< 90 deg).
        # If m.a is negative, the circumcenter is on the opposite side, 
        # meaning the "hole" is larger than 90 deg? 
        # For 50 points, facets are small.
        
        # The angular distance from m to a
        cos_theta = np.clip(np.dot(m, a), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        # If the circumcenter is "behind" the triangle (dot < 0), 
        # the actual furthest point in the Voronoi cell might be different?
        # For small triangles, m is the center.
        if theta > max_rad:
            max_rad = theta
            
    return max_rad

def lloyd_update(points, k=15):
    # Find k nearest neighbors for each point
    # Using cdist
    from scipy.spatial.distance import cdist
    dists = cdist(points, points)
    # Set self distance to inf
    np.fill_diagonal(dists, np.inf)
    # Get indices of k smallest
    nn_indices = np.argpartition(dists, k, axis=1)[:, :k]
    
    # Average neighbors
    new_points = np.zeros_like(points)
    for i in range(len(points)):
        neighbors = points[nn_indices[i]]
        new_points[i] = np.mean(neighbors, axis=0)
        norm = np.linalg.norm(new_points[i])
        if norm > 0:
            new_points[i] /= norm
    return new_points

def facet_shrink(points, iterations=100, step_size=0.1):
    for _ in range(iterations):
        # Compute hull and find worst facet
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # Calculate circumradius for each facet
        radii = []
        for simplex in hull.simplices:
            a, b, c = points[simplex]
            n = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                radii.append(0)
                continue
            m = n / norm_n
            cos_theta = np.clip(np.dot(m, a), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            radii.append(theta)
            
        radii = np.array(radii)
        if np.max(radii) < 1e-6:
            break
            
        # Find the worst facet
        worst_idx = np.argmax(radii)
        simplex = hull.simplices[worst_idx]
        a, b, c = points[simplex]
        
        # Recalculate m for this facet
        n = np.cross(b - a, c - a)
        m = n / np.linalg.norm(n)
        
        # Move vertices towards m
        # Adaptive step size?
        alpha = step_size
        
        new_a = a + alpha * (m - a)
        new_b = b + alpha * (m - b)
        new_c = c + alpha * (m - c)
        
        points[simplex[0]] = new_a / np.linalg.norm(new_a)
        points[simplex[1]] = new_b / np.linalg.norm(new_b)
        points[simplex[2]] = new_c / np.linalg.norm(new_c)
        
    return points

def construct():
    start_time = time.time()
    best_points = None
    best_radius = np.inf
    
    # Multiple restarts
    for restart in range(10):
        if time.time() - start_time > DEADLINE:
            break
            
        # Jittered Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        # Add jitter
        jitter = np.random.uniform(-0.1, 0.1, (N, 3))
        # Apply jitter in tangent plane? Or just perturb and renormalize?
        # Perturb and renormalize is simpler but might break symmetry less effectively if not careful.
        # Let's use a random rotation + jitter
        
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Random rotation
        # Generate random quaternion or rotation matrix
        # Simple: random axis, random angle
        axis = np.random.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = np.random.uniform(0, 2*np.pi)
        # Rotation matrix
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
        pts = pts @ R.T
        
        # Jitter
        pts += np.random.normal(0, 0.05, (N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        current_radius = compute_covering_radius(pts)
        
        # Lloyd's Algorithm
        lloyd_iters = 50
        for _ in range(lloyd_iters):
            if time.time() - start_time > DEADLINE:
                break
            pts = lloyd_update(pts, k=15)
        
        # Facet Shrink
        shrink_iters = 200
        for i_iter in range(shrink_iters):
            if time.time() - start_time > DEADLINE:
                break
            # Decrease step size
            step = 0.1 * (0.99 ** i_iter)
            pts = facet_shrink(pts, iterations=1, step_size=step)
            
        current_radius = compute_covering_radius(pts)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = pts.copy()
            
    if best_points is None:
        # Fallback
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    return best_points
