import numpy as np
import time
from scipy.spatial import ConvexHull
import random

N = 130

# EVOLVE-BLOCK-START
def _covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    The covering radius is the maximum angular distance from any point on the sphere
    to the nearest point in the set. This is equivalent to the maximum circumradius
    of the facets of the convex hull of the points.
    """
    if points.shape[0] < 4:
        return np.pi / 2  # Fallback for degenerate cases
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi / 2
        
    max_r = 0.0
    # Iterate over all facets (simplices)
    for simplex in hull.simplices:
        # simplex contains indices of points forming a tetrahedron (in 3D, facets are triangles)
        # Wait, in 3D, convex hull facets are triangles.
        # The circumcenter of a triangle on the unit sphere is the normalized sum of the vertices?
        # No, the circumcenter of a spherical triangle is the point on the sphere equidistant from the vertices.
        # It is the normalized vector sum of the vertices only if they form an equilateral triangle?
        # Actually, the circumcenter of a set of points on a sphere is the intersection of the sphere
        # with the line passing through the origin and the circumcenter of the planar triangle formed by the points.
        
        # Let's compute the circumcenter of the planar triangle defined by the 3 points.
        p1, p2, p3 = points[simplex]
        
        # Vectors
        v1 = p2 - p1
        v2 = p3 - p1
        
        # Normal to the plane of the triangle
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        n = n / n_norm
        
        # Circumcenter of the triangle in 3D space
        # The circumcenter C satisfies |C - p1| = |C - p2| = |C - p3|
        # C = p1 + a*v1 + b*v2
        # 2*a*(v1.v1) + 2*b*(v1.v2) = |v1|^2
        # 2*a*(v2.v1) + 2*b*(v2.v2) = |v2|^2
        
        a = np.dot(v1, v1)
        b = np.dot(v1, v2)
        c = np.dot(v2, v2)
        
        d1 = 0.5 * np.dot(v1, v1)
        d2 = 0.5 * np.dot(v2, v2)
        
        det = a * c - b * b
        if abs(det) < 1e-10:
            continue
            
        # Solve for parameters s, t such that C = p1 + s*v1 + t*v2
        # s*a + t*b = d1
        # s*b + t*c = d2
        s = (d1 * c - d2 * b) / det
        t = (d2 * a - d1 * b) / det
        
        c_center = p1 + s * v1 + t * v2
        
        # The circumradius of the spherical triangle is the angle between the circumcenter (normalized) and any vertex.
        # The point on the sphere closest to the plane of the triangle (in the direction of the circumcenter)
        # is the center of the spherical cap covering the triangle.
        
        # Normalize c_center to get the point on the sphere
        c_norm = np.linalg.norm(c_center)
        if c_norm < 1e-10:
             continue
        c_sphere = c_center / c_norm
        
        # The angular distance from c_sphere to p1 is the circumradius of this facet.
        # cos(theta) = dot(c_sphere, p1)
        cos_theta = np.dot(c_sphere, p1)
        # Clamp to [-1, 1] to avoid numerical issues with arccos
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        if theta > max_r:
            max_r = theta
            
    return max_r

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n):
    # Generate random points on the sphere
    v = np.random.randn(n, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 5 seconds margin
    
    # Initial configuration
    points = _fibonacci_sphere(N)
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Annealing parameters
    temperature = 0.05
    cooling_rate = 0.9999
    min_temperature = 1e-4
    step_size = 0.1
    
    current_radius = best_radius
    
    while time.time() - start_time < time_limit:
        # Check if we should stop
        if temperature < min_temperature:
            break
            
        # Select a random point to move
        idx = np.random.randint(N)
        original_point = points[idx].copy()
        
        # Generate a random perturbation
        # Move in a random direction on the sphere
        # A small rotation is a good way to keep points on the sphere
        axis = np.random.randn(3)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-10:
            continue
        axis = axis / axis_norm
        
        # Random angle
        angle = np.random.uniform(-step_size, step_size)
        
        # Rotation matrix
        c = np.cos(angle)
        s = np.sin(angle)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + s * K + (1 - c) * (K @ K)
        
        new_point = R @ original_point
        points[idx] = new_point
        
        new_radius = _covering_radius(points)
        
        if new_radius < current_radius:
            # Accept
            current_radius = new_radius
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = points.copy()
        else:
            # Metropolis criterion
            delta = new_radius - current_radius
            prob = np.exp(-delta / temperature)
            if np.random.random() < prob:
                current_radius = new_radius
            else:
                # Reject, revert
                points[idx] = original_point
        
        # Cool down
        temperature *= cooling_rate
        # Occasionally increase step size or adjust
        if np.random.random() < 0.01:
             step_size = np.random.uniform(0.05, 0.2)

    return _normalize(best_points)
# EVOLVE-BLOCK-END
