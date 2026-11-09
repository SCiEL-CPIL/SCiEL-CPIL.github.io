"""Optimized covering of the unit sphere with 130 points."""
import time
import numpy as np

N = 130

def _covering_radius(points):
    """Compute the covering radius in degrees for points on the unit sphere."""
    # points: (N, 3) array of unit vectors
    # The covering radius is the maximum over the sphere of the angular distance
    # to the nearest point. This equals the maximum circumradius of the convex hull facets.
    # A faster approximation: for each point, compute the max angular distance to its
    # 3 nearest neighbors (forming a triangle), then take half the max angle of that triangle
    # as a local upper bound. But the true covering radius is the max over all points on the sphere.
    
    # More accurate: the covering radius is determined by the Delaunay triangulation on the sphere.
    # For each triangular facet, the circumradius (angular) is the max distance from the circumcenter
    # to the three vertices. The covering radius is the max of these circumradii over all facets.
    
    # A simpler but valid approach: sample many points on the sphere and compute max min-distance.
    # But we need accuracy. Let's use the fact that the covering radius equals the max over all
    # points of the min angular distance to the set.
    
    # Efficient method: for each point p_i, the worst case is at the circumcenter of its nearest
    # neighbors. A practical approximation: for each point, find its k nearest neighbors, form
    # triangles, compute circumradii, take the max.
    
    # Actually, the most reliable method within time: use scipy's ConvexHull to get facets,
    # then for each facet compute the circumradius.
    
    from scipy.spatial import ConvexHull
    from scipy.spatial import distance_matrix
    
    try:
        hull = ConvexHull(points)
        facets = hull.simplices  # each facet is a triangle (3 points)
        
        max_circumradius = 0.0
        for simplex in facets:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            
            # Compute the circumradius of the triangle on the sphere.
            # The circumcenter of three points on the unit sphere:
            # Find the point equidistant from p1, p2, p3 on the sphere.
            
            # The circumradius is the angular distance from the circumcenter to any vertex.
            # For three points on the unit sphere, we can compute the circumradius using the formula:
            # Given three points with pairwise angular distances a, b, c (opposite to vertices),
            # the circumradius R satisfies: sin(R) = (sin(a) * sin(b) * sin(c)) / (4 * area)
            # where area is the spherical excess area of the triangle.
            
            # Simpler: compute the circumcenter directly.
            # The circumcenter C satisfies C·p1 = C·p2 = C·p3 = cos(R).
            # C is in the plane spanned by p1, p2, p3 (the plane of the triangle).
            # C = alpha * p1 + beta * p2 + gamma * p3 with C·p1 = C·p2 = C·p3.
            
            # Alternative: use the formula for circumradius of a spherical triangle.
            # Let a = angle between p2 and p3, b = angle between p1 and p3, c = angle between p1 and p2.
            # Spherical excess E = a + b + c - pi.
            # Area = E (for unit sphere).
            # sin(R) = sin(a)*sin(b)*sin(c) / (4*sin(E/2)^2) ... this is getting complex.
            
            # Direct computation: The circumcenter of three points on the unit sphere.
            # Let A = p1, B = p2, C = p3.
            # The circumcenter X satisfies |X - A|_angular = |X - B|_angular = |X - C|_angular.
            # Equivalently, X·A = X·B = X·C = cos(R).
            # X lies in the plane of A, B, C: X = u*A + v*B + w*C.
            # And |X| = 1.
            
            # Let's solve: X·A = X·B = X·C.
            # (uA+vB+wC)·A = (uA+vB+wC)·B => u(A·A) + v(B·A) + w(C·A) = u(A·B) + v(B·B) + w(C·B)
            # u(1 - A·B) + v(A·B - 1) + w(A·C - B·C) = 0
            # Similarly for other pairs.
            
            # Let's use a more numerical approach: 
            # The circumcenter is the intersection of the sphere with the line through the origin
            # perpendicular to the plane of the triangle... no, that's the normal.
            
            # Actually, the circumcenter of a spherical triangle is the pole of the great circle
            # through the three points... no, it's the point equidistant from all three.
            
            # Let me use a direct formula. For three unit vectors a, b, c:
            # The circumradius R satisfies:
            # sin(R) = |a × b| |b × c| |c × a| / (4 * |a + b + c| * something)
            
            # Let me just compute it numerically by finding the circumcenter.
            # The circumcenter x satisfies:
            # x · a = x · b = x · c
            # |x| = 1
            
            # Let x = a + s*(b-a) + t*(c-a)  (in the affine plane of a,b,c)
            # But x must be on the unit sphere.
            
            # Alternative: use the fact that for a triangle on the sphere, the circumradius
            # can be computed from the side lengths.
            
            a = np.arccos(np.clip(np.dot(p2, p3), -1, 1))
            b = np.arccos(np.clip(np.dot(p1, p3), -1, 1))
            c = np.arccos(np.clip(np.dot(p1, p2), -1, 1))
            
            # Spherical law of cosines for circumradius:
            # sin(R) = sin(a)*sin(b)*sin(c) / (4*sin((a+b+c)/2)*sin((a-b+c)/2)*sin((a+b-c)/2)*sin((-a+b+c)/2))
            # This is Heron's formula for spherical triangles.
            
            s = (a + b + c) / 2
            denom = 4 * np.sin(s) * np.sin(s-a) * np.sin(s-b) * np.sin(s-c)
            
            if abs(denom) > 1e-15:
                sin_R = np.sin(a) * np.sin(b) * np.sin(c) / denom
                sin_R = np.clip(sin_R, 0, 1)
                R = np.arcsin(sin_R)
            else:
                # Degenerate case: points are nearly collinear
                R = max(a, b, c) / 2
            
            if R > max_circumradius:
                max_circumradius = R
        
        return np.degrees(max_circumradius)
    
    except Exception:
        # Fallback: sample-based estimate
        # Generate many random points and compute max min-distance
        rng = np.random.RandomState(42)
        n_samples = 10000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        
        # Compute min distance from each sample to the set of points
        # This is expensive but reliable
        min_dists = np.inf
        for i in range(0, n_samples, 1000):
            chunk = samples[i:i+1000]
            d = 1 - np.einsum('ij,kj->ik', chunk, points)
            d = np.clip(d, -1, 1)
            min_d = np.min(d, axis=1)
            min_dists = np.minimum(min_dists, min_d)
        
        max_min_dist = np.max(min_dists)
        return np.degrees(np.arccos(np.clip(max_min_dist, -1, 1)))


def _optimize(points, max_time=100):
    """Optimize the covering radius using gradient descent on the sphere."""
    start_time = time.time()
    points = points.copy()
    
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Repulsion-based optimization: minimize sum of exp(-k * min_dist) or similar
    # Actually, let's use a direct approach: minimize the covering radius by moving points
    # away from the worst-covered regions.
    
    # Strategy: iteratively find the worst point (max min-distance), then move the nearest
    # configuration point away from it.
    
    lr = 0.01
    for iteration in range(5000):
        if time.time() - start_time > max_time:
            break
        
        # Find the worst-covered region by sampling
        rng = np.random.RandomState(iteration)
        n_samples = 2000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        
        # Compute min distance from each sample to points
        d = 1 - np.einsum('ij,kj->ik', samples, points)
        d = np.clip(d, -1, 1)
        min_d = np.min(d, axis=1)
        min_idx = np.argmin(d, axis=1)
        
        # Find the worst sample
        worst_idx = np.argmax(min_d)
        worst_sample = samples[worst_idx]
        nearest_point_idx = min_idx[worst_idx]
        worst_dist = min_d[worst_idx]
        
        # Move the nearest point away from the worst sample
        # Gradient: move point in the direction away from worst_sample
        p = points[nearest_point_idx]
        # Direction to move: project (p - worst_sample) onto the tangent space at p
        move_dir = p - worst_sample
        move_dir -= np.dot(move_dir, p) * p
        move_dir /= (np.linalg.norm(move_dir) + 1e-10)
        
        points[nearest_point_idx] = p + lr * move_dir
        # Renormalize
        points[nearest_point_idx] /= np.linalg.norm(points[nearest_point_idx])
        
        # Periodically check the radius
        if iteration % 100 == 0:
            radius = _covering_radius(points)
            if radius < best_radius:
                best_radius = radius
                best_points = points.copy()
    
    return best_points, best_radius


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    # Start with Fibonacci spiral
    points = _fibonacci_sphere(N)
    
    # Optimize
    points, radius = _optimize(points, max_time=60)
    
    # Check deadline
    if time.time() > deadline - 5:
        pass  # Return what we have
    
    # Final normalization
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points
