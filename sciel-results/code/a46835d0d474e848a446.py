"""Model-and-optimize approach for covering the sphere with 130 equal caps."""
import numpy as np
from scipy.optimize import minimize
import time

N = 130

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return points / norms

def _covering_radius(points):
    """
    Compute the covering radius in radians.
    points: (N, 3) unit vectors.
    The covering radius is the maximum over all points y on the sphere
    of the angular distance to the nearest point in the configuration.
    
    This equals the maximum circumradius of the Delaunay triangulation
    (or equivalently, for a set of points on the sphere, the largest
    empty circle radius). A practical computation: for each point, 
    find its nearest neighbors and compute the circumcenters of the 
    triangles (or use the fact that the covering radius is the max
    over all y of min_i angle(y, p_i)).
    
    For a robust estimate, we use the convex hull approach:
    The covering radius is the maximum circumradius of the facets
    of the convex hull of the points.
    """
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        max_circumradius = 0.0
        for simplex in hull.simplices:
            # simplex contains 3 indices for a triangle
            tri = points[simplex]
            # Compute circumcenter of triangle on unit sphere
            a, b, c = tri[0], tri[1], tri[2]
            # Circumcenter is in the direction of a+b+c (for equilateral), 
            # but for general triangles on unit sphere, we need to solve
            # |x - a| = |x - b| = |x - c| with |x| = 1
            # This means x is in the plane of a,b,c and equidistant from a,b,c
            # The circumcenter direction is proportional to the cross product solution
            # Actually, for points on unit sphere, the circumcenter of triangle abc
            # is in the direction of the normal to the plane, scaled appropriately.
            
            # A simpler approach: the circumradius of a spherical triangle
            # can be computed from the side lengths (angular distances)
            # using the spherical law of cosines.
            
            # Side lengths (angular distances)
            d_ab = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
            d_bc = np.arccos(np.clip(np.dot(b, c), -1.0, 1.0))
            d_ca = np.arccos(np.clip(np.dot(c, a), -1.0, 1.0))
            
            # Spherical circumradius formula:
            # R = sqrt( (sin(A) * sin(B) * sin(C)) / (4 * sin(s/2) * sin((s-A)/2) * sin((s-B)/2) * sin((s-C)/2)) )
            # where A, B, C are angles of the spherical triangle, s = (a+b+c)/2
            # This is complex. Let's use a different approach.
            
            # Alternative: the circumcenter of a spherical triangle is the point
            # equidistant from all three vertices. For a triangle on the unit sphere,
            # we can find it by solving the linear system.
            
            # The circumcenter x satisfies:
            # x·a = x·b = x·c = cos(R)
            # and |x| = 1
            
            # This means x is in the null space of [a-b, b-c] and satisfies x·a = cos(R)
            # Let's find the direction of x first.
            
            # x is proportional to (b-c) × (c-a) + ... no, let's think differently.
            
            # The set of points equidistant from a and b is the great circle through the 
            # midpoint of the arc ab and perpendicular to it. Similarly for b and c.
            # The intersection of these two great circles gives the circumcenter.
            
            # The normal to the great circle equidistant from a and b is (a-b) normalized? No.
            # Points x with x·a = x·b satisfy x·(a-b) = 0. So the great circle is 
            # perpendicular to (a-b).
            
            # So x is in the intersection of the planes x·(a-b)=0 and x·(b-c)=0.
            # This gives x ∝ (a-b) × (b-c)
            
            v1 = a - b
            v2 = b - c
            x_dir = np.cross(v1, v2)
            x_norm = np.linalg.norm(x_dir)
            if x_norm < 1e-10:
                continue
            x = x_dir / x_norm
            
            # Check if x is on the same side as the triangle centroid
            centroid = (a + b + c) / 3.0
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 1e-10:
                centroid = centroid / centroid_norm
                if np.dot(x, centroid) < 0:
                    x = -x
            
            # Compute the circumradius
            cos_R = np.clip(np.dot(x, a), -1.0, 1.0)
            R = np.arccos(cos_R)
            
            max_circumradius = max(max_circumradius, R)
        
        return max_circumradius
    except Exception:
        # Fallback: use a grid-based approximation
        # This is slow but robust
        theta = np.linspace(0, 2*np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
        TH, PH = np.meshgrid(theta, phi)
        X = np.sin(PH) * np.cos(TH)
        Y = np.sin(PH) * np.sin(TH)
        Z = np.cos(PH)
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # Compute distances from each grid point to nearest configuration point
        # Using dot product: cos(angle) = dot product
        dots = grid_points @ points.T  # (num_grid, N)
        cos_angles = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(cos_angles)
        min_angles = np.min(angles, axis=1)
        return np.max(min_angles)

def _objective(flat_params):
    """Objective function: covering radius in radians."""
    points = _normalize(flat_params.reshape(N, 3))
    return _covering_radius(points)

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _local_optimization(initial_points, max_time=100.0):
    """Local optimization using gradient descent on the sphere."""
    start_time = time.time()
    points = initial_points.copy()
    
    # Use scipy minimize with L-BFGS-B on the flat parameters
    # We'll optimize on the flat representation and normalize after each step
    
    def flat_objective(flat):
        elapsed = time.time() - start_time
        if elapsed > max_time:
            # Return a large value to signal termination
            return 1e10
        pts = _normalize(flat)
        return _covering_radius(pts)
    
    best_points = points.copy()
    best_val = _covering_radius(best_points)
    
    # Try multiple random perturbations and optimize each
    rng = np.random.default_rng(42)
    
    # First, try optimizing the initial configuration
    try:
        result = minimize(
            flat_objective,
            points.ravel(),
            method='L-BFGS-B',
            options={'maxiter': 50, 'maxfun': 100}
        )
        if result.fun < best_val:
            best_points = _normalize(result.x)
            best_val = result.fun
    except Exception:
        pass
    
    # Try some random restarts if time permits
    elapsed = time.time() - start_time
    if elapsed < max_time - 10:
        for restart in range(5):
            if time.time() - start_time > max_time - 5:
                break
            
            # Perturb the best points
            noise = rng.normal(0, 0.1, size=(N, 3))
            perturbed = _normalize(points + noise)
            
            try:
                result = minimize(
                    flat_objective,
                    perturbed.ravel(),
                    method='L-BFGS-B',
                    options={'maxiter': 30, 'maxfun': 60}
                )
                if result.fun < best_val:
                    best_points = _normalize(result.x)
                    best_val = result.fun
            except Exception:
                pass
    
    return best_points

def construct():
    """Construct 130 points on the unit sphere to minimize covering radius."""
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin: 120s limit, leave 30s buffer
    
    # Start with Fibonacci sphere
    points = _fibonacci_sphere(N)
    
    # Local optimization
    if time.time() < deadline - 10:
        points = _local_optimization(points, max_time=100.0)
    
    # Final normalization
    points = _normalize(points)
    
    # Verify no coincident points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(points[i] - points[j]) < 1e-8:
                # Separate them slightly
                points[j] = _normalize(points[j] + np.random.normal(0, 0.01, 3))
    
    return points
