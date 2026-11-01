"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _compute_covering_radius(points):
    """
    Compute the exact covering radius using the spherical Delaunay triangulation
    (via 3D Convex Hull). The covering radius is the maximum circumradius of the
    triangular facets of the convex hull, assuming the circumcenter lies within
    the corresponding spherical Voronoi cell. For points on a sphere, the
    circumcenter of a Delaunay triangle is the Voronoi vertex.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback if hull construction fails (e.g., coplanar points)
        # Use a random probe method as a fallback
        probe_count = 20000
        i = np.arange(probe_count) + 0.5
        z = 1.0 - 2.0 * i / probe_count
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        probe_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Simple KDTree query for fallback
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        dists, _ = tree.query(probe_points, k=1)
        return np.max(np.arccos(np.clip(dists, -1, 1))) # dists are chord distances? No, KDTree is Euclidean.
        # Euclidean dist d on unit sphere -> angle = 2*arcsin(d/2)
        return np.max(2 * np.arcsin(np.clip(dists / 2.0, 0, 1)))

    max_radius = 0.0
    # Iterate over all triangular facets
    for simplex in hull.simplices:
        # simplex contains indices of 3 points
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        # Compute circumcenter of the triangle p1, p2, p3
        # The circumcenter lies in the plane of the triangle.
        # We can find it by solving the linear system for the intersection of
        # the perpendicular bisectors, or using the formula:
        # C = (||p1||^2 (p2 - p3) + ||p2||^2 (p3 - p1) + ||p3||^2 (p1 - p2)) / (2 * ||(p1 - p2) x (p1 - p3)||)
        # Since points are on unit sphere, ||pi||^2 = 1.
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        denom = 2.0 * np.dot(cross, cross)
        
        if denom < 1e-12:
            # Degenerate triangle (collinear or coincident points)
            # This shouldn't happen in a valid hull, but handle it
            continue
            
        # Formula for circumcenter of triangle with vertices on unit sphere:
        # C = ( (p2 - p3) * ||p1||^2 + (p3 - p1) * ||p2||^2 + (p1 - p2) * ||p3||^2 ) / (2 * ||(p1-p2) x (p1-p3)||)
        # Wait, the standard formula for circumcenter of triangle ABC:
        # C = ( ||A||^2 (B - C) + ||B||^2 (C - A) + ||C||^2 (A - B) ) / (2 * ||(A-B) x (A-C)||)
        # Let's verify.
        # Actually, there is a simpler vector formula for the circumcenter of a triangle on the unit sphere.
        # The circumcenter is in the direction of the normal to the plane if the triangle is right-angled? No.
        
        # Let's use the linear algebra approach.
        # We want C such that |C - p1|^2 = |C - p2|^2 = |C - p3|^2 = R^2.
        # C is in the plane spanned by p1, p2, p3.
        # C = a * p1 + b * p2 + c * p3.
        # This is getting complex. Let's use the explicit formula:
        # C = ( (p2 - p3) * (p1 dot p1) + (p3 - p1) * (p2 dot p2) + (p1 - p2) * (p3 dot p3) ) / (2 * ||(p1 - p2) x (p1 - p3)||)
        # Since p_i are unit vectors:
        numerator = (p2 - p3) * 1.0 + (p3 - p1) * 1.0 + (p1 - p2) * 1.0
        # Wait, (p2-p3) + (p3-p1) + (p1-p2) = 0. This formula is for Euclidean circumcenter?
        # Yes, the formula C = (||A||^2(B-C) + ...) works for general points.
        # But if A,B,C are on the unit sphere, the numerator sums to 0?
        # (B-C) + (C-A) + (A-B) = 0.
        # This implies the circumcenter is at the origin? That's only if the triangle is right-angled at the origin? No.
        
        # Let's re-derive.
        # We want C such that C dot (p2 - p1) = (||p2||^2 - ||p1||^2)/2 = 0.
        # So C is orthogonal to (p2 - p1) and (p3 - p1).
        # Thus C is parallel to the normal vector n = (p2 - p1) x (p3 - p1).
        # So the circumcenter of a triangle on the unit sphere lies on the line passing through the origin and the circumcenter of the planar triangle?
        # No. The circumcenter of the spherical triangle is the projection of the 3D circumcenter of the planar triangle onto the sphere?
        # The 3D circumcenter of the planar triangle formed by p1, p2, p3 is a point O.
        # The spherical circumcenter is O / ||O||.
        
        # Let's compute the planar circumcenter O.
        # O = p1 + t1 * v1 + t2 * v2
        # |O - p1|^2 = |O - p2|^2 => |t1 v1 + t2 v2|^2 = |t1 v1 + t2 v2 - v1|^2
        # 2 t1 ||v1||^2 + 2 t2 (v1 dot v2) = ||v1||^2
        # 2 t1 (v1 dot v2) + 2 t2 ||v2||^2 = ||v2||^2
        # Solve for t1, t2.
        
        a11 = np.dot(v1, v1)
        a12 = np.dot(v1, v2)
        a22 = np.dot(v2, v2)
        b1 = 0.5 * a11
        b2 = 0.5 * a22
        
        det = a11 * a22 - a12 * a12
        if abs(det) < 1e-12:
            continue
            
        t1 = (b1 * a22 - b2 * a12) / det
        t2 = (a11 * b2 - a12 * b1) / det
        
        O = p1 + t1 * v1 + t2 * v2
        
        # Normalize O to get the spherical circumcenter
        norm_O = np.linalg.norm(O)
        if norm_O < 1e-10:
            continue
        C = O / norm_O
        
        # The circumradius is the angle between C and any vertex, e.g., p1
        cos_theta = np.dot(C, p1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        # We must ensure C is the "farthest" point in the Voronoi cell.
        # For a Delaunay triangle, the circumcenter is the Voronoi vertex.
        # The distance from C to the vertices is the circumradius.
        # Is it possible that the covering radius is determined by a point on the edge of the Voronoi cell?
        # No, the farthest point in a convex polytope (Voronoi cell) from the origin (center of sphere) is a vertex.
        # Wait, the covering radius is the max distance from any point y on the sphere to the nearest p_i.
        # This is equivalent to the max circumradius of the Delaunay triangles.
        # Why? The set of points closer to p_i than any other p_j is the Voronoi cell V_i.
        # The function d(y, p_i) is convex on the sphere? No.
        # But the maximum of the minimum distance function occurs at a Voronoi vertex (circumcenter of a Delaunay triangle).
        # So checking all Delaunay triangle circumcenters is sufficient.
        
        if theta > max_radius:
            max_radius = theta
            
    return max_radius

def _local_search(points, time_limit=100.0):
    """
    Local search: Identify the worst facet (largest circumradius) and move its vertices
    slightly towards the circumcenter to shrink the triangle.
    """
    points = points.copy()
    best_radius = _compute_covering_radius(points)
    start_time = time.time()
    
    step_size = 0.05
    while time.time() - start_time < time_limit:
        # Find the worst facet
        try:
            hull = ConvexHull(points)
        except Exception:
            break
            
        worst_simplex = None
        worst_radius = 0.0
        
        for simplex in hull.simplices:
            p1 = points[simplex[0]]
            p2 = points[simplex[1]]
            p3 = points[simplex[2]]
            
            v1 = p2 - p1
            v2 = p3 - p1
            
            a11 = np.dot(v1, v1)
            a12 = np.dot(v1, v2)
            a22 = np.dot(v2, v2)
            b1 = 0.5 * a11
            b2 = 0.5 * a22
            
            det = a11 * a22 - a12 * a12
            if abs(det) < 1e-12:
                continue
                
            t1 = (b1 * a22 - b2 * a12) / det
            t2 = (a11 * b2 - a12 * b1) / det
            
            O = p1 + t1 * v1 + t2 * v2
            norm_O = np.linalg.norm(O)
            if norm_O < 1e-10:
                continue
            C = O / norm_O
            
            cos_theta = np.clip(np.dot(C, p1), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            
            if theta > worst_radius:
                worst_radius = theta
                worst_simplex = simplex
                worst_center = C
                
        if worst_simplex is None:
            break
            
        # Move the three vertices of the worst facet towards the circumcenter
        # We do this by taking a small step towards the center.
        # This shrinks the triangle, reducing its circumradius.
        # However, we must be careful not to make points coincide or cross.
        
        new_points = points.copy()
        for idx in worst_simplex:
            p = points[idx]
            # Move towards C
            # p_new = p + step * (C - p)
            # Then normalize
            direction = worst_center - p
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-10:
                direction = direction / norm_dir
                p_new = p + step_size * direction
                norm_p_new = np.linalg.norm(p_new)
                if norm_p_new > 1e-10:
                    new_points[idx] = p_new / norm_p_new
        
        # Evaluate new radius
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < best_radius:
            points = new_points
            best_radius = new_radius
            # Decay step size slightly
            step_size *= 0.95
            if step_size < 1e-5:
                break
        else:
            # If it didn't improve, try a smaller step or random perturbation of the worst points
            step_size *= 0.8
            if step_size < 1e-6:
                break
            
            # Random perturbation of the worst points to escape local minima
            for idx in worst_simplex:
                noise = np.random.randn(3) * 0.01
                p_new = points[idx] + noise
                norm_p_new = np.linalg.norm(p_new)
                if norm_p_new > 1e-10:
                    new_points[idx] = p_new / norm_p_new
            
            new_radius = _compute_covering_radius(new_points)
            if new_radius < best_radius:
                points = new_points
                best_radius = new_radius
            else:
                # If random perturbation also fails, we might be stuck.
                # Try a different facet? Or just break?
                # Let's try to perturb all points slightly randomly to escape.
                all_noise = np.random.randn(N, 3) * 0.005
                new_points = points + all_noise
                new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
                new_radius = _compute_covering_radius(new_points)
                if new_radius < best_radius:
                    points = new_points
                    best_radius = new_radius
                else:
                    break
                    
    return points

def _generate_initial_configurations():
    """
    Generate initial configurations.
    """
    configs = []
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    configs.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # 2. Random Configurations
    rng = np.random.RandomState(12345)
    for _ in range(10):
        pts = rng.randn(N, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        configs.append(pts)
        
    return configs

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    initial_configs = _generate_initial_configurations()
    
    best_points = None
    best_radius = float('inf')
    
    for idx, init_pts in enumerate(initial_configs):
        if time.time() - start_time > time_limit:
            break
            
        # Run Local Search
        relaxed_pts = _local_search(init_pts, time_limit=min(10.0, time_limit - (time.time() - start_time)))
        
        # Evaluate covering radius
        radius = _compute_covering_radius(relaxed_pts)
        
        if radius < best_radius:
            best_radius = radius
            best_points = relaxed_pts
            
    if best_points is None:
        best_points = initial_configs[0]
        
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
