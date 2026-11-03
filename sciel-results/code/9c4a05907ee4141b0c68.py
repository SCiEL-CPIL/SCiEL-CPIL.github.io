import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    Returns radius in degrees and the indices of the worst facets.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_radius = 0.0
        worst_facets = []
        
        for tri in facets:
            p1 = points[tri[0]]
            p2 = points[tri[1]]
            p3 = points[tri[2]]
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-12:
                continue
                
            c = cross / norm_cross
            
            dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
            dist = np.arccos(dot_val)
            
            if dist > np.pi / 2:
                dist = np.pi - dist
                
            if dist > max_radius:
                max_radius = dist
                worst_facets = [tri]
            elif abs(dist - max_radius) < 1e-6:
                worst_facets.append(tri)
                
    except Exception:
        # If hull fails, return a large value and empty facets
        return 180.0, []
        
    return np.degrees(max_radius), worst_facets

def _compute_all_facet_data(points):
    """
    Compute circumcenters and radii for all facets.
    Returns list of (circumcenter_vector, radius_radians, facet_indices)
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return []
        
    facets = hull.simplices
    results = []
    
    for tri in facets:
        p1 = points[tri[0]]
        p2 = points[tri[1]]
        p3 = points[tri[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            continue
            
        c = cross / norm_cross
        
        dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        
        # Ensure c points towards the "hole" (the empty cap center)
        # For a convex hull, the outward normal points away from the interior.
        # The "hole" is the region of the sphere not covered by the caps of the vertices.
        # If the triangle is small, the circumcenter is the outward normal.
        # If the triangle is large (covering > hemisphere), the circumcenter of the *empty* part
        # is the inward normal.
        # However, for a good packing, all facets are small.
        # The covering radius is determined by the maximum distance to the nearest point.
        # This maximum occurs at a circumcenter of a Delaunay triangle.
        # For small triangles, the circumcenter is c (outward normal).
        # The distance from c to the vertices is dist.
        # We want to minimize this dist.
        
        # Note: if dist > pi/2, the "hole" is actually on the other side?
        # No, the circumcenter of the spherical triangle is unique.
        # If the triangle contains the origin in its convex hull (in 3D), i.e., is large,
        # then the circumcenter on the sphere is far.
        # But we are minimizing the max distance.
        # So we just use dist.
        
        results.append((c, dist, tri))
        
    return results

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    points += rng.normal(0, 0.01, points.shape)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = points.copy()
    best_radius, _ = _compute_covering_radius(points)
    
    # Structural Change:
    # We use a "simulated annealing" style approach with a focus on the worst facets.
    # We compute the gradient of the max circumradius with respect to the points.
    # The max circumradius is the max over all facets of the circumradius.
    # Let R = max_k r_k.
    # We want to minimize R.
    # The gradient of r_k with respect to p_i is non-zero only if p_i is a vertex of facet k.
    # For a facet k with vertices i, j, l and circumcenter c_k,
    # r_k = arccos(c_k . p_i).
    # d(r_k)/d(p_i) = -c_k / sqrt(1 - (c_k . p_i)^2) (tangent component).
    # We sum the gradients of the top K worst facets, weighted by their radius.
    
    step_size = 0.02
    min_step_size = 1e-6
    temperature = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        
        # 1. Compute current covering radius and worst facets
        current_radius, worst_facets = _compute_covering_radius(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        # 2. If no worst facets, break
        if not worst_facets:
            break
            
        # 3. Compute gradients for the top K worst facets
        # We limit K to 5 to focus on the most problematic areas.
        K = 5
        top_facets = worst_facets[:K]
        
        gradient = np.zeros_like(points)
        
        for tri in top_facets:
            p1 = points[tri[0]]
            p2 = points[tri[1]]
            p3 = points[tri[2]]
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-12:
                continue
                
            c = cross / norm_cross
            
            # Ensure c is the circumcenter of the "hole"
            # If the triangle is small, c is the outward normal.
            # The distance is arccos(c . p_i).
            # We want to reduce this distance.
            # So we move p_i towards c.
            
            for idx in tri:
                p_i = points[idx]
                dot_pc = np.dot(p_i, c)
                dot_pc = np.clip(dot_pc, -1.0, 1.0)
                
                # Gradient of arccos(dot) wrt p_i is -c / sqrt(1 - dot^2)
                # This vector points from p_i towards c (tangent direction).
                # We add this to the gradient to move p_i towards c.
                
                denom = np.sqrt(1.0 - dot_pc * dot_pc)
                if denom > 1e-9:
                    grad_vec = -c / denom
                    gradient[idx] += grad_vec
                    
        # 4. Normalize gradient
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > 1e-9:
            gradient /= grad_norm
            
            # 5. Update points
            # Move in the direction of the gradient (towards the hole centers)
            # But we also add a small random component to escape local minima.
            noise = rng.normal(0, 0.005, points.shape)
            new_points = points + step_size * (gradient + noise)
            new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
            
            # 6. Check if this step improved the radius
            new_radius, _ = _compute_covering_radius(new_points)
            
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = new_points.copy()
                points = new_points
                # Reduce step size slightly
                step_size = max(step_size * 0.98, min_step_size)
                temperature *= cooling
            else:
                # If worse, revert and decrease step size
                step_size = max(step_size * 0.95, min_step_size)
                
                # Occasionally, try a larger random kick to escape local minima
                if rng.random() < 0.01:
                    kick = rng.normal(0, 0.05, points.shape)
                    points += kick
                    points /= np.linalg.norm(points, axis=1, keepdims=True)
                    step_size = 0.02

    return best_points
