import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_radius = 0.0
        
        # Vectorized calculation of circumradii is complex due to degenerate cases,
        # so we stick to the loop but keep it tight.
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
                
    except Exception:
        # If hull fails, return a large value
        return 180.0
        
    return np.degrees(max_radius)

def _get_circumcenters_and_radii(points, hull):
    """
    Compute circumcenters and radii for all hull facets.
    Returns centers (M, 3), radii (M,), and facets (M, 3).
    """
    facets = hull.simplices
    M = len(facets)
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    for k in range(M):
        tri = facets[k]
        p1 = points[tri[0]]
        p2 = points[tri[1]]
        p3 = points[tri[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            centers[k] = p1 # Degenerate, place at vertex
            radii[k] = 0.0
            continue
            
        c = cross / norm_cross
        # Ensure c points in the direction of the "empty cap" center relative to the facet normal
        # The normal of the facet points outwards from the origin.
        # The circumcenter on the sphere is either c or -c.
        # We want the one that is "inside" the Voronoi cell region defined by the hull.
        # For a convex hull of points on the sphere, the origin is inside.
        # The normal c points away from the origin.
        # The circumcenter of the spherical triangle is the point on the sphere equidistant from p1, p2, p3.
        # The angular distance is arccos(dot(c, p1)) or arccos(dot(-c, p1)).
        # Usually, for small triangles, the circumcenter is close to c if c . p1 > 0.
        # Let's check the dot product with the centroid of the triangle.
        centroid = (p1 + p2 + p3) / 3.0
        if np.dot(c, centroid) < 0:
            c = -c
            
        centers[k] = c
        dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        
        if dist > np.pi / 2:
            dist = np.pi - dist
            # If we flipped c, the distance logic holds, but let's ensure we take the small angle
            # Actually, if we flipped c to align with centroid, dist should be small.
            # If dist is still large, the triangle is large.
            
        radii[k] = dist
        
    return centers, radii, facets

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
    best_radius = _compute_covering_radius(points)
    
    # Optimization parameters
    step_size = 0.05
    min_step_size = 1e-6
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        
        # Compute Hull and circumcenters
        try:
            hull = ConvexHull(points)
            centers, radii, facets = _get_circumcenters_and_radii(points, hull)
        except Exception:
            # If hull fails, try a random kick
            kick = rng.normal(0, 0.01, points.shape)
            points += kick
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            continue

        # For each point, find the adjacent triangle with the maximum radius
        # We need to map from point index to list of adjacent facet indices
        # This is a sparse operation. We can do it in a loop for N=130.
        max_move = np.zeros_like(points)
        
        # Initialize max radius per point to 0
        point_max_radius = np.zeros(N)
        point_max_center = np.zeros((N, 3))
        
        for k in range(len(facets)):
            r_k = radii[k]
            if r_k <= 0: continue
            
            c_k = centers[k]
            tri = facets[k]
            for idx in tri:
                if r_k > point_max_radius[idx]:
                    point_max_radius[idx] = r_k
                    point_max_center[idx] = c_k
        
        # Move points towards their max center
        # The direction is (center - point) projected onto tangent plane
        # Actually, center is on sphere. Point is on sphere.
        # We want to move point towards center.
        # Vector: center - point.
        # Tangent: v - (v . p) p
        for i in range(N):
            if point_max_radius[i] > 0:
                v = point_max_center[i] - points[i]
                # Project v onto tangent plane at points[i]
                dot_vp = np.dot(v, points[i])
                tangent_v = v - dot_vp * points[i]
                norm_tv = np.linalg.norm(tangent_v)
                if norm_tv > 1e-12:
                    max_move[i] = tangent_v / norm_tv
                    # Scale by radius? Larger holes should attract more?
                    # Or just fixed step?
                    # Let's scale by the radius to prioritize large holes.
                    max_move[i] *= point_max_radius[i]
        
        # Update points
        new_points = points + step_size * max_move
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        current_radius = _compute_covering_radius(new_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = new_points.copy()
            points = new_points
            no_improve_count = 0
            # Increase step size slightly if we are making progress
            step_size = min(step_size * 1.05, 0.1)
        else:
            # If worse, revert and decrease step size
            step_size = max(step_size * 0.8, min_step_size)
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Stagnation: try a random kick
                kick = rng.normal(0, 0.05, points.shape)
                points += kick
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                step_size = 0.01
                no_improve_count = 0
                current_radius = _compute_covering_radius(points)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = points.copy()

    return best_points
