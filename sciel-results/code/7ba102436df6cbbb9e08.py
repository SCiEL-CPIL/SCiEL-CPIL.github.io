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
        
        # Vectorized calculation of circumradii
        # For each facet (triangle), find the circumcenter on the sphere.
        # The circumcenter is the pole of the great circle through the vertices,
        # but specifically the one that is the center of the empty cap.
        
        # Get the three points for each facet
        p1 = points[facets[:, 0]]
        p2 = points[facets[:, 1]]
        p3 = points[facets[:, 2]]
        
        # Compute normal vector to the plane of the triangle
        # v1 = p2 - p1, v2 = p3 - p1
        # cross = v1 x v2
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross, axis=1)
        
        # Avoid division by zero
        valid_mask = norm_cross > 1e-12
        if not np.any(valid_mask):
            return 180.0
            
        # Normalize cross to get unit normal
        c = cross[valid_mask] / norm_cross[valid_mask, np.newaxis]
        
        # The circumcenter on the sphere is either c or -c.
        # We need the one that is the center of the "empty" cap.
        # For a convex hull facet, the normal points outward.
        # The "hole" is on the side of the normal? 
        # Actually, the vertices p1, p2, p3 are on the sphere.
        # The distance from c to p1 is arccos(c . p1).
        # The distance from -c to p1 is arccos(-c . p1) = pi - arccos(c . p1).
        # The covering radius is the max of the minimum distances to vertices.
        # For a facet, the point on the sphere in that facet region farthest from vertices
        # is the circumcenter.
        # If the triangle is small, the circumcenter is "above" the triangle (in the direction of the normal).
        # Let's check the dot product with the centroid or just pick the one that gives a smaller angle?
        # No, we want the distance from the circumcenter to the vertices.
        # If c . p1 > 0, the angle is acute. 
        # Usually, for small triangles, c points in the direction of the centroid of the triangle (roughly).
        # So c . p1 should be positive and close to 1.
        # Let's compute dist = arccos(c . p1).
        # If dist > pi/2, we should use -c, which gives pi - dist.
        # So effectively, the radius is min(dist, pi - dist).
        
        dot_vals = np.einsum('ij,ij->i', c, p1[valid_mask])
        dot_vals = np.clip(dot_vals, -1.0, 1.0)
        dists = np.arccos(dot_vals)
        
        # The radius is the angular distance from the circumcenter to the vertices.
        # If dists > pi/2, the "other" circumcenter (-c) is closer?
        # No, the distance from c to p1 is dists.
        # The distance from -c to p1 is pi - dists.
        # We want the center of the empty cap. 
        # The empty cap is the one that doesn't contain any other points.
        # For a convex hull facet, the outward normal points away from the rest of the points.
        # So the circumcenter should be on the side of the outward normal.
        # Therefore, the radius is dists if dists < pi/2, else pi - dists?
        # Actually, if the triangle is very large, the "hole" might be on the other side?
        # But for 130 points, triangles are small.
        # Let's assume the radius is the angle between c and p1, clamped to [0, pi/2].
        # Wait, if c is the normal, and the triangle is small, c is roughly the average of p1,p2,p3.
        # So c . p1 is close to 1. dists is small.
        # So we take dists.
        
        # But what if c . p1 is negative? Then c points away from p1.
        # Then the circumcenter is -c.
        # So the radius is min(dists, pi - dists).
        
        radii = np.minimum(dists, np.pi - dists)
        max_radius = np.max(radii)
        
    except Exception:
        return 180.0
        
    return np.degrees(max_radius)

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
    
    # Use a more robust optimization: 
    # 1. Compute the Voronoi cells (or approximations) 
    # 2. Move each point towards the circumcenter of its largest facet
    # This is a "spherical Lloyd's algorithm" or similar.
    
    # For now, let's stick to the repulsive force model but improve the step control.
    # The previous method had issues with step size decay.
    
    # Let's try a different approach: 
    # Minimize the sum of squared distances to the nearest neighbor? No, that's packing.
    # We want to minimize the maximum distance.
    
    # Let's use the "largest facet circumcenter" method.
    # For each point, find the facet(s) it belongs to.
    # Actually, it's easier to find the circumcenters of all facets.
    # Then, for each point, move it towards the circumcenter of the facet 
    # that is "farthest" from it? No.
    
    # Standard algorithm for spherical covering:
    # 1. Find the convex hull.
    # 2. For each facet, compute the circumcenter.
    # 3. If the circumcenter is inside the facet (on the sphere), 
    #    the distance to the vertices is the radius.
    # 4. Move the vertices of the facet towards the circumcenter.
    
    # Let's implement this.
    
    max_iter = 500
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        try:
            hull = ConvexHull(points)
            facets = hull.simplices
            
            # Compute circumcenters for all facets
            p1 = points[facets[:, 0]]
            p2 = points[facets[:, 1]]
            p3 = points[facets[:, 2]]
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross, axis=1)
            
            valid_mask = norm_cross > 1e-12
            if not np.any(valid_mask):
                break
                
            c = cross[valid_mask] / norm_cross[valid_mask, np.newaxis]
            
            # Determine the correct circumcenter (c or -c)
            # The circumcenter should be on the same side as the facet normal?
            # Actually, we want the point on the sphere equidistant from p1, p2, p3.
            # This is c or -c.
            # We choose the one that is the center of the "empty" cap.
            # For a convex hull, the normal points outward.
            # The "hole" is in the direction of the normal.
            # So we want c such that c . (p1+p2+p3) > 0? 
            # Let's check the dot product with the centroid.
            centroids = (p1[valid_mask] + p2[valid_mask] + p3[valid_mask]) / 3.0
            dot_centroid = np.einsum('ij,ij->i', c, centroids)
            
            # If dot_centroid < 0, flip c
            flip_mask = dot_centroid < 0
            c[flip_mask] *= -1
            
            # Compute the radius (distance from c to p1)
            dot_vals = np.einsum('ij,ij->i', c, p1[valid_mask])
            dot_vals = np.clip(dot_vals, -1.0, 1.0)
            radii = np.arccos(dot_vals)
            
            # The radius is the angular distance.
            # We want to move the vertices towards c to reduce the radius.
            
            # For each facet, we have a circumcenter c and a radius r.
            # We want to move p1, p2, p3 towards c.
            # The step size should be proportional to the radius?
            # Or just a small fixed step?
            
            # Let's accumulate the movement for each point.
            # For each facet, we move each vertex towards the circumcenter.
            # The direction is (c - p_i) / |c - p_i|? 
            # No, on the sphere, we move along the geodesic.
            # The direction is the projection of c onto the tangent plane at p_i.
            # Or simply, we add c to p_i and normalize?
            # Let's use: new_p = p + alpha * (c - (c.p)p)
            # This moves p towards c along the great circle.
            
            # We want to move vertices of the LARGEST facets more?
            # Or all facets?
            # Let's move all vertices towards the circumcenters of the facets they belong to.
            
            # Accumulate forces
            forces = np.zeros_like(points)
            
            # For each valid facet
            for idx in range(len(c)):
                facet_idx = facets[valid_mask][idx]
                circumcenter = c[idx]
                
                # Move each vertex towards the circumcenter
                for v_idx in facet_idx:
                    p_v = points[v_idx]
                    # Project circumcenter onto tangent plane at p_v
                    # Tangent component of c relative to p_v
                    dot_pc = np.dot(circumcenter, p_v)
                    tangent_comp = circumcenter - dot_pc * p_v
                    
                    # Normalize the tangent component to get a unit direction
                    norm_tc = np.linalg.norm(tangent_comp)
                    if norm_tc > 1e-12:
                        direction = tangent_comp / norm_tc
                        forces[v_idx] += direction
                        
            # Normalize forces to prevent large steps
            force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
            force_norms[force_norms == 0] = 1.0
            forces = forces / force_norms
            
            # Update points
            step_size = 0.01
            new_points = points + step_size * forces
            new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
            
            # Check improvement
            current_radius = _compute_covering_radius(new_points)
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = new_points.copy()
                points = new_points
            else:
                # If no improvement, reduce step size or break
                # For now, just break if no improvement for a while?
                # Let's just continue with smaller step?
                # To keep it simple, let's just break if we don't improve.
                # This is a bit aggressive, but avoids getting stuck.
                # Alternatively, we can randomize.
                # Let's try a random kick if we don't improve.
                if it > 10: # Give it a few iterations to start
                    kick = rng.normal(0, 0.01, points.shape)
                    points += kick
                    points /= np.linalg.norm(points, axis=1, keepdims=True)
                    current_radius = _compute_covering_radius(points)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_points = points.copy()
                    else:
                        break
                        
        except Exception:
            break
            
    return best_points
