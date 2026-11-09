import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _covering_radius(points):
    """
    Compute the covering radius in degrees.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        p1 = points[facets[:, 0]]
        p2 = points[facets[:, 1]]
        p3 = points[facets[:, 2]]
        
        # Vector from p2 to p1 and p3 to p1
        v1 = p1 - p2
        v2 = p1 - p3
        
        # Normal vector to the plane (unnormalized circumcenter direction)
        c_dir = np.cross(v1, v2)
        norms = np.linalg.norm(c_dir, axis=1)
        
        valid = norms > 1e-10
        if not np.any(valid):
            return 90.0
            
        c = np.zeros_like(c_dir)
        c[valid] = c_dir[valid] / norms[valid, np.newaxis]
        
        # Orient towards centroid
        centroids = (p1 + p2 + p3) / 3.0
        c_norms = np.linalg.norm(centroids, axis=1)
        valid_centroids = c_norms > 1e-10
        centroids[valid_centroids] /= c_norms[valid_centroids, np.newaxis]
        
        dots = np.sum(c * centroids, axis=1)
        c[dots < 0] *= -1
        
        # Angular distance from c to p1
        cos_theta = np.sum(c * p1, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        theta[~valid] = 0.0
        return np.max(theta) * 180.0 / np.pi
        
    except Exception:
        return 180.0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Parameters
    # We will alternate between a global Lloyd-like step (smoothing) 
    # and a local Facet Repair step (targeting the worst case).
    
    lr_global = 0.05
    lr_repair = 0.02
    
    iterations = 3000
    
    for iter in range(iterations):
        if time.time() > deadline:
            break
            
        # Every 20 iterations, evaluate the true radius
        if iter % 20 == 0:
            current_radius = _covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
                
        # Phase 1: Global Lloyd's Step (Voronoi Centroid Approximation)
        # Move each point towards the centroid of its adjacent facets.
        # This is expensive to do exactly with Delaunay, so we approximate 
        # by moving towards the average of neighbors in the Delaunay graph.
        # However, for speed, we can just use a simple repulsion force 
        # or a simplified Lloyd step.
        # Let's use a simple repulsion for the global phase to maintain spread.
        # F_i = sum_{j != i} (P_i - P_j) / ||P_i - P_j||^3  (Coulomb-like)
        # Or simply move away from the average of all other points? No, that's not right.
        
        # Let's stick to the Facet Repair as the primary driver, 
        # with a small global repulsion to prevent collapse.
        
        # Compute Repulsion Force (Global)
        # Vectorized repulsion:
        # diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        # dists = np.linalg.norm(diff, axis=2)
        # mask = np.eye(N, dtype=bool)
        # np.fill_diagonal(dists, 1e-10)
        # forces = np.sum(diff / (dists[:, :, np.newaxis] ** 3), axis=1)
        
        # This is O(N^2). N=130, so 16900 ops. Very fast.
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        mask = np.eye(N, dtype=bool)
        np.fill_diagonal(dists, 1.0) # Avoid div by zero
        forces_global = np.sum(diff / (dists[:, :, np.newaxis] ** 3), axis=1)
        
        # Phase 2: Facet Repair
        facet_idx, circumcenter, current_r = _get_worst_facet_and_center(points)
        
        if facet_idx is not None:
            v1, v2, v3 = facet_idx
            # Move these 3 vertices towards the circumcenter
            # This shrinks the triangle, reducing its circumradius.
            # We move them a small fraction of the way.
            p1 = points[v1]
            p2 = points[v2]
            p3 = points[v3]
            
            # Direction towards circumcenter
            d1 = circumcenter - p1
            d2 = circumcenter - p2
            d3 = circumcenter - p3
            
            # Normalize directions
            n1 = np.linalg.norm(d1)
            n2 = np.linalg.norm(d2)
            n3 = np.linalg.norm(d3)
            
            if n1 > 1e-10: d1 /= n1
            if n2 > 1e-10: d2 /= n2
            if n3 > 1e-10: d3 /= n3
            
            # Update points
            points[v1] += lr_repair * d1
            points[v2] += lr_repair * d2
            points[v3] += lr_repair * d3
            
            # To prevent the 3 points from collapsing into a single point,
            # we also apply a strong repulsion between them specifically.
            # The global repulsion handles this, but let's boost it for these 3.
            # Actually, moving towards circumcenter *reduces* the triangle size.
            # If we reduce the triangle size, the circumradius decreases.
            # But if we make it too small, the neighbors might form larger triangles.
            # The global repulsion counteracts the shrinking.
            
        # Apply Global Repulsion
        points += lr_global * 0.1 * forces_global
        
        # Project to sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Periodic global "Lloyd" style correction:
        # Every 100 iterations, do a full Lloyd step to escape local minima 
        # where facet repair might get stuck.
        if iter % 100 == 0:
            # Simple Lloyd: move each point to the average of its nearest neighbors
            # This is a heuristic for Voronoi centroid.
            # Find 10 nearest neighbors for each point.
            # For N=130, this is fast.
            for i in range(N):
                dists_to_i = np.linalg.norm(points - points[i], axis=1)
                dists_to_i[i] = np.inf
                nearest_indices = np.argsort(dists_to_i)[:10]
                centroid = np.mean(points[nearest_indices], axis=0)
                # Move towards centroid
                points[i] += 0.5 * (centroid - points[i])
            
            # Re-normalize
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms

    # Final check
    final_radius = _covering_radius(points)
    if final_radius < best_radius:
        best_radius = final_radius
        best_points = points.copy()
        
    return best_points

def _get_worst_facet_and_center(points):
    """
    Returns the indices of the worst facet (max circumradius) and its circumcenter.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        p1 = points[facets[:, 0]]
        p2 = points[facets[:, 1]]
        p3 = points[facets[:, 2]]
        
        v1 = p1 - p2
        v2 = p1 - p3
        c_dir = np.cross(v1, v2)
        norms = np.linalg.norm(c_dir, axis=1)
        
        valid = norms > 1e-10
        if not np.any(valid):
            return None, None, 0.0
            
        c = np.zeros_like(c_dir)
        c[valid] = c_dir[valid] / norms[valid, np.newaxis]
        
        centroids = (p1 + p2 + p3) / 3.0
        c_norms = np.linalg.norm(centroids, axis=1)
        valid_centroids = c_norms > 1e-10
        centroids[valid_centroids] /= c_norms[valid_centroids, np.newaxis]
        
        dots = np.sum(c * centroids, axis=1)
        c[dots < 0] *= -1
        
        cos_theta = np.sum(c * p1, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        theta[~valid] = 0.0
        
        max_idx = np.argmax(theta)
        if not valid[max_idx]:
            return None, None, 0.0
            
        return facets[max_idx], c[max_idx], theta[max_idx]
    except Exception:
        return None, None, 0.0

if __name__ == "__main__":
    pts = construct()
    print(f"Covering radius: {_covering_radius(pts)} degrees")
    print(pts.shape)
