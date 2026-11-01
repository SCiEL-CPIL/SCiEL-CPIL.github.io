"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
    def fibonacci_sphere(n, offset=0.5):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts

    def get_facet_data(pts):
        """
        Computes convex hull, facet radii, centers, and for each point,
        the center of the worst facet it belongs to.
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            return None, None, None, None

        num_facets = len(hull.simplices)
        radii = np.zeros(num_facets)
        centers = np.zeros((num_facets, 3))
        
        for i, simplex in enumerate(hull.simplices):
            v1 = pts[simplex[0]]
            v2 = pts[simplex[1]]
            v3 = pts[simplex[2]]
            
            edge1 = v2 - v1
            edge2 = v3 - v1
            n = np.cross(edge1, edge2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
                
            c = n / norm_n
            if np.dot(c, v1) < 0:
                c = -c
                
            d = np.linalg.norm(c - v1)
            radius = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
            
            radii[i] = radius
            centers[i] = c
            
        # Map points to facets
        point_to_facets = [[] for _ in range(N)]
        for i, simplex in enumerate(hull.simplices):
            for idx in simplex:
                point_to_facets[idx].append(i)
                
        target_centers = np.zeros((N, 3))
        worst_radii = np.zeros(N)
        
        for i in range(N):
            if not point_to_facets[i]:
                target_centers[i] = pts[i]
                continue
                
            max_r = -1
            max_f = -1
            for f_idx in point_to_facets[i]:
                if radii[f_idx] > max_r:
                    max_r = radii[f_idx]
                    max_f = f_idx
                    
            worst_radii[i] = max_r
            target_centers[i] = centers[max_f]
            
        return radii, centers, target_centers, worst_radii

    def calc_covering_radius(pts):
        radii, _, _, _ = get_facet_data(pts)
        if radii is None:
            return np.pi
        return np.max(radii)

    # 1. Initialization
    current_points = fibonacci_sphere(N).copy()
    best_points = current_points.copy()
    best_radius = calc_covering_radius(current_points)
    
    # 2. Improved Lloyd Iteration
    # Move all points towards the circumcenter of their worst facet.
    
    base_step = 0.05
    decay_rate = 0.995
    min_step = 0.0001
    max_iter = 5000
    
    current_step = base_step
    stuck_count = 0
    
    for iter in range(max_iter):
        if time.time() > deadline - 15:
            break
            
        radii, centers, target_centers, worst_radii = get_facet_data(current_points)
        
        if radii is None:
            break
            
        current_radius = np.max(radii)
        
        if current_radius < best_radius - 1e-9:
            best_radius = current_radius
            best_points = current_points.copy()
            stuck_count = 0
        else:
            stuck_count += 1
            
        # Calculate weights based on worst radius
        # Points with larger local errors should move more
        max_wr = np.max(worst_radii)
        if max_wr > 1e-10:
            weights = worst_radii / max_wr
        else:
            weights = np.ones(N)
            
        # Update points
        new_points = current_points.copy()
        for i in range(N):
            dir_vec = target_centers[i] - current_points[i]
            norm_dir = np.linalg.norm(dir_vec)
            if norm_dir > 1e-10:
                dir_vec /= norm_dir
                move = current_step * weights[i]
                new_points[i] += move * dir_vec
                
        # Normalize
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        # Check for invalid configurations (e.g., points too close)
        # If points are too close, hull might be unstable.
        # We just accept and let the next iteration handle it, 
        # but if we get stuck, we perturb.
        
        current_points = new_points
        
        # Decay step size
        current_step *= decay_rate
        
        # If stuck or step too small, perturb and reset step
        if stuck_count > 50 or current_step < min_step:
            noise = np.random.randn(N, 3) * 0.05
            current_points = current_points + noise
            current_points /= np.linalg.norm(current_points, axis=1, keepdims=True)
            current_step = base_step * 0.5
            stuck_count = 0

        # Occasional restart
        if stuck_count > 500:
            R = np.random.randn(3, 3)
            R, _ = np.linalg.qr(R)
            current_points = fibonacci_sphere(N) @ R.T
            current_points /= np.linalg.norm(current_points, axis=1, keepdims=True)
            current_step = base_step
            stuck_count = 0

    # 3. Final Polish: Coordinate Descent on Worst Facet Vertices
    # This is a local refinement to squeeze out the last bit of improvement.
    
    if time.time() < deadline - 5:
        current_best = best_points.copy()
        current_score = best_radius
        step = 0.01
        
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            improved = False
            indices = np.random.permutation(N)
            for j in indices:
                if time.time() > deadline - 5:
                    break
                for _trial in range(10):
                    rand_vec = np.random.randn(3)
                    rand_vec /= np.linalg.norm(rand_vec)
                    
                    old_p = current_best[j].copy()
                    new_p = current_best[j] + step * rand_vec
                    new_p /= np.linalg.norm(new_p)
                    
                    current_best[j] = new_p
                    new_score = calc_covering_radius(current_best)
                    
                    if new_score < current_score - 1e-9:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        current_best[j] = old_p
                        
            if not improved:
                step *= 0.5
                if step < 1e-5:
                    break
                    
        if current_score < best_radius:
            best_points = current_best
            best_radius = current_score

    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
