import numpy as np
import scipy.spatial
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer for 120s limit

    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = np.inf

    # Precompute for speed
    # We will use a simple iterative local search
    
    # Helper to calculate covering radius from points
    def get_covering_radius(pts):
        try:
            hull = scipy.spatial.ConvexHull(pts)
            max_r = 0.0
            for simplex in hull.simplices:
                a = pts[simplex[0]]
                b = pts[simplex[1]]
                c = pts[simplex[2]]
                
                # Normal vector
                n = np.cross(b - a, c - a)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                u = n / norm_n
                
                # Ensure u is on the "outer" side (same hemisphere as normal)
                # The circumcenter for covering radius is the one farthest from origin in the cap
                # Actually, for a convex hull of points on a sphere, the normal points outward.
                # The circumcenter of the spherical triangle is u.
                # The angular radius is arccos(u . a)
                
                dot_val = np.dot(u, a)
                # Clip for numerical safety
                dot_val = np.clip(dot_val, -1.0, 1.0)
                r_ang = np.arccos(dot_val)
                if r_ang > max_r:
                    max_r = r_ang
            return max_r
        except Exception:
            return np.inf

    # Helper to get gradients for the worst facets
    def get_worst_facet_gradients(pts, num_worst=3):
        try:
            hull = scipy.spatial.ConvexHull(pts)
            facet_radii = []
            for idx, simplex in enumerate(hull.simplices):
                a = pts[simplex[0]]
                b = pts[simplex[1]]
                c = pts[simplex[2]]
                
                n = np.cross(b - a, c - a)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                u = n / norm_n
                dot_val = np.clip(np.dot(u, a), -1.0, 1.0)
                r_ang = np.arccos(dot_val)
                facet_radii.append((r_ang, simplex, u))
            
            if not facet_radii:
                return None, None
                
            # Sort by radius descending
            facet_radii.sort(key=lambda x: x[0], reverse=True)
            
            # Select top K
            worst_facets = facet_radii[:num_worst]
            
            # Accumulate gradients for vertices involved in worst facets
            # We want to move vertices towards the circumcenter of the facet they belong to
            # If a vertex is in multiple worst facets, we average the directions
            
            gradients = np.zeros_like(pts)
            counts = np.zeros(N)
            
            for r, simplex, u in worst_facets:
                for vertex_idx in simplex:
                    p = pts[vertex_idx]
                    # Tangential direction towards u
                    # proj = u - (u.p)p
                    # We want to move p towards u
                    tang = u - np.dot(u, p) * p
                    gradients[vertex_idx] += tang
                    counts[vertex_idx] += 1
            
            # Normalize gradients
            for i in range(N):
                if counts[i] > 0:
                    gradients[i] /= counts[i]
                    # Normalize the gradient vector itself? 
                    # The step size will handle magnitude
                    g_norm = np.linalg.norm(gradients[i])
                    if g_norm > 1e-10:
                        gradients[i] /= g_norm
            
            return gradients, worst_facets[0][0] # Return max radius too
        except Exception:
            return None, None

    # Main Loop
    step_size = 0.1
    # Adaptive step size: start larger, decay
    # We use a simple schedule
    
    for iter_num in range(10000):
        if time.time() > deadline:
            break
            
        grads, current_r = get_worst_facet_gradients(points, num_worst=5)
        
        if grads is None:
            # If hull failed, restart or break
            points = best_points.copy()
            continue
            
        if current_r < best_radius:
            best_radius = current_r
            best_points = points.copy()
            
        # Apply gradient step
        # Move points in the direction of the gradient (towards circumcenters)
        # This reduces the size of the worst caps
        points = points + step_size * grads
        
        # Renormalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Add small random noise to escape local minima (Simulated Annealing)
        # Temperature decays
        temp = 0.05 * np.exp(-iter_num / 500.0)
        if temp > 1e-5:
            noise = np.random.normal(0, temp, points.shape)
            points = points + noise
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            
        # Decay step size slightly to converge
        if iter_num % 100 == 0:
            step_size *= 0.95
            if step_size < 0.001:
                step_size = 0.001

    # Final Polish: Run a few more iterations with very small step size on best_points
    points = best_points.copy()
    for _ in range(500):
        if time.time() > deadline:
            break
        grads, current_r = get_worst_facet_gradients(points, num_worst=10)
        if grads is None:
            break
        points = points + 0.01 * grads
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        if current_r < best_radius:
            best_radius = current_r
            best_points = points.copy()
            
    return best_points
