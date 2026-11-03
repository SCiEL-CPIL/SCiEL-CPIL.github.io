import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

N = 130

def construct():
    start_time = time.time()
    deadline = 140.0  # 140 seconds, leaving 10s for cleanup/return
    
    # Initial configuration: Fibonacci sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimization parameters
    step_size = 0.05
    decay_rate = 0.999
    min_step = 1e-5
    
    best_radius = np.inf
    best_points = points.copy()
    
    # Pre-compute some constants
    def normalize(v):
        norms = np.linalg.norm(v, axis=1)
        norms[norms == 0] = 1.0
        return v / norms[:, np.newaxis]
    
    def compute_covering_radius_and_gradients(points):
        # Compute convex hull
        try:
            hull = ConvexHull(points)
        except Exception:
            # If hull fails, return a large radius and zero gradients
            return np.pi, np.zeros_like(points)
        
        gradients = np.zeros_like(points)
        max_dist = 0.0
        worst_points_idx = []
        
        # Iterate over facets
        for simplex in hull.simplices:
            a, b, c = points[simplex]
            
            # Compute circumcenter
            # u is orthogonal to (a-b) and (b-c)
            v = np.cross(a - b, b - c)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-10:
                continue
            u = v / norm_v
            
            # Ensure u is on the same side as the origin relative to the facet?
            # Actually, we want the u such that u . a > 0
            if np.dot(u, a) < 0:
                u = -u
            
            # Check if u is a valid Voronoi vertex
            # It must be closer to a,b,c than to any other point
            dists = points @ u
            min_dist = np.min(dists)
            
            # If u is the nearest point to a, b, c, it's a valid vertex
            # dists[simplex] should be equal to min_dist (approximately)
            if np.allclose(dists[simplex], min_dist, rtol=1e-5, atol=1e-7):
                angle = np.arccos(np.clip(min_dist, -1.0, 1.0))
                if angle > max_dist:
                    max_dist = angle
                    worst_points_idx = [simplex]
                elif np.isclose(angle, max_dist, rtol=1e-5, atol=1e-7):
                    worst_points_idx.append(simplex)
        
        # If no valid vertices found (should not happen for a good config), fallback
        if max_dist == 0.0:
             # Calculate max distance to nearest point for all points?
             # Or just return a small value
             return 0.1, np.zeros_like(points)
            
        # Calculate gradients for the worst points
        # We want to move a, b, c towards u to reduce the distance
        # Gradient of distance w.r.t a is roughly (a - (a.u)u) / sin(dist)
        # But we just want to reduce the max distance.
        # Moving a towards u reduces the circumradius of the triangle.
        
        # Average gradients for points appearing in multiple worst facets
        grad_acc = np.zeros_like(points)
        count_acc = np.zeros(N)
        
        for simplex in worst_points_idx:
            a, b, c = points[simplex]
            v = np.cross(a - b, b - c)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-10:
                continue
            u = v / norm_v
            if np.dot(u, a) < 0:
                u = -u
            
            # Project (u - a) onto tangent space at a
            # Tangent direction: u - (u.a)a
            # But we want to move a towards u.
            # The vector u - a is not tangent.
            # The projection of u onto the tangent space at a is u - (u.a)a.
            # Since u is the circumcenter, u.a = cos(theta).
            # The direction to move is u - (u.a)a.
            
            for idx in simplex:
                p = points[idx]
                # Tangent component of u at p
                tangent_u = u - np.dot(u, p) * p
                grad_acc[idx] += tangent_u
                count_acc[idx] += 1
        
        # Normalize gradients
        mask = count_acc > 0
        gradients[mask] = grad_acc[mask] / count_acc[mask]
        
        return max_dist, gradients

    current_points = points.copy()
    
    while time.time() < deadline:
        radius, gradients = compute_covering_radius_and_gradients(current_points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = current_points.copy()
        
        # Update points
        # Riemannian gradient descent
        # p_new = normalize(p - step * grad)
        # But grad is already in tangent space?
        # tangent_u = u - (u.p)p is tangent to p.
        # So we move p in the direction of tangent_u.
        # Wait, moving p towards u means p_new = p + alpha * tangent_u?
        # Let's check: if p moves towards u, the distance decreases.
        # Yes.
        
        updates = step_size * gradients
        # Ensure updates are tangent
        # gradients are already tangent components.
        
        new_points = current_points + updates
        new_points = normalize(new_points)
        
        # Check for coincidence
        # If points get too close, it's bad.
        # We can add a small repulsion if needed, but the covering radius objective
        # naturally prevents points from coinciding if the hull is well-formed.
        
        current_points = new_points
        step_size *= decay_rate
        
        if step_size < min_step:
            break
            
    # Final check
    final_radius, _ = compute_covering_radius_and_gradients(best_points)
    if final_radius < best_radius:
        best_radius = final_radius
        # best_points is already updated in the loop if it was better
        
    return best_points
