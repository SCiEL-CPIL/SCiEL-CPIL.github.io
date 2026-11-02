"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def construct():
    # Deadline guard
    start_time = time.time()
    max_time = 110.0  # Leave 10s margin for 120s limit

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def get_covering_radius(points):
        # Compute convex hull
        try:
            hull = ConvexHull(points)
        except Exception:
            return np.pi  # Invalid or degenerate, return large value
            
        max_rad = 0.0
        # Hull equations are n_dot_x = offset
        # Facets are triangles in 3D for generic points
        for simplex in hull.simplices:
            p1 = points[simplex[0]]
            p2 = points[simplex[1]]
            p3 = points[simplex[2]]
            
            # Calculate normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            
            # Ensure normal points outward (away from origin)
            # The hull contains the origin, so the normal should point away from the centroid?
            # Actually, for points on the unit sphere, the normal to the facet points outward from the hull.
            # The Voronoi vertex is in the direction of this normal.
            # Let's verify: p1, p2, p3 are on the sphere.
            # The plane is n . x = d.
            # Since p1 is on the plane, d = n . p1.
            # The point on the sphere in direction n is y = n (since n is unit).
            # Distance from y to p1 is arccos(y . p1).
            
            # Check if n points in the right direction.
            # If n . p1 is negative, it means the normal points inwards?
            # For a convex hull of points on the sphere, the origin is inside.
            # The normal should point away from the origin.
            # So n . p1 should be positive (since p1 is on the sphere and n is outward).
            # If n . p1 < 0, flip n.
            if np.dot(normal, p1) < 0:
                normal = -normal

            # The Voronoi vertex is 'normal'
            # Distance is arccos(normal . p1)
            # Note: normal . p1 might be slightly > 1 or < -1 due to numerical errors
            cos_val = np.clip(np.dot(normal, p1), -1.0, 1.0)
            rad = np.arccos(cos_val)
            
            if rad > max_rad:
                max_rad = rad
                
        return max_rad

    # Initial points
    points = fibonacci_spiral(N)
    current_radius = get_covering_radius(points)
    best_points = points.copy()
    best_radius = current_radius
    
    # Local search
    step_size = 0.05
    no_improve_count = 0
    
    while time.time() - start_time < max_time:
        # Find the worst facet
        try:
            hull = ConvexHull(points)
        except Exception:
            # If hull fails, perturb points
            points += np.random.normal(0, 0.01, (N, 3))
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            continue

        worst_rad = 0.0
        worst_simplex = None
        worst_normal = None
        
        for simplex in hull.simplices:
            p1 = points[simplex[0]]
            p2 = points[simplex[1]]
            p3 = points[simplex[2]]
            
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            
            if np.dot(normal, p1) < 0:
                normal = -normal

            cos_val = np.clip(np.dot(normal, p1), -1.0, 1.0)
            rad = np.arccos(cos_val)
            
            if rad > worst_rad:
                worst_rad = rad
                worst_simplex = simplex
                worst_normal = normal
        
        if worst_simplex is None:
            break

        # Try moving each vertex of the worst simplex towards the normal
        improved = False
        for idx in worst_simplex:
            p = points[idx]
            # Move towards worst_normal
            # Geodesic step: p_new = normalize(p + step * normal)
            # We want to reduce the angle between p and normal.
            # So we move p in the direction of normal.
            new_p = p + step_size * worst_normal
            new_p /= np.linalg.norm(new_p)
            
            # Store old point
            old_p = p.copy()
            points[idx] = new_p
            
            new_radius = get_covering_radius(points)
            
            if new_radius < current_radius:
                current_radius = new_radius
                improved = True
                no_improve_count = 0
                # Adjust step size: increase slightly if improved
                step_size = min(step_size * 1.05, 0.1)
                break
            else:
                # Revert
                points[idx] = old_p
        
        if not improved:
            no_improve_count += 1
            # Reduce step size
            step_size = max(step_size * 0.9, 0.005)
            
            # If stuck for a while, try a random perturbation
            if no_improve_count > 20:
                # Randomly perturb a few points
                for _ in range(3):
                    idx = np.random.randint(N)
                    p = points[idx]
                    # Random direction on sphere
                    rand_dir = np.random.normal(0, 1, 3)
                    rand_dir /= np.linalg.norm(rand_dir)
                    new_p = p + step_size * rand_dir
                    new_p /= np.linalg.norm(new_p)
                    points[idx] = new_p
                
                new_radius = get_covering_radius(points)
                if new_radius < current_radius:
                    current_radius = new_radius
                    no_improve_count = 0
                    step_size = 0.05
                else:
                    # Revert if worse? Or accept with probability (simulated annealing)?
                    # For simplicity, revert if worse to ensure monotonic improvement, 
                    # but this can get stuck. Let's accept if it's not much worse.
                    if new_radius < current_radius + 0.01:
                        current_radius = new_radius
                        no_improve_count = 0
                    else:
                        # Revert to best
                        points = best_points.copy()
                        current_radius = best_radius
                        # Restart with new random seed?
                        # Let's just reset no_improve_count
                        no_improve_count = 0
                        step_size = 0.05

        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()

    return best_points
# EVOLVE-BLOCK-END
