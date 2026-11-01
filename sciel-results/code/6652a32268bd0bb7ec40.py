"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (5s buffer)
    
    def fibonacci_sphere(n, offset=0.5):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts

    def get_max_circumradius(pts):
        """
        Computes the convex hull and returns the maximum angular distance
        from the center of the sphere to the vertices of any facet.
        This is the covering radius.
        Returns (radius_in_radians, worst_simplex_indices, worst_center).
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            # If hull fails (e.g., points coplanar or degenerate), return large penalty
            return np.pi, None, None
        
        max_radius = 0.0
        worst_simplex = None
        worst_center = None
        
        for simplex in hull.simplices:
            v1 = pts[simplex[0]]
            v2 = pts[simplex[1]]
            v3 = pts[simplex[2]]
            
            # Calculate circumcenter of the triangle formed by v1, v2, v3
            # Since points are on unit sphere, the circumcenter of the spherical triangle
            # is the normalized cross product of the edge vectors (normal to the plane)
            # IF the triangle is small enough or we are looking for the pole of the great circle.
            # Actually, the covering radius is determined by the circumcircle of the Delaunay triangle on the sphere.
            # The center of this circumcircle is the point on the sphere furthest from the vertices.
            
            edge1 = v2 - v1
            edge2 = v3 - v1
            n = np.cross(edge1, edge2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
                
            # The normal vector to the plane of the triangle
            c = n / norm_n
            
            # Ensure c points in the same hemisphere as the vertices (outward)
            # The center of the circumcircle on the sphere is c or -c.
            # We want the one that is "outside" the triangle relative to the origin?
            # No, the circumcenter of a spherical triangle is the intersection of the perpendicular bisectors.
            # For a triangle on the unit sphere, the circumcenter is indeed the normalized cross product
            # of the edges, provided the triangle does not contain the antipode of the center.
            # Let's check the dot product.
            if np.dot(c, v1) < 0:
                c = -c
                
            # Calculate angular distance from c to v1
            # cos(angle) = c.dot(v1)
            # angle = arccos(c.dot(v1))
            # Using d = ||c - v1||, angle = 2*arcsin(d/2) is more stable for small angles
            d = np.linalg.norm(c - v1)
            radius = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
            
            if radius > max_radius:
                max_radius = radius
                worst_simplex = simplex.copy()
                worst_center = c
                
        if worst_center is None:
            return np.pi, None, None
            
        return max_radius, worst_simplex, worst_center

    # 1. Initialization
    current_points = fibonacci_sphere(N).copy()
    current_radius, _, _ = get_max_circumradius(current_points)
    
    best_points = current_points.copy()
    best_radius = current_radius
    
    # 2. Simulated Annealing on Discrete Permutations
    # Strategy: 
    # - Identify the worst facet (triangle) with the largest circumradius.
    # - Propose a move: randomly select one vertex from the worst facet and swap it with a random other point.
    # - This changes the topology of the hull.
    # - Accept based on Metropolis criterion to escape local minima.
    
    T = 0.05  # Initial temperature
    T_min = 0.001
    cooling_rate = 0.999
    
    # Pre-compute a list of all indices for random selection
    all_indices = list(range(N))
    
    iterations = 0
    max_iterations = 100000  # Cap iterations, but time is the real limit
    
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        
        # Get current worst facet
        _, worst_simplex, _ = get_max_circumradius(current_points)
        
        if worst_simplex is None:
            break
            
        # Proposal: Pick a vertex from the worst simplex
        # and swap it with a random vertex NOT in the worst simplex
        # (or any vertex, but swapping within the simplex doesn't change the set of points, just indices? 
        #  No, swapping indices of points doesn't change the geometry. 
        #  We need to CHANGE the position of a point or SWAP the identity of points?
        #  Wait, the problem is "Place 50 points". The points are indistinguishable.
        #  Swapping indices of the array does nothing to the geometry.
        #  We need to MOVE a point.
        
        # Revised Strategy for Discrete Search:
        # "Discrete" here is tricky because the space is continuous.
        # However, we can use a "Jump" strategy:
        # 1. Identify worst facet.
        # 2. Randomly select one vertex of the worst facet.
        # 3. Move that vertex to a RANDOM position on the sphere.
        # 4. This is a very aggressive move that can jump out of local minima.
        
        # Actually, pure random restarts are slow. 
        # Let's use a "Perturbed Local Search" with large jumps for the worst points.
        
        # Select a vertex from the worst simplex to perturb
        idx_to_move = random.choice(worst_simplex)
        
        # Save old position
        old_pos = current_points[idx_to_move].copy()
        
        # Generate a new random position on the sphere
        # Use Gaussian distribution normalized
        new_vec = np.random.randn(3)
        new_vec /= np.linalg.norm(new_vec)
        
        # Apply the move
        current_points[idx_to_move] = new_vec
        
        # Evaluate new state
        new_radius, _, _ = get_max_circumradius(current_points)
        
        # Metropolis Criterion
        delta = new_radius - current_radius
        if delta < 0 or random.random() < np.exp(-delta / T):
            # Accept
            current_radius = new_radius
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = current_points.copy()
        else:
            # Reject
            current_points[idx_to_move] = old_pos
            
        # Cool down
        T = max(T * cooling_rate, T_min)
        
        # If temperature is too low and we are stuck, reheat slightly
        # Check if we've made progress recently? 
        # Simple heuristic: if no improvement in last 1000 steps, reheat
        if iterations % 1000 == 0 and best_radius > 0.32: # ~18.3 degrees is ~0.32 rad
             # Check if we are stuck
             if current_radius > best_radius * 1.01:
                 T = T * 1.5
                 if T > 0.05: T = 0.05

    # 3. Final Polish: Local Gradient Descent
    # Now that we have a good topology (hopefully), refine positions.
    if time.time() < deadline - 10:
        current_points = best_points.copy()
        current_radius = best_radius
        
        step_size = 0.01
        for _ in range(5000):
            if time.time() > deadline - 5:
                break
            
            radius, worst_simplex, worst_center = get_max_circumradius(current_points)
            
            if radius < best_radius - 1e-9:
                best_radius = radius
                best_points = current_points.copy()
                
            if worst_simplex is None:
                break
                
            # Move vertices of the worst simplex towards the circumcenter
            moved = False
            for idx in worst_simplex:
                v = current_points[idx]
                dir_vec = worst_center - v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir < 1e-10:
                    continue
                dir_vec /= norm_dir
                
                new_v = v + step_size * dir_vec
                new_v /= np.linalg.norm(new_v)
                current_points[idx] = new_v
                moved = True
            
            if not moved:
                break
                
            # Decay step size
            step_size *= 0.99
            if step_size < 1e-5:
                break

    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
