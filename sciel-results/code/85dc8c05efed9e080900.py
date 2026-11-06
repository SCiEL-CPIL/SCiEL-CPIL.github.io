import numpy as np
import time
import sys
from scipy.spatial import ConvexHull

N = 130

def _compute_covering_radius(points):
    """
    Computes the covering radius in radians.
    The covering radius is the maximum of the circumradii of the facets of the convex hull.
    Each facet defines a spherical cap. The center of the cap is the circumcenter of the facet triangle.
    The angular distance from the center to any vertex of the facet is the circumradius.
    """
    try:
        hull = ConvexHull(points)
        # Facets are triangles
        facets = hull.simplices
        # For each facet, compute the circumradius
        # The circumcenter of a triangle on the unit sphere is the normalized sum of the vertices
        # (if the triangle is acute and small enough, or generally the projection of the origin onto the plane)
        # Actually, for a spherical triangle, the "center" of the cap is the point on the sphere
        # equidistant from the 3 vertices. This is the normalized vector sum of the 3 vertices,
        # provided the triangle does not contain the antipode of the origin (which it doesn't for small caps).
        
        max_circumradius = 0.0
        for i, j, k in facets:
            p_i = points[i]
            p_j = points[j]
            p_k = points[k]
            
            # Circumcenter direction
            c = p_i + p_j + p_k
            norm_c = np.linalg.norm(c)
            if norm_c < 1e-10:
                # Degenerate case, unlikely for distinct points
                continue
            c = c / norm_c
            
            # Circumradius is the angle between c and any of the vertices, e.g., p_i
            # cos(theta) = c . p_i
            dot = np.dot(c, p_i)
            dot = np.clip(dot, -1.0, 1.0)
            theta = np.arccos(dot)
            
            if theta > max_circumradius:
                max_circumradius = theta
                
        return max_circumradius
    except Exception:
        return np.pi # Invalid or degenerate

def construct():
    """
    Constructs a set of N points on the unit sphere to minimize the covering radius.
    Strategy: Stochastic Local Search (Simulated Annealing) targeting the worst facet.
    1. Initialize with a Fibonacci spiral.
    2. Compute convex hull and identify the facet with the largest circumradius.
    3. Perturb the vertices of that facet (and potentially neighbors) to reduce that specific gap.
    4. Accept moves if they improve the global covering radius (or with probability based on temperature).
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s kill

    # 1. Initialization: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms

    current_radius = _compute_covering_radius(points)
    best_points = points.copy()
    best_radius = current_radius
    
    # Temperature for simulated annealing
    # We want to accept some worse moves to escape local minima, but mostly improve.
    T = 1e-3
    T_min = 1e-8
    cooling_rate = 0.999
    
    rng = np.random.default_rng(42)
    
    # We will cache the hull to avoid recomputing it every single step if we only move 1-2 points?
    # ConvexHull is O(N log N) or O(N^2). For N=130, it's fast enough to recompute frequently.
    # But we need the *worst* facet.
    
    def get_worst_facet(pts):
        try:
            hull = ConvexHull(pts)
            facets = hull.simplices
            max_cr = 0.0
            worst_idx = -1
            for idx, (i, j, k) in enumerate(facets):
                p_i = pts[i]
                p_j = pts[j]
                p_k = pts[k]
                c = p_i + p_j + p_k
                norm_c = np.linalg.norm(c)
                if norm_c < 1e-10:
                    continue
                c = c / norm_c
                dot = np.clip(np.dot(c, p_i), -1.0, 1.0)
                theta = np.arccos(dot)
                if theta > max_cr:
                    max_cr = theta
                    worst_idx = idx
            return worst_idx, max_cr, hull
        except Exception:
            return -1, np.pi, None

    current_worst_idx, current_cr, _ = get_worst_facet(points)
    
    while time.time() - start_time < time_limit:
        # Identify the worst facet
        worst_idx, current_cr, hull = get_worst_facet(points)
        
        if worst_idx == -1:
            break
            
        # Update best if improved
        if current_cr < best_radius:
            best_radius = current_cr
            best_points = points.copy()
            
        # Get the vertices of the worst facet
        # We need to map the facet index to the actual vertex indices
        # scipy.spatial.ConvexHull.simplices gives indices directly
        i, j, k = hull.simplices[worst_idx]
        
        # Strategy: Move one of the vertices (i, j, or k) towards the "hole" center.
        # The hole center is the circumcenter c.
        # Moving a vertex towards c reduces the circumradius of that facet.
        # However, it might increase the circumradius of adjacent facets.
        
        # Choose a random vertex from the worst facet to move
        vertex_to_move = rng.choice([i, j, k])
        
        # Compute the circumcenter of the worst facet to define the "pull" direction
        p_i = points[i]
        p_j = points[j]
        p_k = points[k]
        c = p_i + p_j + p_k
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-10:
            c = points[vertex_to_move] # Fallback
        else:
            c = c / norm_c
            
        # Perturbation: Move vertex_to_move towards c by a small random amount
        # Also add some random noise to escape local minima
        step_size = 0.01 * (1.0 + T)
        noise = rng.normal(0, step_size * 0.5, 3)
        
        # Direction is (c - p_v) projected to tangent plane
        p_v = points[vertex_to_move]
        dir_vec = c - p_v
        # Project to tangent plane
        dot = np.dot(dir_vec, p_v)
        dir_tangent = dir_vec - dot * p_v
        norm_dir = np.linalg.norm(dir_tangent)
        if norm_dir > 1e-10:
            dir_tangent = dir_tangent / norm_dir
        
        # New position
        new_p_v = p_v + step_size * dir_tangent + noise
        
        # Renormalize
        new_p_v = new_p_v / np.linalg.norm(new_p_v)
        
        # Apply move
        points[vertex_to_move] = new_p_v
        
        # Evaluate new covering radius
        new_worst_idx, new_cr, _ = get_worst_facet(points)
        
        # Acceptance criterion
        if new_cr < current_cr:
            # Always accept improvement
            current_cr = new_cr
        else:
            # Accept worse move with probability exp(-(delta)/T)
            delta = new_cr - current_cr
            if delta > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_cr = new_cr
                else:
                    # Reject: revert the move
                    points[vertex_to_move] = p_v
            
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Optional: If we stagnate for a long time, we might need a bigger kick, 
        # but the time limit is the primary constraint.
        
    # Final check and return
    final_radius = _compute_covering_radius(best_points)
    if final_radius < best_radius:
        best_points = best_points # Already best
    
    # Ensure normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Shape: {pts.shape}")
    print(f"Norms: {np.linalg.norm(pts, axis=1)[:5]}")
    print(f"Covering Radius: {_compute_covering_radius(pts) * 180 / np.pi:.6f} degrees")
