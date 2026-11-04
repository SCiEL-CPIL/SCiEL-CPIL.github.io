import numpy as np
import time
from scipy.optimize import minimize

N = 11
MIN_DIST = np.sqrt(3)

def construct():
    # Normal vectors for the 6 sides of a regular hexagon (flat-top orientation)
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    def get_enclosing_hexagon_side(points):
        """Compute the minimum side length of a regular hexagon enclosing the given points."""
        # Center the points
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # For each of the 6 orientations, compute the required side length
        # The distance from center to each side is L * sqrt(3) / 2
        # So L = max_projection / (sqrt(3)/2)
        projections = np.abs(points_centered @ normals.T)
        max_proj = np.max(projections)
        L = max_proj / (np.sqrt(3) / 2)
        return L, centroid, points_centered
    
    def cost_func(params):
        # params is a 1D array of 22 elements: [x1, y1, x2, y2, ..., x11, y11]
        points = params.reshape(-1, 2)
        
        L, centroid, points_centered = get_enclosing_hexagon_side(points)
        
        # Calculate overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points_centered[i] - points_centered[j])
                if d < MIN_DIST:
                    penalty += (MIN_DIST - d) ** 2
        
        # Combine costs
        return L + 100.0 * penalty
    
    def is_valid_solution(points):
        """Check if the solution has no overlaps."""
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points_centered[i] - points_centered[j])
                if d < MIN_DIST - 1e-6:
                    return False
        return True
    
    # Random restarts to find a good local minimum
    best_L = float('inf')
    best_params = None
    
    # Generate multiple initializations
    initializations = []
    
    # Hexagonal lattice initialization
    pts = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = 1.5 * q
            y = np.sqrt(3)/2 * q + np.sqrt(3) * r
            pts.append([x, y])
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    indices = np.argsort(dists)[:N]
    init_pts = pts[indices].copy()
    initializations.append(init_pts)
    
    # Add noisy versions
    for noise_level in [0.05, 0.1, 0.2]:
        init_noisy = init_pts + np.random.normal(0, noise_level, init_pts.shape)
        initializations.append(init_noisy)
    
    # Random initializations
    for _ in range(10):
        init_random = np.random.uniform(-3, 3, (N, 2))
        initializations.append(init_random)
    
    # Optimize each initialization
    for init_pts in initializations:
        initial_params = init_pts.flatten()
        
        # Use Nelder-Mead for optimization
        res = minimize(cost_func, initial_params, method='Nelder-Mead', 
                       options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5})
        
        final_pts = res.x.reshape(-1, 2)
        
        # Check for overlaps
        if not is_valid_solution(final_pts):
            continue
        
        # Calculate the enclosing hexagon side length
        L, centroid, points_centered = get_enclosing_hexagon_side(final_pts)
        
        if L < best_L:
            best_L = L
            best_params = final_pts
    
    # If no valid solution found, fallback to a safe large solution
    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Construct the output
    inner_pts = best_params
    inner = [[float(x), float(y), 0.0] for x, y in inner_pts]
    
    # The outer hexagon side length needs to account for the fact that inner hexagons
    # have side length 1, so their vertices extend by 1 from their centers
    # The enclosing hexagon must contain the vertices of all inner hexagons
    # For a unit hexagon centered at origin with flat-top orientation, the vertices are at distance 1 from center
    # So the enclosing hexagon side length should be best_L + 1 (approximately)
    # More precisely, we need to compute the minimum enclosing hexagon of the union of all inner hexagon vertices
    
    # Compute all vertices of all inner hexagons
    all_vertices = []
    for cx, cy in inner_pts:
        # Vertices of a unit hexagon centered at (cx, cy) with flat-top orientation
        for k in range(6):
            angle = k * np.pi / 3
            vx = cx + np.cos(angle)
            vy = cy + np.sin(angle)
            all_vertices.append([vx, vy])
    
    all_vertices = np.array(all_vertices)
    
    # Compute the minimum enclosing regular hexagon for all vertices
    # Try all 6 orientations and find the minimum
    min_side = float('inf')
    best_center = np.array([0.0, 0.0])
    
    for orientation in range(6):
        # Rotate the normals by orientation * 60 degrees
        rot_angle = orientation * np.pi / 3
        rot_matrix = np.array([[np.cos(rot_angle), -np.sin(rot_angle)], 
                               [np.sin(rot_angle), np.cos(rot_angle)]])
        rotated_normals = normals @ rot_matrix.T
        
        # For each orientation, find the center that minimizes the max projection
        # This is equivalent to finding the minimum width in each of the 6 directions
        # and centering accordingly
        
        # Compute projections onto each normal
        projections = all_vertices @ rotated_normals.T
        
        # For each pair of opposite normals, find the center that minimizes the range
        # The center should be at the midpoint of the min and max projections
        centers = []
        max_widths = []
        
        for k in range(3):
            n1 = rotated_normals[k]
            n2 = rotated_normals[k+3]  # Opposite normal
            
            proj1 = all_vertices @ n1
            proj2 = all_vertices @ n2
            
            # The constraint is: -L*sqrt(3)/2 <= p.n <= L*sqrt(3)/2 for all points
            # So L*sqrt(3)/2 >= max(|p.n|) for all points and normals
            # The optimal center shifts to balance the min and max projections
            
            # For a given center c, the projection of point p is (p-c).n = p.n - c.n
            # We want to minimize max_k max_i |(p_i - c).n_k|
            
            # This is a linear programming problem, but we can use a simple approach:
            # The optimal c is such that the max positive and max negative projections are balanced
            
            # Let's compute the range of projections for each normal
            for n in rotated_normals:
                proj = all_vertices @ n
                min_proj = np.min(proj)
                max_proj = np.max(proj)
                # The center projection should be at (min_proj + max_proj) / 2
                # But we need to find a single center that works for all normals
            
            # Simple approach: iterate to find the center
            c = np.mean(all_vertices, axis=0)
            for _ in range(100):
                projections = (all_vertices - c) @ rotated_normals.T
                max_pos = np.max(projections, axis=0)
                max_neg = np.min(projections, axis=0)
                
                # Update center to balance the projections
                # For each normal, if max_pos > -max_neg, shift in the negative direction
                # If max_pos < -max_neg, shift in the positive direction
                for k in range(6):
                    if max_pos[k] > -max_neg[k]:
                        c -= (max_pos[k] + max_neg[k]) / 2 * rotated_normals[k]
                    else:
                        c += (max_pos[k] + max_neg[k]) / 2 * rotated_normals[k]
                
                c /= 1.0  # No normalization needed
                
            # Compute the final max projection
            projections = (all_vertices - c) @ rotated_normals.T
            max_abs_proj = np.max(np.abs(projections))
            side_length = max_abs_proj / (np.sqrt(3) / 2)
            
            if side_length < min_side:
                min_side = side_length
                best_center = c
    
    # Return the result
    # The inner hexagons are at their positions
    # The outer hexagon is centered at best_center with side length min_side
    # We need to adjust the inner positions relative to the outer center
    # Actually, the inner positions are absolute, and the outer center is best_center
    
    return inner, (float(best_center[0]), float(best_center[1])), float(min_side), 0.0
