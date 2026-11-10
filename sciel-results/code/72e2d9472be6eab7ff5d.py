import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    deadline = time.time() + 600  # 600 second deadline
    
    def get_hex_vertices(cx, cy, angle_deg, s=1.0):
        """Get vertices of a regular hexagon with side length s."""
        angle_rad = np.deg2rad(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle_rad + i * np.pi / 3
            x = cx + s * np.cos(theta)
            y = cy + s * np.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)
    
    def hexagon_separation(verts1, verts2):
        """
        Check if two hexagons are separated by at least gap.
        Returns the minimum gap (positive if separated, negative if overlapping).
        Uses the 12 edge normals approach.
        """
        min_gap = np.inf
        
        # Get edge normals for both hexagons
        for i in range(6):
            # Edge of hex1 from vertex i to vertex i+1
            p1 = verts1[i]
            p2 = verts1[(i+1) % 6]
            edge = p2 - p1
            # Normal (pointing outward)
            normal = np.array([-edge[1], edge[0]])
            normal = normal / np.linalg.norm(normal)
            
            # Project all vertices of both hexagons onto this normal
            proj1 = np.dot(verts1, normal)
            proj2 = np.dot(verts2, normal)
            
            # The gap is the distance between the intervals
            # If proj2 is all on one side of proj1, gap = min(proj2) - max(proj1) or vice versa
            max1 = np.max(proj1)
            min1 = np.min(proj1)
            max2 = np.max(proj2)
            min2 = np.min(proj2)
            
            # Check if separated
            if min2 > max1:
                gap = min2 - max1
            elif max1 < min2:
                gap = min2 - max1
            elif min1 > max2:
                gap = min1 - max2
            elif max2 < min1:
                gap = min1 - max2
            else:
                # Overlapping on this normal
                gap = -np.inf
            
            min_gap = min(min_gap, gap)
        
        # Also check the 6 normals from hex2
        for i in range(6):
            p1 = verts2[i]
            p2 = verts2[(i+1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            normal = normal / np.linalg.norm(normal)
            
            proj1 = np.dot(verts1, normal)
            proj2 = np.dot(verts2, normal)
            
            max1 = np.max(proj1)
            min1 = np.min(proj1)
            max2 = np.max(proj2)
            min2 = np.min(proj2)
            
            if min2 > max1:
                gap = min2 - max1
            elif min1 > max2:
                gap = min1 - max2
            else:
                gap = -np.inf
            
            min_gap = min(min_gap, gap)
        
        return min_gap
    
    def outer_hex_side(points, alpha):
        """Compute the minimum side length of a hexagon with angle alpha that encloses all points."""
        axes = []
        for i in range(3):
            theta = alpha + i * np.pi / 3
            axes.append([np.cos(theta), np.sin(theta)])
        axes = np.array(axes)
        
        projections = points @ axes.T
        max_abs_proj = np.max(np.abs(projections), axis=1)
        S = max_abs_proj * 2 / np.sqrt(3)
        return np.max(S)
    
    def objective(params):
        """
        Objective: minimize the outer hexagon side length.
        params: 22 values = 11 x-coords + 11 y-coords
        All hexagons have angle 0 for simplicity (we can add angles later if needed).
        """
        if time.time() > deadline:
            return np.inf
        
        centers = params.reshape(11, 2)
        
        # Check separation
        verts = [get_hex_vertices(c[0], c[1], 0.0) for c in centers]
        
        for i in range(11):
            for j in range(i+1, 11):
                gap = hexagon_separation(verts[i], verts[j])
                if gap < 1e-6:
                    return np.inf
        
        # Compute outer hexagon
        all_vertices = np.vstack(verts)
        
        # Optimize alpha for the outer hexagon
        best_S = np.inf
        best_alpha = 0
        alphas = np.linspace(0, np.pi/3, 200)
        for a in alphas:
            S = outer_hex_side(all_vertices, a)
            if S < best_S:
                best_S = S
                best_alpha = a
        
        return best_S
    
    # Initial configuration: 7-cluster with small gaps
    # Centers of 7-cluster (touching at distance sqrt(3))
    # We'll add a small gap by scaling
    gap_factor = 1.01  # 1% gap
    
    centers = [[0, 0]]
    for i in range(6):
        theta = i * np.pi / 3
        centers.append([gap_factor * np.sqrt(3) * np.cos(theta), gap_factor * np.sqrt(3) * np.sin(theta)])
    
    # Attach 4 more at indices 0, 1, 4, 5
    attach_indices = [0, 1, 4, 5]
    for idx in attach_indices:
        base_center = np.array(centers[idx])
        theta = np.arctan2(base_center[1], base_center[0])
        new_center = gap_factor * (np.sqrt(3) + np.sqrt(3)) * np.array([np.cos(theta), np.sin(theta)])
        centers.append(new_center)
    
    # Convert to params
    params0 = np.array(centers).flatten()
    
    # Optimize
    result = minimize(objective, params0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    if time.time() > deadline - 10:
        # Use initial configuration
        centers = np.array(centers)
        S = np.inf
        all_vertices = np.vstack([get_hex_vertices(c[0], c[1], 0.0) for c in centers])
        alphas = np.linspace(0, np.pi/3, 500)
        best_alpha = 0
        for a in alphas:
            S_val = outer_hex_side(all_vertices, a)
            if S_val < S:
                S = S_val
                best_alpha = a
    else:
        centers = result.x.reshape(11, 2)
        verts = [get_hex_vertices(c[0], c[1], 0.0) for c in centers]
        all_vertices = np.vstack(verts)
        S = np.inf
        best_alpha = 0
        alphas = np.linspace(0, np.pi/3, 500)
        for a in alphas:
            S_val = outer_hex_side(all_vertices, a)
            if S_val < S:
                S = S_val
                best_alpha = a
    
    # Build output
    inner_hex_data = np.zeros((11, 3))
    for i, c in enumerate(centers):
        inner_hex_data[i, 0] = c[0]
        inner_hex_data[i, 1] = c[1]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([0, 0, np.rad2deg(best_alpha)])
    outer_hex_side_length = float(S)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
