import numpy as np
from scipy.optimize import minimize
import math

def hexagon_packing_11():
    # Constants
    N = 11
    target_side = math.sqrt(11)
    
    # Initial guess: 7-hex cluster + 4 extra
    # Axial coordinates to Cartesian
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r/2)
        y = 1.5 * r
        return x, y

    # 7-hex cluster centers
    centers = [
        (0,0),
        (1,0), (-1,0),
        (0,1), (0,-1),
        (1,-1), (-1,1)
    ]
    # Add 4 more, initially placed far away or on the boundary
    # Let's place them at (2,0), (-2,0), (1,1), (-1,-1) in axial?
    # (2,0) -> x=3.46
    # This will be large. We will optimize.
    
    # We need 11 points.
    # Let's define the variables as x,y for each of the 11 hexes.
    # Angles are fixed at 0 for simplicity (optimal for hexagonal symmetry).
    
    # Initial positions (Cartesian)
    init_pos = []
    for q,r in centers:
        init_pos.append(axial_to_cartesian(q,r))
    
    # Add 4 more
    # Place them roughly where they fit
    extra_axial = [(2,0), (-2,0), (1,1), (-1,-1)] # Note: (1,1) is not in 7-cluster
    for q,r in extra_axial:
        init_pos.append(axial_to_cartesian(q,r))
        
    x0 = np.array(init_pos).flatten()
    
    def get_bounding_side(pos):
        # pos is (11, 2)
        # Calculate vertices of all hexagons
        # Vertex i of hex j:
        # cx + cos(a + 60i), cy + sin(a + 60i)
        # a = 0
        angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
        vertices = np.zeros((11, 6, 2))
        for i in range(11):
            cx, cy = pos[i]
            for k in range(6):
                vertices[i, k, 0] = cx + np.cos(angles[k])
                vertices[i, k, 1] = cy + np.sin(angles[k])
        
        # Flatten vertices
        all_vertices = vertices.reshape(-1, 2)
        
        # Find minimal enclosing regular hexagon side
        # A regular hexagon of side S centered at origin (we can shift pos to center of mass first)
        # Let's center the points at their centroid
        centroid = np.mean(pos, axis=0)
        shifted_vertices = all_vertices - centroid
        
        # The constraint for a regular hexagon of side S centered at origin:
        # For each vertex (x,y), it must satisfy:
        # |x| <= S/2 ? No.
        # The hexagon is defined by 6 half-planes.
        # Normal angles: 0, 60, 120, 180, 240, 300
        # Distance from origin to side is S * sqrt(3) / 2
        # So for each vertex, max over k of (x*cos(n_k) + y*sin(n_k)) <= S * sqrt(3) / 2
        # where n_k are the normal angles.
        
        normals = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
        projections = np.zeros((6,))
        for k in range(6):
            proj = shifted_vertices[:, 0] * np.cos(normals[k]) + shifted_vertices[:, 1] * np.sin(normals[k])
            projections[k] = np.max(proj)
            
        max_proj = np.max(projections)
        # max_proj <= S * sqrt(3) / 2
        # S >= 2 * max_proj / sqrt(3)
        return 2 * max_proj / np.sqrt(3)
        
    def objective(x):
        pos = x.reshape(11, 2)
        side = get_bounding_side(pos)
        
        # Penalty for overlaps
        # Distance between any two centers must be >= sqrt(3)
        penalty = 0
        for i in range(11):
            for j in range(i+1, 11):
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist < np.sqrt(3) - 1e-6:
                    penalty += (np.sqrt(3) - dist) ** 2
        
        return side + penalty * 100

    # Optimize
    # Use Powell or Nelder-Mead
    res = minimize(objective, x0, method='Powell', options={'maxiter': 1000, 'xatol': 1e-6})
    
    best_pos = res.x.reshape(11, 2)
    best_side = get_bounding_side(best_pos)
    
    # Construct output
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = best_pos[i, 0]
        inner_hex_data[i, 1] = best_pos[i, 1]
        inner_hex_data[i, 2] = 0 # angle
        
    # Center the outer hexagon at the centroid
    centroid = np.mean(best_pos, axis=0)
    outer_hex_data = np.array([centroid[0], centroid[1], 0])
    
    return inner_hex_data, outer_hex_data, best_side

