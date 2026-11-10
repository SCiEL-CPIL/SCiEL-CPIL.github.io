import numpy as np
import time

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    def hex_vertices(cx, cy, d, angle_deg):
        """Compute vertices of a regular hexagon with center (cx,cy), side d, angle a."""
        a = np.radians(angle_deg)
        angles = a + np.radians(60) * np.arange(6)
        return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])
    
    def compute_min_outer_side(all_vertices, outer_angle_deg):
        """Compute the minimum outer hexagon side length needed to contain all vertices.
        
        The outer hexagon has center (0,0), side s, and angle outer_angle_deg.
        We need to find the smallest s such that all vertices are inside.
        
        For a regular hexagon with center (0,0), side s, angle a:
        - The 6 outward normals are at angles (a + 30 + 60*i) degrees for i=0..5
        - The apothem (distance from center to edge midpoint) is s * sqrt(3) / 2
        - For a point p to be inside, we need: p . n_i <= s * sqrt(3) / 2 for all i
        - So s >= 2/sqrt(3) * max_i (max_j (v_j . n_i))
        """
        a = np.radians(outer_angle_deg)
        normal_angles = a + np.radians(30 + 60 * np.arange(6))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_s = 0.0
        for normal in normals:
            projections = all_vertices @ normal
            max_proj = np.max(projections)
            min_s_for_this_normal = 2.0 / np.sqrt(3.0) * max_proj
            max_s = max(max_s, min_s_for_this_normal)
        
        return max_s
    
    # Define a fixed topological adjacency graph for 12 hexagons
    # Using a hexagonal cluster: 1 center + 6 in ring 1 + 5 in ring 2
    # This gives a connected, compact cluster
    
    # For unit hexagons in a perfect tiling, the distance between centers of adjacent
    # hexagons is 2 * apothem = sqrt(3)
    d_adj = np.sqrt(3)
    
    # Axial coordinate to cartesian for hex grid with spacing d_adj
    def axial_to_cartesian(q, r, spacing):
        x = spacing * (q + r / 2.0)
        y = spacing * (np.sqrt(3) / 2.0) * r
        return x, y
    
    # Define the 12 hexagon positions in axial coordinates
    # Center: (0,0)
    # Ring 1: 6 positions
    # Ring 2: 5 positions (chosen for compactness)
    
    axial_positions = [
        (0, 0),       # center
        (1, 0),       # ring 1
        (1, -1),
        (0, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (2, 0),       # ring 2
        (2, -1),
        (1, -2),
        (-2, 0),
        (-2, 1),
    ]
    
    # Convert to cartesian (local coordinates, before global rotation/translation)
    local_positions = np.array([axial_to_cartesian(q, r, d_adj) for q, r in axial_positions])
    
    # All local hexagons have angle 0 (aligned with the grid)
    local_angles = np.zeros(12)
    
    def objective(params):
        """Objective function: minimize the outer hexagon side length.
        
        params: [tx, ty, global_angle_deg, outer_angle_deg]
        - tx, ty: translation of the cluster
        - global_angle_deg: rotation of the entire cluster
        - outer_angle_deg: angle of the outer hexagon
        """
        tx, ty, global_angle, outer_angle = params
        
        # Rotate local positions by global_angle
        theta = np.radians(global_angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        rotated_positions = np.zeros((12, 2))
        for i in range(12):
            x, y = local_positions[i]
            rotated_positions[i, 0] = x * cos_t - y * sin_t
            rotated_positions[i, 1] = x * sin_t + y * cos_t
        
        # Translate
        rotated_positions[:, 0] += tx
        rotated_positions[:, 1] += ty
        
        # Compute all vertices of all inner hexagons
        all_vertices = []
        for i in range(12):
            cx, cy = rotated_positions[i]
            # Each inner hexagon is rotated by global_angle (rigid body)
            ang = global_angle
            verts = hex_vertices(cx, cy, 1.0, ang)
            all_vertices.append(verts)
        
        all_vertices = np.vstack(all_vertices)
        
        # Compute minimum outer side length
        return compute_min_outer_side(all_vertices, outer_angle)
    
    # Initial parameters: centered at origin, no rotation
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    
    # Try optimization using Nelder-Mead
    from scipy.optimize import minimize
    
    best_params = x0.copy()
    best_s = objective(x0)
    
    # Try multiple restarts with different initial angles
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            params_init = x0.copy()
        else:
            # Random initial angles
            params_init = np.array([
                0.0,  # tx
                0.0,  # ty
                np.random.uniform(0, 360),  # global_angle
                np.random.uniform(0, 60),  # outer_angle (hex symmetry: 60 deg)
            ])
        
        result = minimize(objective, params_init, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10,
                                  'adaptive': True})
        
        if result.fun < best_s:
            best_s = result.fun
            best_params = result.x
    
    # Extract best solution
    tx, ty, global_angle, outer_angle = best_params
    
    # Rotate and translate local positions
    theta = np.radians(global_angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    
    final_positions = np.zeros((12, 2))
    for i in range(12):
        x, y = local_positions[i]
        final_positions[i, 0] = x * cos_t - y * sin_t + tx
        final_positions[i, 1] = x * sin_t + y * cos_t + ty
    
    # All inner hexagons have the same angle (rigid body)
    final_angles = np.full(12, global_angle)
    
    # Create output arrays
    inner_hex_data = np.column_stack([final_positions, final_angles])
    
    # Outer hexagon: centered at origin, with the optimized angle
    outer_hex_data = np.array([0.0, 0.0, outer_angle])
    
    return inner_hex_data, outer_hex_data, best_s
