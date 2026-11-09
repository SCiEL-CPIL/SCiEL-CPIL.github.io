import numpy as np
import time
from scipy.optimize import minimize

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    # Unit hexagon vertices (flat-top, side=1, circumradius=1)
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    hex_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    def get_hex_verts(cx, cy, angle_deg=0.0):
        theta = angle_deg * np.pi / 180.0
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return (rot @ hex_verts.T).T + np.array([cx, cy])
    
    def min_outer_side_for_config(positions, inner_angles, outer_center, outer_angle_deg=0.0):
        """Compute minimum side length of outer hexagon to contain all inner hexagons."""
        theta = outer_angle_deg * np.pi / 180.0
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        
        # Edge normals in outer hexagon's local frame (pointing to edge midpoints)
        edge_angles_local = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
        normals_local = np.column_stack([np.cos(edge_angles_local), np.sin(edge_angles_local)])
        
        max_proj = 0.0
        for i in range(len(positions)):
            cx, cy = positions[i]
            angle_i = inner_angles[i]
            verts = get_hex_verts(cx, cy, angle_i)
            for vx, vy in verts:
                dx = vx - outer_center[0]
                dy = vy - outer_center[1]
                local = rot.T @ np.array([dx, dy])
                projections = normals_local @ local
                max_proj = max(max_proj, float(np.max(projections)))
        
        return 2.0 * max_proj / np.sqrt(3)
    
    def check_no_overlap(positions, inner_angles, tol=0.001):
        """Check that no two inner hexagons overlap."""
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                # Get vertices of both hexagons
                vi = get_hex_verts(positions[i][0], positions[i][1], inner_angles[i])
                vj = get_hex_verts(positions[j][0], positions[j][1], inner_angles[j])
                
                # Check if any vertex of i is inside j or vice versa
                # Simple check: if centers are too close
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                # For same orientation, min non-overlapping distance is sqrt(3)
                # For different orientations, it can be less
                # Use a conservative check: if dist < 1.5, do detailed check
                if dist < 1.7:
                    # Check if any vertex of hex i is inside hex j
                    for vx, vy in vi:
                        # Transform to hex j's local frame
                        dx2 = vx - positions[j][0]
                        dy2 = vy - positions[j][1]
                        theta = -inner_angles[j] * np.pi / 180.0
                        rot2 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                        lx, ly = rot2 @ np.array([dx2, dy2])
                        # Check if in unit hexagon
                        edge_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
                        normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
                        inradius = np.sqrt(3) / 2.0
                        if np.all(normals @ np.array([lx, ly]) <= inradius + tol):
                            return False
        return True
    
    # Initial arrangement: hexagonal lattice
    d = np.sqrt(3)  # center-to-center distance for edge-sharing
    
    # 11 hexagons in a compact pattern
    # Let me use: center + 6 around + 4 more
    positions = np.array([
        [0.0, 0.0],                          # 0: center
        [d, 0.0],                            # 1
        [d/2, d*3/4],                        # 2
        [-d/2, d*3/4],                       # 3
        [-d, 0.0],                           # 4
        [-d/2, -d*3/4],                      # 5
        [d/2, -d*3/4],                       # 6
        # 4 more: extend the hexagon
        [2*d, 0.0],                          # 7
        [d, d*3/4],                          # 8  -- wait, this might overlap
        [-d, d*3/4],                         # 9
        [0.0, d*3/4*2],                      # 10
    ])
    
    # Actually, let me be more careful. In a hex grid with spacing d:
    # Neighbors of (0,0) are at angles 0,60,120,180,240,300 at distance d
    # The 7-hexagon cluster has extent: from -d to d in x, -d*3/4 to d*3/4 in y (roughly)
    # To add 4 more, I can extend in 4 directions
    
    # Let me try a different initial config: a 3-4-3-1 row pattern or similar
    # Or just optimize from the 7-cluster + 4 extensions
    
    inner_angles = np.zeros(N)
    
    best_side = float('inf')
    best_pos = positions.copy()
    best_outer_c = np.array([0.0, 0.0])
    best_outer_angle = 0.0
    
    # Objective function
    def objective(params):
        pos = params[:22].reshape(11, 2)
        ocx = params[22]
        ocy = params[23]
        outer_angle_deg = params[24]
        
        side = min_outer_side_for_config(pos, inner_angles, (ocx, ocy), outer_angle_deg)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < np.sqrt(3) - 0.01:
                    penalty += (np.sqrt(3) - dist) ** 2
        
        return side + 5.0 * penalty
    
    x0 = np.concatenate([positions.flatten(), [0.0, 0.0, 0.0]])
    
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7})
    
    if result.fun < best_side:
        pos = result.x[:22].reshape(11, 2)
        ocx, ocy = result.x[22], result.x[23]
        outer_angle_deg = result.x[24]
        
        if check_no_overlap(pos, inner_angles):
            side = min_outer_side_for_config(pos, inner_angles, (ocx, ocy), outer_angle_deg)
            if side < best_side:
                best_side = side
                best_pos = pos
                best_outer_c = np.array([ocx, ocy])
                best_outer_angle = outer_angle_deg
    
    inner = [[best_pos[i, 0], best_pos[i, 1], 0.0] for i in range(N)]
    return inner, tuple(best_outer_c), best_side, best_outer_angle
