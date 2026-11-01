import numpy as np
import time
import random

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 30.0  # 30 seconds for optimization
    
    # Initial configuration
    # Hexagon vertices relative to center
    def get_hex_vertices(cx, cy, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        angles = angle_rad + np.arange(6) * np.pi / 3
        xs = cx + np.cos(angles)
        ys = cy + np.sin(angles)
        return np.column_stack((xs, ys))

    # Objective function: returns the required outer side length L
    def calc_L(positions):
        # positions: (N, 3) array [x, y, angle_deg]
        max_proj = 0.0
        # Normals for flat-top hexagon (normals at 0, 60, 120, 180, 240, 300)
        # Actually, if normals are at 0, 60..., the hexagon is "pointy" in x?
        # Let's check. Normal at 0 is (1,0). Side is vertical? No, side is perpendicular to normal.
        # If normal is (1,0), side is vertical line x = R.
        # This corresponds to a hexagon with vertical sides? No, horizontal top/bottom?
        # If normal is (1,0), the boundary is x = R.
        # The vertices would be at 30, 90, 150...
        # This is a "flat-top" hexagon if we consider the top side horizontal?
        # Wait. Normal (1,0) -> Side is vertical.
        # Normal (0,1) -> Side is horizontal.
        # If normals are at 0, 60, 120, 180, 240, 300.
        # Sides are at angles 30, 90, 150...
        # Vertices are at 0, 60, 120...
        # This is a "pointy-top" hexagon (vertex at top).
        
        for i in range(N):
            x, y, ang = positions[i]
            verts = get_hex_vertices(x, y, ang)
            # Project all vertices onto the 6 normal directions
            # Normals: cos(k*pi/3), sin(k*pi/3) for k=0..5
            for k in range(6):
                nx = np.cos(k * np.pi / 3)
                ny = np.sin(k * np.pi / 3)
                projections = verts[:, 0] * nx + verts[:, 1] * ny
                # We need max absolute projection
                m = np.max(np.abs(projections))
                if m > max_proj:
                    max_proj = m
        
        # R_outer = max_proj
        # L = 2 * R_outer / sqrt(3)
        return 2.0 * max_proj / np.sqrt(3.0)

    # Simple Local Search
    # State: list of [x, y, angle]
    # Initial guess: 
    # Center (0,0)
    # 6 neighbors
    # 4 more
    
    pos = []
    pos.append([0.0, 0.0, 0.0])
    dist = np.sqrt(3.0) # Distance between centers of touching hexes
    for k in range(6):
        ang = k * 60.0
        x = dist * np.cos(np.deg2rad(ang))
        y = dist * np.sin(np.deg2rad(ang))
        pos.append([x, y, ang])
    
    # 4 more. Place them further out.
    # Maybe at the "corners" of the 7-cluster?
    # The 7-cluster has vertices at dist + 1.
    # Let's place them at distance dist + 1.5 or so, aligned with gaps.
    # Gaps are at 30, 90, 150...
    # Let's try placing 4 of them at 0, 60, 120, 180 but further out?
    # Or just randomize a bit and let search fix it.
    for k in range(4):
        ang = (k * 90.0) + 30.0 # Start at gaps
        # Push out
        d = dist * 1.8
        x = d * np.cos(np.deg2rad(ang))
        y = d * np.sin(np.deg2rad(ang))
        pos.append([x, y, ang])
        
    pos = np.array(pos)
    
    best_L = calc_L(pos)
    best_pos = pos.copy()
    
    # Local Search
    step_size = 0.5
    angle_step = 10.0
    
    while time.time() < deadline:
        improved = False
        for i in range(N):
            # Try moving x, y, angle
            for delta_x in [-step_size, 0, step_size]:
                for delta_y in [-step_size, 0, step_size]:
                    for delta_a in [-angle_step, 0, angle_step]:
                        if delta_x == 0 and delta_y == 0 and delta_a == 0:
                            continue
                        
                        new_pos = best_pos.copy()
                        new_pos[i, 0] += delta_x
                        new_pos[i, 1] += delta_y
                        new_pos[i, 2] += delta_a
                        
                        L = calc_L(new_pos)
                        if L < best_L:
                            best_L = L
                            best_pos = new_pos
                            improved = True
                            
        if not improved:
            # Reduce step size
            step_size *= 0.9
            angle_step *= 0.9
            if step_size < 0.01:
                break

    # Convert best_pos to list of lists
    inner = []
    for p in best_pos:
        inner.append([float(p[0]), float(p[1]), float(p[2])])
        
    return inner, (0.0, 0.0), float(best_L), 0.0

# EVOLVE-BLOCK-END
