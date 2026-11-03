import numpy as np
import random
import time

def construct():
    start_time = time.time()
    deadline = 10.0  # seconds, leave plenty of margin
    
    N = 11
    
    # Generate initial hexagonal lattice positions
    # Hex lattice basis vectors for side-to-side touching (distance sqrt(3))
    # But we need to ensure no overlap, so add a small epsilon
    eps = 1e-3
    d = np.sqrt(3) + eps
    
    # Directions
    angles = [0, 60, 120, 180, 240, 300]
    vecs = []
    for a in angles:
        rad = np.deg2rad(a)
        vecs.append([d * np.cos(rad), d * np.sin(rad)])
    vecs = np.array(vecs)
    
    # Positions: center + 6 neighbors + 4 more
    # 4 more: pick 4 from the next ring or just extend the lattice
    # Let's use axial coordinates (q, r)
    # x = d * (q + r/2)
    # y = d * (sqrt(3)/2) * r
    
    def get_pos(q, r):
        x = d * (q + r / 2.0)
        y = d * (np.sqrt(3) / 2.0) * r
        return [x, y]
    
    # 11 points in a compact hex shape
    # (0,0)
    # (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1) -> 6 neighbors
    # (2,-1), (-2,1), (2,-2), (-2,2) ? No, let's just pick 4 valid ones
    # Actually, a 19-point hex has side 2 (in grid units).
    # We want 11 points.
    # Let's take the 7 central, and 4 from the 12 outer.
    outer_ring = [
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
        (-1, -1), (0, -2), (1, -2), (2, -1), (2, -2) # Wait, coords need to be consistent
    ]
    # Let's just generate a grid of points and pick the 11 closest to origin
    candidates = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            x = d * (q + r / 2.0)
            y = d * (np.sqrt(3) / 2.0) * r
            dist = np.sqrt(x*x + y*y)
            candidates.append((dist, q, r, x, y))
            
    candidates.sort(key=lambda x: x[0])
    
    # Take the 11 closest
    positions = []
    for i in range(11):
        positions.append([candidates[i][3], candidates[i][4]])
        
    positions = np.array(positions)
    
    def get_hex_vertices(center, angle_deg=0):
        # Unit hexagon vertices
        cx, cy = center
        rad = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        verts = []
        for k in range(6):
            theta = np.deg2rad(60 * k + 30) # Pointy top? Let's use flat top for consistency with container
            # Flat top vertices: (1,0), (0.5, sqrt(3)/2)...
            # Let's use standard flat-top: angle 0, 60, 120...
            theta = np.deg2rad(60 * k)
            vx = np.cos(theta)
            vy = np.sin(theta)
            # Rotate
            rx = vx * cos_a - vy * sin_a
            ry = vx * sin_a + vy * cos_a
            verts.append([cx + rx, cy + ry])
        return np.array(verts)
    
    def calc_min_side(positions, angles):
        # Get all vertices
        all_verts = []
        for i, p in enumerate(positions):
            v = get_hex_vertices(p, angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        # Calculate required S
        max_s = 0.0
        for x, y in all_verts:
            # Constraints for flat-top hexagon of side S
            # |x| <= S
            # |y| <= S * sqrt(3)/2
            # |sqrt(3)*x + y| <= S * sqrt(3)  (Wait, check slope)
            # Line: sqrt(3)x + y = S*sqrt(3)
            # So S >= (sqrt(3)x + y) / sqrt(3)
            # And S >= (-sqrt(3)x + y) / sqrt(3) ? No.
            
            # Let's derive properly.
            # Vertices of container S:
            # (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
            # Inequalities:
            # 1. x <= S  => S >= x
            # 2. -x <= S => S >= -x
            # 3. sqrt(3)*x + y <= S*sqrt(3) => S >= (sqrt(3)*x + y) / sqrt(3)
            # 4. -sqrt(3)*x + y <= S*sqrt(3) => S >= (-sqrt(3)*x + y) / sqrt(3)
            # 5. -sqrt(3)*x - y <= S*sqrt(3) => S >= (-sqrt(3)*x - y) / sqrt(3)
            # 6. sqrt(3)*x - y <= S*sqrt(3) => S >= (sqrt(3)*x - y) / sqrt(3)
            
            s1 = abs(x)
            s2 = abs(y) * 2 / np.sqrt(3) # From |y| <= S*sqrt(3)/2
            s3 = abs(np.sqrt(3)*x + y) / np.sqrt(3)
            s4 = abs(-np.sqrt(3)*x + y) / np.sqrt(3)
            # s3 and s4 cover the diagonal constraints?
            # Let's check vertex (S/2, S*sqrt(3)/2).
            # s1 = S/2.
            # s2 = S.
            # s3 = (S*sqrt(3)/2 + S*sqrt(3)/2) / sqrt(3) = S.
            # s4 = (-S*sqrt(3)/2 + S*sqrt(3)/2) / sqrt(3) = 0.
            # So max is S. Correct.
            
            max_s = max(max_s, s1, s2, s3, s4)
            
        return max_s
    
    best_s = calc_min_side(positions, [0.0]*N)
    best_positions = positions.copy()
    best_angles = [0.0]*N
    
    # Simple local search
    time_limit = 5.0
    while time.time() - start_time < time_limit:
        # Randomly perturb a few positions and angles
        new_pos = best_positions.copy()
        new_angles = best_angles.copy()
        
        for _ in range(5):
            idx = random.randint(0, N-1)
            new_pos[idx] += np.random.normal(0, 0.1, 2)
            new_angles[idx] = random.uniform(0, 60)
            
        s = calc_min_side(new_pos, new_angles)
        if s < best_s:
            best_s = s
            best_positions = new_pos
            best_angles = new_angles
            
    # Format output
    inner = []
    for i in range(N):
        inner.append([best_positions[i][0], best_positions[i][1], best_angles[i]])
        
    return inner, (0.0, 0.0), best_s, 0.0
