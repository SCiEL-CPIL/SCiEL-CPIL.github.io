import numpy as np
import time

N = 11
SQRT3 = np.sqrt(3.0)

def hex_vertices(cx, cy, angle_deg):
    a = np.radians(angle_deg)
    base_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180
    angles = base_angles + a
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def point_in_hex(px, py, cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    r_in = side * SQRT3 / 2.0
    for k in range(6):
        theta = k * np.pi / 3.0
        if lx * np.cos(theta) + ly * np.sin(theta) > r_in + 1e-9:
            return False
    return True

def hexes_disjoint(p1, a1, p2, a2):
    """p1, p2 are (x,y) tuples, a1, a2 are angles in degrees."""
    v1 = hex_vertices(p1[0], p1[1], a1)
    v2 = hex_vertices(p2[0], p2[1], a2)
    for v in v1:
        if point_in_hex(v[0], v[1], p2[0], p2[1], 1.0, a2):
            return False
    for v in v2:
        if point_in_hex(v[0], v[1], p1[0], p1[1], 1.0, a1):
            return False
    return True

def min_enclosing_hex(positions, angles, outer_angle=0.0):
    """positions: list of (x,y), angles: list of angle_deg. Returns (side, center)."""
    all_verts = []
    for pos, ang in zip(positions, angles):
        v = hex_vertices(pos[0], pos[1], ang)
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    cx = np.mean(all_verts[:, 0])
    cy = np.mean(all_verts[:, 1])
    centered = all_verts - np.array([cx, cy])
    
    a = np.radians(outer_angle)
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * centered[:, 0] - sa * centered[:, 1]
    ly = sa * centered[:, 0] + ca * centered[:, 1]
    
    r_in_needed = 0.0
    for k in range(6):
        theta = k * np.pi / 3.0
        nx, ny = np.cos(theta), np.sin(theta)
        dots = np.abs(lx * nx + ly * ny)
        r_in_needed = max(r_in_needed, float(np.max(dots)))
    
    side = r_in_needed * 2.0 / SQRT3
    return side, (cx, cy)

def evaluate(positions, angles, outer_angle=0.0):
    """Returns (side, center) or (inf, None) if invalid."""
    # Check all pairs are disjoint
    for i in range(N):
        for j in range(i + 1, N):
            if not hexes_disjoint(positions[i], angles[i], positions[j], angles[j]):
                return float('inf'), None
    return min_enclosing_hex(positions, angles, outer_angle)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Generate initial configurations on triangular lattice
    # Lattice vectors for pointy-top hexagons:
    # e1 = (SQRT3, 0), e2 = (SQRT3/2, 1.5)
    
    configs = []
    
    # Config 1: Hexagonal cluster radius 2 (7) + 4 extra
    idx1 = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),
            (2,0),(0,2),(-2,0),(2,-2)]
    
    # Config 2: 3-3-3-2 arrangement
    idx2 = [(0,0),(1,0),(2,0),
            (0,1),(1,1),(2,1),
            (0,2),(1,2),(2,2),
            (0,3),(1,3)]
    
    # Config 3: More symmetric
    idx3 = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),
            (2,0),(-2,0),(0,2),(0,-2)]
    
    # Config 4: 2-3-2-3-1
    idx4 = [(0,0),(1,0),
            (0,1),(1,1),(2,1),
            (0,2),(1,2),
            (0,3),(1,3),(2,3),
            (0,4)]
    
    for idx_list in [idx1, idx2, idx3, idx4]:
        if len(idx_list) != N:
            continue
        pos = []
        for (i, j) in idx_list:
            x = i * SQRT3 + j * SQRT3 / 2.0
            y = j * 1.5
            pos.append((x, y))
        # Center
        cx = np.mean([p[0] for p in pos])
        cy = np.mean([p[1] for p in pos])
        pos = [(p[0]-cx, p[1]-cy) for p in pos]
        angles = [0.0] * N
        configs.append((pos, angles))
    
    best_side = float('inf')
    best_pos = None
    best_ang = None
    best_center = None
    
    for pos, ang in configs:
        side, center = evaluate(pos, ang, 0.0)
        if side < best_side:
            best_side = side
            best_pos = [p for p in pos]
            best_ang = list(ang)
            best_center = center
    
    # Local search
    if best_pos is not None:
        pos = np.array(best_pos, dtype=float)
        ang = np.array(best_ang, dtype=float)
        current_side = best_side
        
        # Adaptive step size
        step = 0.1
        no_improve = 0
        
        while time.time() < deadline and step > 0.001:
            improved = False
            for _ in range(200):
                if time.time() > deadline:
                    break
                # Random perturbation
                dpos = np.random.randn(N, 2) * step
                dang = np.random.randn(N) * step * 2.0
                
                new_pos = pos + dpos
                new_ang = ang + dang
                
                new_pos_list = [tuple(p) for p in new_pos]
                new_ang_list = list(new_ang)
                
                side, center = evaluate(new_pos_list, new_ang_list, 0.0)
                if side < current_side - 1e-12:
                    pos = new_pos
                    ang = new_ang
                    current_side = side
                    best_center = center
                    improved = True
                    no_improve = 0
                    break
            
            if not improved:
                no_improve += 1
                step *= 0.8
                if no_improve > 5:
                    break
    
    inner = [[float(pos[i,0]), float(pos[i,1]), float(ang[i])] for i in range(N)]
    return inner, (float(best_center[0]), float(best_center[1])), float(current_side), 0.0
