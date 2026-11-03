import numpy as np
from scipy.optimize import linprog
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Unit hexagon vertices (circumradius = 1)
    def hex_vertices(cx, cy, theta):
        angles = theta + np.arange(6) * np.pi / 3
        return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])
    
    # Generate candidate configurations
    configs = []
    
    # 1. Linear arrangement along x-axis
    for spacing in [np.sqrt(3), 2.0, 1.5, 1.8]:
        positions = []
        for i in range(N):
            positions.append([i * spacing, 0.0, 0.0])
        configs.append(positions)
    
    # 2. Two-row arrangement
    for dx in [np.sqrt(3), 1.5, 1.8]:
        for dy in [1.0, 1.5, 2.0]:
            positions = []
            n1 = N // 2  # 5
            n2 = N - n1  # 6
            for i in range(n1):
                positions.append([i * dx, -dy/2, 0.0])
            for i in range(n2):
                positions.append([(i - 0.5) * dx, dy/2, 0.0])
            configs.append(positions)
    
    # 3. Three-row arrangement (4-3-4)
    for dx in [np.sqrt(3), 1.5, 1.8]:
        for dy in [1.0, 1.5, 2.0]:
            positions = []
            for i in range(4):
                positions.append([i * dx, -dy, 0.0])
            for i in range(3):
                positions.append([(i + 0.5) * dx, 0.0, 0.0])
            for i in range(4):
                positions.append([i * dx, dy, 0.0])
            configs.append(positions)
    
    # 4. Hexagonal lattice pattern (center + ring)
    # Center at origin, 6 neighbors, then 10 more in second ring... too many
    # Let's try: 1 center, 6 at distance √3, then 4 more
    for r1 in [np.sqrt(3), 1.5, 1.8]:
        for r2 in [2*np.sqrt(3), 3.0, 3.5]:
            positions = [[0.0, 0.0, 0.0]]
            for k in range(6):
                theta = k * np.pi / 3
                positions.append([r1 * np.cos(theta), r1 * np.sin(theta), 0.0])
            # Add 4 more in second ring at alternating positions
            for k in range(0, 12, 3):
                if len(positions) >= N:
                    break
                theta = k * np.pi / 6 + np.pi/6
                positions.append([r2 * np.cos(theta), r2 * np.sin(theta), 0.0])
            if len(positions) == N:
                configs.append(positions)
    
    # 5. Staggered rows with various offsets
    for dx in [np.sqrt(3), 1.4, 1.6, 1.8]:
        for dy in [0.8, 1.0, 1.2, 1.5]:
            for offset in [0.0, 0.3, 0.5, 0.7]:
                positions = []
                rows = [4, 3, 4]
                y = 0.0
                for ri, count in enumerate(rows):
                    for i in range(count):
                        x = i * dx + offset * (ri % 2)
                        positions.append([x, y, 0.0])
                    y += dy
                if len(positions) == N:
                    configs.append(positions)
    
    # Remove duplicates (approximately)
    unique_configs = []
    for cfg in configs:
        cfg_arr = np.array(cfg)
        is_dup = False
        for ucfg in unique_configs:
            ucfg_arr = np.array(ucfg)
            if np.allclose(cfg_arr, ucfg_arr, atol=0.01):
                is_dup = True
                break
        if not is_dup:
            unique_configs.append(cfg)
    
    configs = unique_configs
    
    best_score = float('inf')
    best_result = None
    
    # Try various outer rotations
    outer_rotations = np.arange(0, 60, 5)  # 0, 5, 10, ..., 55 degrees
    
    for cfg in configs:
        if time.time() > deadline:
            break
        
        cfg_arr = np.array(cfg)
        # Compute all vertices of all inner hexagons
        all_vertices = []
        for pos in cfg:
            cx, cy, theta = pos
            verts = hex_vertices(cx, cy, theta)
            all_vertices.extend(verts)
        all_vertices = np.array(all_vertices)
        
        for rot_deg in outer_rotations:
            if time.time() > deadline:
                break
            
            alpha = np.radians(rot_deg)
            
            # Constraints: for each vertex v, for each face normal direction:
            # (vx - ox) * cos(alpha + k*pi/3) + (vy - oy) * sin(alpha + k*pi/3) <= s * sqrt(3)/2
            # => -vx*cos(...) - vy*sin(...) + ox*cos(...) + oy*sin(...) - s*sqrt(3)/2 <= 0
            # Rearranging: ox*cos(...) + oy*sin(...) - s*sqrt(3)/2 <= vx*cos(...) + vy*sin(...)
            
            n_verts = len(all_vertices)
            n_normals = 6
            
            # Variables: [ox, oy, s]
            # Minimize: s (index 2)
            c = np.array([0.0, 0.0, 1.0])
            
            A_ub = []
            b_ub = []
            
            for v in all_vertices:
                vx, vy = v
                for k in range(n_normals):
                    angle = alpha + k * np.pi / 3
                    ca, sa = np.cos(angle), np.sin(angle)
                    # ox*ca + oy*sa - s*sqrt(3)/2 <= vx*ca + vy*sa
                    A_ub.append([ca, sa, -np.sqrt(3)/2])
                    b_ub.append(vx * ca + vy * sa)
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            
            # s must be positive
            bounds = [(None, None), (None, None), (0.1, None)]
            
            try:
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.success:
                    s = res.x[2]
                    ox, oy = res.x[0], res.x[1]
                    if s < best_score:
                        best_score = s
                        best_result = (cfg, (ox, oy), s, rot_deg)
            except:
                pass
    
    if best_result is not None:
        cfg, center, side, angle = best_result
        inner = [[pos[0], pos[1], pos[2]] for pos in cfg]
        return inner, center, side, angle
    else:
        # Fallback: linear arrangement
        inner = [[i * np.sqrt(3), 0.0, 0.0] for i in range(N)]
        return inner, (0.0, 0.0), (N + 1) / np.sqrt(3), 0.0

