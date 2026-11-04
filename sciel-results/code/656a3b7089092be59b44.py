import numpy as np
import time
from scipy.optimize import minimize

N = 12

def _hex_verts(center, side, angle):
    """Return vertices of a regular hexagon."""
    cx, cy = center
    r = angle * np.pi / 180.0
    verts = []
    for k in range(6):
        a = r + k * np.pi / 3.0
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def _check_containment(inner_verts, outer_center, outer_side, outer_angle):
    """
    Check if all inner vertices are inside the outer hexagon.
    Returns the maximum violation (0 if contained).
    """
    ocx, ocy = outer_center
    r = outer_angle * np.pi / 180.0
    apothem = np.sqrt(3) / 2 * outer_side
    
    # Normals to the 6 sides of the outer hexagon
    # The sides are perpendicular to the directions r + k*pi/3
    # Wait, the vertices are at r + k*pi/3.
    # The sides are between vertices.
    # The normal to the side between vertex k and k+1 is at angle r + (k+0.5)*pi/3.
    # Let's verify.
    # Vertex 0: (1, 0). Vertex 1: (0.5, sqrt(3)/2).
    # Midpoint: (0.75, sqrt(3)/4).
    # Vector from center to midpoint: (0.75, sqrt(3)/4).
    # Angle: atan2(sqrt(3)/4, 0.75) = atan2(sqrt(3), 3) = 30 deg = pi/6.
    # Yes, normals are at r + pi/6 + k*pi/3.
    
    max_viol = 0.0
    for k in range(6):
        normal_angle = r + np.pi / 6.0 + k * np.pi / 3.0
        nx = np.cos(normal_angle)
        ny = np.sin(normal_angle)
        
        # Project all inner vertices onto this normal
        # Vector from outer center to inner vertex
        # P - C
        # Dot product with normal
        # Distance from center along normal = nx * (px - cx) + ny * (py - cy)
        # This must be <= apothem.
        
        # Vectorized for all vertices
        # inner_verts is shape (M, 2)
        # proj = inner_verts[:, 0] * nx + inner_verts[:, 1] * ny
        # shift = ocx * nx + ocy * ny
        # dists = proj - shift
        # max_dist = np.max(dists)
        # viol = max_dist - apothem
        # if viol > max_viol: max_viol = viol
        
        proj = inner_verts[:, 0] * nx + inner_verts[:, 1] * ny
        shift = ocx * nx + ocy * ny
        dists = proj - shift
        max_dist = np.max(dists)
        viol = max_dist - apothem
        if viol > max_viol:
            max_viol = viol
            
    return max_viol

def _check_overlap(inner_positions, inner_angles, inner_side=1.0):
    """
    Check for overlaps between inner hexagons.
    Since we assume aligned hexagons for simplicity in overlap check (distance >= 2),
    this is fast. If we allow rotation, this is an approximation.
    For now, let's assume aligned inner hexagons to make the problem tractable.
    """
    max_viol = 0.0
    n = len(inner_positions)
    for i in range(n):
        for j in range(i + 1, n):
            pi = inner_positions[i]
            pj = inner_positions[j]
            dist = np.linalg.norm(pi - pj)
            # Minimum distance between centers of non-overlapping unit hexagons is 2.0
            viol = 2.0 - dist
            if viol > max_viol:
                max_viol = viol
    return max_viol

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_score = np.inf
    best_result = None
    
    # Initial configurations
    configs = []
    
    # 1. 3x4 Grid
    pos = []
    for j in range(4):
        for i in range(3):
            x = i * 2.0 + (j % 2) * 1.0
            y = j * np.sqrt(3)
            pos.append([x, y])
    # Center the grid
    pos = np.array(pos)
    pos -= np.mean(pos, axis=0)
    configs.append((pos, np.zeros(N), 0.0, 0.0, 0.0, 5.0)) # pos, angles, cx, cy, angle, side
    
    # 2. Ring + Center + 5
    pos = [[0,0]]
    for k in range(6):
        a = k * np.pi / 3.0
        pos.append([2.0 * np.cos(a), 2.0 * np.sin(a)])
    # Add 5 more in gaps?
    # Gaps are at distance sqrt(3) from center? No.
    # Let's just add them radially outward
    for k in range(5):
        a = k * 2 * np.pi / 5.0
        pos.append([4.0 * np.cos(a), 4.0 * np.sin(a)])
    pos = np.array(pos)
    pos -= np.mean(pos, axis=0)
    configs.append((pos, np.zeros(N), 0.0, 0.0, 0.0, 5.0))

    # 3. Random
    for _ in range(5):
        pos = np.random.uniform(-5, 5, (N, 2))
        configs.append((pos, np.zeros(N), 0.0, 0.0, 0.0, 8.0))

    def penalty_func(x):
        # x: [x0, y0, ..., x11, y11, cx, cy, angle, side]
        pos = x[:2*N].reshape(N, 2)
        cx = x[2*N]
        cy = x[2*N+1]
        angle = x[2*N+2]
        side = x[2*N+3]
        
        # Penalty for side < 1
        if side < 1.0:
            side = 1.0
        
        # Calculate vertices of all inner hexagons (assuming angle 0 for all)
        # This is an approximation if we allowed rotation, but we fixed angles to 0
        # in the variable set for now.
        # If we want to allow rotation, we need more vars.
        # Let's stick to aligned inner hexagons for speed and simplicity.
        
        verts_list = []
        for i in range(N):
            v = _hex_verts(pos[i], 1.0, 0.0)
            verts_list.append(v)
        all_verts = np.vstack(verts_list)
        
        # Containment violation
        v_cont = _check_containment(all_verts, (cx, cy), side, angle)
        
        # Overlap violation
        v_ovl = _check_overlap(pos, np.zeros(N))
        
        # Total penalty
        pen = 0.0
        if v_cont > 0:
            pen += v_cont ** 2
        if v_ovl > 0:
            pen += v_ovl ** 2
            
        return side + 100.0 * pen

    for cfg in configs:
        if time.time() > deadline:
            break
            
        pos_init, ang_init, cx_init, cy_init, ang_out_init, side_init = cfg
        x0 = np.concatenate([pos_init.flatten(), [cx_init, cy_init, ang_out_init, side_init]])
        
        # Run optimization
        try:
            res = minimize(penalty_func, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            
            if res.success or res.fun < best_score:
                # Extract solution
                sol_pos = res.x[:2*N].reshape(N, 2)
                sol_cx = res.x[2*N]
                sol_cy = res.x[2*N+1]
                sol_angle = res.x[2*N+2]
                sol_side = res.x[2*N+3]
                
                # Verify solution
                verts_list = []
                for i in range(N):
                    v = _hex_verts(sol_pos[i], 1.0, 0.0)
                    verts_list.append(v)
                all_verts = np.vstack(verts_list)
                
                v_cont = _check_containment(all_verts, (sol_cx, sol_cy), sol_side, sol_angle)
                v_ovl = _check_overlap(sol_pos, np.zeros(N))
                
                if v_cont <= 1e-6 and v_ovl <= 1e-6:
                    if sol_side < best_score:
                        best_score = sol_side
                        best_result = (sol_pos, (sol_cx, sol_cy), sol_side, sol_angle)
                        
        except Exception as e:
            pass

    if best_result is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner_pos, center, side, angle = best_result
    inner = [[p[0], p[1], 0.0] for p in inner_pos]
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END
