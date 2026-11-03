import math
import time
import random
import numpy as np

N = 12
TIME_LIMIT = 145.0  # Safety margin below 150s

# Pre-compute base hexagon vertices (side=1, angle=0)
# Vertices at angles 0, 60, 120, 180, 240, 300
_BASE_ANGLES = np.deg2rad(np.arange(0, 360, 60))
_BASE_VERTS = np.column_stack((np.cos(_BASE_ANGLES), np.sin(_BASE_ANGLES)))

def _hex_vertices_array(cx, cy, angle_deg):
    """
    Returns (6, 2) array of vertices for a hexagon.
    angle_deg is in degrees.
    """
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    R = np.array([[ca, -sa], [sa, ca]])
    return (R @ _BASE_VERTS.T).T + np.array([cx, cy])

def _check_pair_collision(verts1, verts2):
    """
    Fast SAT check for two convex hexagons represented as (6,2) arrays.
    Returns True if they collide (overlap).
    """
    def get_normals(verts):
        # verts is (6,2)
        edges = np.roll(verts, -1, axis=0) - verts
        # normals are perpendicular to edges
        # if edge is (dx, dy), normal is (-dy, dx)
        normals = np.column_stack((-edges[:, 1], edges[:, 0]))
        # Normalize normals
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths[lengths == 0] = 1.0
        return normals / lengths

    n1 = get_normals(verts1)
    n2 = get_normals(verts2)
    
    all_normals = np.vstack((n1, n2))
    
    # Project verts1 and verts2 onto all_normals
    proj1 = verts1 @ all_normals.T
    proj2 = verts2 @ all_normals.T
    
    min1 = np.min(proj1, axis=1)
    max1 = np.max(proj1, axis=1)
    min2 = np.min(proj2, axis=1)
    max2 = np.max(proj2, axis=1)
    
    # Check for separation on any axis
    separates = (max1 < min2) | (max2 < min1)
    
    if np.any(separates):
        return False
    return True

def _is_valid_fast(cfg):
    """
    cfg: list of [x, y, angle_deg]
    Returns True if all pairs are disjoint.
    """
    # Pre-compute all vertices
    verts_list = []
    for p in cfg:
        verts_list.append(_hex_vertices_array(p[0], p[1], p[2]))
    
    for i in range(N):
        xi, yi, _ = cfg[i]
        for j in range(i + 1, N):
            xj, yj, _ = cfg[j]
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx*dx + dy*dy
            # If dist > 2.0, they are definitely disjoint (circumradius is 1.0)
            if dist_sq > 4.0:
                continue
            if _check_pair_collision(verts_list[i], verts_list[j]):
                return False
    return True

def _outer_hex_side_fast(cfg, center, angle_deg):
    """
    Compute minimum side length S of outer hexagon containing all inner hexagons.
    cfg: list of [x, y, angle_deg]
    center: (cx, cy)
    angle_deg: rotation of outer hexagon
    """
    a = np.deg2rad(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    
    # 6 outward normals of the outer hexagon
    normals_angles = a + np.deg2rad(90 + 60 * np.arange(6))
    normals = np.column_stack((np.cos(normals_angles), np.sin(normals_angles)))
    
    cx, cy = center
    
    max_need = 0.0
    
    # Collect all vertices into a single array for vectorization
    all_verts = np.vstack([_hex_vertices_array(p[0], p[1], p[2]) for p in cfg])
    
    # Relative to center
    rel_verts = all_verts - np.array([cx, cy])
    
    # Projections: (N*6, 6)
    projections = rel_verts @ normals.T
    
    # Max projection for each normal
    max_projections = np.max(projections, axis=0)
    
    # The side length is determined by the max projection over all normals
    # S = max_proj / apothem
    max_proj = np.max(max_projections)
    
    return max_proj / apothem_unit

def construct():
    t0 = time.time()
    random.seed(42)
    
    def make_hex_lattice():
        """
        3x4 hexagonal lattice.
        Touching distance for aligned hexagons is 2*sqrt(3) ~ 3.464? No.
        For flat-topped hexagons (angle 0), width is 2, height is sqrt(3).
        Horizontal neighbors: distance 2.
        Vertical neighbors: distance sqrt(3).
        Diagonal neighbors: distance sqrt(3^2 + 1^2) = 2.
        Wait, let's use a standard hex grid spacing.
        Row height: sqrt(3).
        Col width: 2.
        Offset: 1.
        """
        pts = []
        row_height = math.sqrt(3)
        col_width = 2.0
        # We need 12 hexagons.
        # 3 rows of 4?
        for r in range(3):
            for c in range(4):
                x = c * col_width + (col_width / 2 if r % 2 else 0.0)
                y = r * row_height
                # Center the configuration
                pts.append([x - 3.0, y - row_height, 0.0])
        return pts

    def make_flower():
        """
        1 center, 6 around, 5 outer.
        """
        pts = []
        pts.append([0.0, 0.0, 0.0])
        # Ring 1
        R1 = 2 * math.sqrt(3)
        for i in range(6):
            angle = math.radians(60 * i + 30)
            x = R1 * math.cos(angle)
            y = R1 * math.sin(angle)
            pts.append([x, y, 0.0])
        # Ring 2
        R2 = 3.46410161514 # 2*sqrt(3)
        angles2 = [0, 72, 144, 216, 288]
        for deg in angles2:
            rad = math.radians(deg)
            x = R2 * math.cos(rad)
            y = R2 * math.sin(rad)
            pts.append([x, y, 0.0])
        return pts[:12]

    def make_ring():
        """
        12 hexagons in a ring.
        """
        pts = []
        R = 4.0
        for i in range(N):
            theta = 2 * math.pi * i / N
            x = R * math.cos(theta)
            y = R * math.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        return pts

    initializers = [
        make_hex_lattice,
        make_flower,
        make_ring,
    ]
    
    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Find best valid initial configuration
    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 60:
            break
        cfg = init_func()
        if _is_valid_fast(cfg):
            side = _outer_hex_side_fast(cfg, (0.0, 0.0), 0.0)
            if side < best_side:
                best_side = side
                best_cfg = [p[:] for p in cfg]
                best_center = (0.0, 0.0)
                best_angle = 0.0

    if best_cfg is None:
        # Fallback: loose lattice
        cfg = make_hex_lattice()
        for p in cfg:
            p[0] *= 1.5
            p[1] *= 1.5
        best_cfg = cfg
        best_side = _outer_hex_side_fast(cfg, (0.0, 0.0), 0.0)
        best_center = (0.0, 0.0)
        best_angle = 0.0

    # Simulated Annealing
    cfg = [p[:] for p in best_cfg]
    cur_score = best_side
    T = 0.5
    T_end = 0.001
    alpha = 0.998
    step_scale = 1.0
    
    # Precompute normals for quick score if needed, but we use vectorized version
    # which is fast enough.
    
    while time.time() - t0 < TIME_LIMIT - 20:
        r = random.random()
        idx = random.randrange(N)
        
        old_x, old_y, old_a = cfg[idx][0], cfg[idx][1], cfg[idx][2]
        
        if r < 0.65:
            # Move one
            cfg[idx][0] += random.gauss(0, 0.2 * step_scale)
            cfg[idx][1] += random.gauss(0, 0.2 * step_scale)
            cfg[idx][2] += random.gauss(0, 5.0 * step_scale)
        elif r < 0.85:
            # Rotate one
            cfg[idx][2] += random.gauss(0, 10.0 * step_scale)
        else:
            # Move two
            idx2 = random.randrange(N)
            if idx2 == idx:
                idx2 = (idx + 1) % N
            old_x2, old_y2, old_a2 = cfg[idx2][0], cfg[idx2][1], cfg[idx2][2]
            cfg[idx][0] += random.gauss(0, 0.15 * step_scale)
            cfg[idx][1] += random.gauss(0, 0.15 * step_scale)
            cfg[idx][2] += random.gauss(0, 3.0 * step_scale)
            cfg[idx2][0] += random.gauss(0, 0.15 * step_scale)
            cfg[idx2][1] += random.gauss(0, 0.15 * step_scale)
            cfg[idx2][2] += random.gauss(0, 3.0 * step_scale)
        
        if _is_valid_fast(cfg):
            new_score = _outer_hex_side_fast(cfg, (0.0, 0.0), 0.0)
            
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                step_scale = max(0.5, step_scale * 0.99)
                if new_score < best_side:
                    best_side = new_score
                    best_cfg = [p[:] for p in cfg]
                    best_center = (0.0, 0.0)
                    best_angle = 0.0
            else:
                # Revert
                cfg[idx][0], cfg[idx][1], cfg[idx][2] = old_x, old_y, old_a
                if r >= 0.85:
                    cfg[idx2][0], cfg[idx2][1], cfg[idx2][2] = old_x2, old_y2, old_a2
        else:
            # Revert
            cfg[idx][0], cfg[idx][1], cfg[idx][2] = old_x, old_y, old_a
            if r >= 0.85:
                cfg[idx2][0], cfg[idx2][1], cfg[idx2][2] = old_x2, old_y2, old_a2
                
        T = max(T_end, T * alpha)

    # Phase 2: Refine outer hexagon
    # Try different outer angles and center offsets
    best_final_side = best_side
    best_final_cfg = best_cfg[:]
    best_final_center = best_center
    best_final_angle = best_angle
    
    # Coarse grid for angle
    for outer_angle in range(0, 60, 2):
        if time.time() - t0 > TIME_LIMIT - 5:
            break
        side = _outer_hex_side_fast(best_cfg, (0.0, 0.0), float(outer_angle))
        if side < best_final_side:
            best_final_side = side
            best_final_cfg = best_cfg[:]
            best_final_center = (0.0, 0.0)
            best_final_angle = float(outer_angle)
            
    # Fine tune center and angle around the best found
    if best_final_angle > 0:
        base_angle = best_final_angle
        base_center = best_final_center
        for da in [-1, -0.5, 0, 0.5, 1]:
            for dx in [-0.2, -0.1, 0, 0.1, 0.2]:
                for dy in [-0.2, -0.1, 0, 0.1, 0.2]:
                    a = base_angle + da
                    c = (base_center[0] + dx, base_center[1] + dy)
                    side = _outer_hex_side_fast(best_cfg, c, a)
                    if side < best_final_side:
                        best_final_side = side
                        best_final_cfg = best_cfg[:]
                        best_final_center = c
                        best_final_angle = a

    # Final verification
    if not _is_valid_fast(best_final_cfg):
        # Fallback to a valid loose configuration
        cfg = make_hex_lattice()
        for p in cfg:
            p[0] *= 1.5
            p[1] *= 1.5
        best_final_cfg = cfg
        best_final_side = _outer_hex_side_fast(cfg, (0.0, 0.0), 0.0)
        best_final_center = (0.0, 0.0)
        best_final_angle = 0.0
    else:
        # Recalculate side to be sure
        best_final_side = _outer_hex_side_fast(best_final_cfg, best_final_center, best_final_angle)

    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_final_cfg]
    return inner, best_final_center, best_final_side, best_final_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)

