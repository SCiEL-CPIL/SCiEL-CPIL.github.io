"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import itertools

N = 12
SIDE = 1.0  # side length of inner hexagons

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, side, angle_deg):
    """Compute vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    vertices = []
    for k in range(6):
        a = angle + k * np.pi / 3.0
        x = cx + side * np.cos(a)
        y = cy + side * np.sin(a)
        vertices.append((x, y))
    return np.array(vertices)

def _hex_distance_to_edge(cx, cy, side, angle_deg, px, py):
    """For a given point, compute the minimum signed distance to the hexagon boundary
    (negative if inside). Returns the max over edges of the penetration."""
    angle = np.radians(angle_deg)
    # For each of 6 edges, compute the distance
    min_dist = np.inf
    for k in range(6):
        # Edge midpoint angle
        edge_angle = angle + (k + 0.5) * np.pi / 3.0
        # Normal direction (outward)
        nx = np.cos(edge_angle)
        ny = np.sin(edge_angle)
        # Distance from center to this edge along normal
        edge_dist = side * np.sqrt(3) / 2.0
        # Project point onto normal
        proj = (px - cx) * nx + (py - cy) * ny
        # Signed distance from edge (positive if outside)
        d = proj - edge_dist
        min_dist = min(min_dist, d)
    return min_dist  # negative means inside

def _check_overlap(pos1, pos2, side=SIDE):
    """Check if two hexagons overlap. Use vertex containment as a quick check."""
    x1, y1, a1 = pos1
    x2, y2, a2 = pos2
    # Check if any vertex of hex1 is inside hex2 or vice versa
    verts1 = _hex_vertices(x1, y1, side, a1)
    verts2 = _hex_vertices(x2, y2, side, a2)
    for vx, vy in verts1:
        if _hex_distance_to_edge(x2, y2, side, a2, vx, vy) < -1e-9:
            return True
    for vx, vy in verts2:
        if _hex_distance_to_edge(x1, y1, side, a1, vx, vy) < -1e-9:
            return True
    return False

def _is_valid_configuration(positions, side=SIDE, tol=1e-9):
    """Check if all hexagons are pairwise non-overlapping."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if _check_overlap(positions[i], positions[j], side):
                return False
    return True

def _min_enclosing_hexagon_side(positions, side_inner=SIDE):
    """Given positions of inner hexagons, find the minimum side length of an
    enclosing regular hexagon. Returns (min_side, center_x, center_y, angle_deg)."""
    # Get all vertices of all inner hexagons
    all_pts = []
    for x, y, ang in positions:
        verts = _hex_vertices(x, y, side_inner, ang)
        all_pts.extend(verts.tolist())
    all_pts = np.array(all_pts)
    
    if len(all_pts) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    
    # The minimum enclosing hexagon can be found by optimizing over the rotation angle
    # For a fixed rotation, the problem becomes: find center (cx, cy) and side S such that
    # for all points p and all k=0..5: (p - c) . n_k <= S * sqrt(3)/2
    # This is equivalent to S >= max_{p,k} (p-c).n_k / (sqrt(3)/2)
    
    # We'll sample rotation angles and for each, solve for optimal center
    best_S = np.inf
    best_c = None
    best_angle = 0.0
    
    # Sample angles from 0 to 60 degrees (hexagon has 60-degree symmetry)
    angles = np.linspace(0, np.pi/3, 61)  # 0 to 60 degrees
    
    for theta in angles:
        # Compute the 6 normal directions for this hexagon orientation
        normals = []
        for k in range(6):
            alpha = theta + k * np.pi/3.0 - np.pi/6.0  # edge normal direction
            normals.append((np.cos(alpha), np.sin(alpha)))
        normals = np.array(normals)  # shape (6, 2)
        
        # For each point p and each normal n_k, compute the constraint
        # (p - c) . n_k <= S * sqrt(3)/2
        # => -c . n_k <= S * sqrt(3)/2 - p . n_k
        # We want to minimize S subject to these constraints
        
        # Let t = S * sqrt(3)/2. Minimize t subject to:
        # c . n_k >= p . n_k - t for all p, k
        
        # This is a linear program in variables (cx, cy, t)
        # Minimize t
        # Subject to: cx * n_k[0] + cy * n_k[1] + t >= p_i . n_k for all i, k
        
        # Number of constraints: len(all_pts) * 6
        m = len(all_pts) * 6
        # Variables: [cx, cy, t]
        # c matrix: for each constraint (i, k): [n_k[0], n_k[1], 1]
        # b vector: for each constraint (i, k): p_i . n_k
        
        A = np.zeros((m, 3))
        b = np.zeros(m)
        for i, p in enumerate(all_pts):
            for k in range(6):
                idx = i * 6 + k
                A[idx, 0] = normals[k, 0]
                A[idx, 1] = normals[k, 1]
                A[idx, 2] = 1.0
                b[idx] = p[0] * normals[k, 0] + p[1] * normals[k, 1]
        
        # Minimize t = A_obj @ [cx, cy, t] where A_obj = [0, 0, 1]
        # Subject to A @ x >= b  =>  -A @ x <= -b
        
        try:
            from scipy.optimize import linprog
            c_obj = np.array([0.0, 0.0, 1.0])
            A_ub = -A
            b_ub = -b
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, method='highs')
            if res.status == 0:
                t_val = res.x[2]
                S_val = t_val / (np.sqrt(3) / 2.0)
                if S_val < best_S:
                    best_S = S_val
                    best_c = (res.x[0], res.x[1])
                    best_angle = np.degrees(theta)
        except Exception:
            pass
    
    if best_c is None:
        # Fallback: use centroid
        cx = np.mean(all_pts[:, 0])
        cy = np.mean(all_pts[:, 1])
        # Compute required S for angle 0
        S = 0
        for p in all_pts:
            for k in range(6):
                alpha = k * np.pi/3.0 - np.pi/6.0
                nx = np.cos(alpha)
                ny = np.sin(alpha)
                proj = (p[0] - cx) * nx + (p[1] - cy) * ny
                S = max(S, proj / (np.sqrt(3)/2.0))
        return (S, cx, cy, 0.0)
    
    return (best_S, best_c[0], best_c[1], best_angle)

def _initial_hex_lattice(rows_config):
    """Generate initial positions in hex lattice pattern.
    rows_config: list of (num_hex, x_offset, y_row)
    """
    positions = []
    dist = np.sqrt(3) * SIDE  # center-to-center distance for adjacent hexagons
    for num_hex, x_off, y_row in rows_config:
        for k in range(num_hex):
            x = x_off + k * dist
            y = y_row
            positions.append([x, y, 0.0])
    return positions

def _optimize_configuration(initial_positions, side=SIDE, max_iter=500):
    """Optimize positions to minimize enclosing hexagon side length."""
    positions = np.array(initial_positions, dtype=float)
    
    def objective(x):
        # x: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12]
        pos = []
        for i in range(N):
            px = x[i*3]
            py = x[i*3+1]
            pa = x[i*3+2]
            pos.append([px, py, pa])
        
        # Check overlaps - if any overlap, return large penalty
        for i in range(N):
            for j in range(i+1, N):
                if _check_overlap(pos[i], pos[j], side):
                    return 1e6
        side_enc, cx, cy, ang = _min_enclosing_hexagon_side(pos, side)
        return side_enc
    
    # Try to optimize
    best_x = positions.flatten()
    best_obj = objective(best_x)
    
    # Use Nelder-Mead for the optimization
    try:
        res = minimize(objective, best_x, method='Nelder-Mead', 
                      options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.fun < best_obj:
            best_x = res.x
            best_obj = res.fun
    except Exception:
        pass
    
    # Reconstruct positions
    final_pos = []
    for i in range(N):
        final_pos.append([best_x[i*3], best_x[i*3+1], best_x[i*3+2]])
    
    return final_pos

def construct():
    # Try multiple initial configurations and pick the best
    # Configuration 1: 3-4-3-2 rows
    rows1 = [
        (3, 0.0, 0.0),
        (4, np.sqrt(3)/2, 1.5 * SIDE),
        (3, 0.0, 3.0 * SIDE),
        (2, np.sqrt(3)/2, 4.5 * SIDE),
    ]
    
    # Configuration 2: 4-4-4 rows
    rows2 = [
        (4, 0.0, 0.0),
        (4, np.sqrt(3)/2, 1.5 * SIDE),
        (4, 0.0, 3.0 * SIDE),
    ]
    
    # Configuration 3: 2-3-4-3-2 rows
    rows3 = [
        (2, 0.0, 0.0),
        (3, np.sqrt(3)/2, 1.5 * SIDE),
        (4, 0.0, 3.0 * SIDE),
        (3, np.sqrt(3)/2, 4.5 * SIDE),
        (2, 0.0, 6.0 * SIDE),
    ]
    
    # Configuration 4: Flower pattern (1 center + 6 ring + 5 outer)
    def flower_pattern():
        pos = [[0.0, 0.0, 0.0]]  # center
        dist = np.sqrt(3) * SIDE
        # 6 around
        for k in range(6):
            angle = k * np.pi / 3.0
            x = dist * np.cos(angle)
            y = dist * np.sin(angle)
            pos.append([x, y, 0.0])
        # 5 more in outer ring, placed in the gaps
        outer_dist = dist * 2.0  # roughly
        for k in range(5):
            angle = k * 2 * np.pi / 5.0 + np.pi/5.0
            x = outer_dist * np.cos(angle)
            y = outer_dist * np.sin(angle)
            pos.append([x, y, 0.0])
        return pos
    
    candidates = []
    for rows in [rows1, rows2, rows3]:
        init_pos = _initial_hex_lattice(rows)
        opt_pos = _optimize_configuration(init_pos, SIDE, max_iter=300)
        side_enc, cx, cy, ang = _min_enclosing_hexagon_side(opt_pos, SIDE)
        candidates.append((side_enc, opt_pos, cx, cy, ang))
    
    # Also try flower
    init_flower = flower_pattern()
    opt_flower = _optimize_configuration(init_flower, SIDE, max_iter=300)
    side_enc_f, cx_f, cy_f, ang_f = _min_enclosing_hexagon_side(opt_flower, SIDE)
    candidates.append((side_enc_f, opt_flower, cx_f, cy_f, ang_f))
    
    # Pick the best
    candidates.sort(key=lambda x: x[0])
    best_side, best_pos, best_cx, best_cy, best_ang = candidates[0]
    
    # Normalize: center the configuration at origin
    # The outer hexagon center is (best_cx, best_cy)
    # Shift all inner hexagons so that their bounding region is centered
    # Actually, the enclosing hexagon already has center (best_cx, best_cy)
    # So we return the positions as-is, with outer center at (best_cx, best_cy)
    
    return best_pos, (best_cx, best_cy), best_side, best_ang

# EVOLVE-BLOCK-END

