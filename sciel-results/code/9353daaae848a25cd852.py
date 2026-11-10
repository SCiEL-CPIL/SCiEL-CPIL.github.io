# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, s, a_deg):
    a = np.radians(a_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _min_outer_hex_side(inner_hex_data):
    """Minimum side length of outer hexagon (centered at origin, angle 0) to contain all inner hexagons."""
    max_side = 0.0
    sqrt3_over_2 = np.sqrt(3) / 2
    for i in range(len(inner_hex_data)):
        cx, cy, a_deg = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1.0, a_deg)
        for j in range(6):
            px, py = verts[j]
            for k in range(6):
                angle_k = np.radians(60 * k)
                nx, ny = np.cos(angle_k), np.sin(angle_k)
                dist = px * nx + py * ny
                if dist > max_side * sqrt3_over_2:
                    required_s = dist / sqrt3_over_2
                    if required_s > max_side:
                        max_side = required_s
    return max_side

def _separation(i_hex, j_hex, tol=1e-6):
    """Check if two hexagons are separated (non-overlapping) using SAT."""
    ci = _hex_vertices(i_hex[0], i_hex[1], 1.0, i_hex[2])
    cj = _hex_vertices(j_hex[0], j_hex[1], 1.0, j_hex[2])
    for poly in [ci, cj]:
        for k in range(6):
            p1 = poly[k]
            p2 = poly[(k+1) % 6]
            ex, ey = p2[0] - p1[0], p2[1] - p1[1]
            nx, ny = -ey, ex
            length = np.sqrt(nx**2 + ny**2)
            if length < 1e-10:
                continue
            nx, ny = nx/length, ny/length
            proj_i = ci[:, 0] * nx + ci[:, 1] * ny
            proj_j = cj[:, 0] * nx + cj[:, 1] * ny
            if np.max(proj_i) + tol < np.min(proj_j) or np.max(proj_j) + tol < np.min(proj_i):
                return True  # Separated on this axis
    return False  # Overlapping

def _all_separated(inner_hex_data, tol=1e-6):
    n = len(inner_hex_data)
    for i in range(n):
        for j in range(i+1, n):
            if not _separation(inner_hex_data[i], inner_hex_data[j], tol):
                return False
    return True

def _penalty_overlap(inner_hex_data, tol=1e-6):
    """Total overlap penalty."""
    penalty = 0.0
    n = len(inner_hex_data)
    for i in range(n):
        for j in range(i+1, n):
            ci = _hex_vertices(inner_hex_data[i,0], inner_hex_data[i,1], 1.0, inner_hex_data[i,2])
            cj = _hex_vertices(inner_hex_data[j,0], inner_hex_data[j,1], 1.0, inner_hex_data[j,2])
            min_sep = np.inf
            for poly in [ci, cj]:
                for k in range(6):
                    p1 = poly[k]
                    p2 = poly[(k+1) % 6]
                    ex, ey = p2[0] - p1[0], p2[1] - p1[1]
                    nx, ny = -ey, ex
                    length = np.sqrt(nx**2 + ny**2)
                    if length < 1e-10:
                        continue
                    nx, ny = nx/length, ny/length
                    proj_i = ci[:, 0] * nx + ci[:, 1] * ny
                    proj_j = cj[:, 0] * nx + cj[:, 1] * ny
                    gap = max(np.min(proj_j) - np.max(proj_i), np.min(proj_i) - np.max(proj_j))
                    if gap < min_sep:
                        min_sep = gap
            if min_sep < tol:
                penalty += (tol - min_sep) ** 2 * 1e4
    return penalty

def hexagon_packing_11():
    deadline = time.time() + 590
    
    # Hexagonal flower of 7 + 4 extra at cardinal directions
    sqrt3 = np.sqrt(3)
    d = sqrt3  # center-to-center distance for touching hexagons
    
    # 7-hex flower centered at origin
    initial = np.array([
        [0, 0, 0],
        [d, 0, 0],
        [d*0.5, d*sqrt3*0.5, 0],
        [-d*0.5, d*sqrt3*0.5, 0],
        [-d, 0, 0],
        [-d*0.5, -d*sqrt3*0.5, 0],
        [d*0.5, -d*sqrt3*0.5, 0],
        # 4 more: extend the ring
        [1.5*d, 0, 0],
        [0.75*d, 1.5*d*sqrt3*0.5, 0],
        [-0.75*d, 1.5*d*sqrt3*0.5, 0],
        [-1.5*d, 0, 0],
        [-0.75*d, -1.5*d*sqrt3*0.5, 0],
    ])
    
    best_x = initial.copy()
    best_side = _min_outer_hex_side(initial)
    
    # Nelder-Mead optimization
    def objective(x):
        if time.time() > deadline:
            return 1e10
        x = x.reshape(11, 3)
        side = _min_outer_hex_side(x)
        pen = _penalty_overlap(x)
        return side + pen
    
    try:
        from scipy.optimize import minimize
        result = minimize(objective, initial.flatten(), method='Nelder-Mead',
                         options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11})
        if time.time() < deadline:
            x_opt = result.x.reshape(11, 3)
            if _all_separated(x_opt):
                side_opt = _min_outer_hex_side(x_opt)
                if side_opt < best_side:
                    best_x = x_opt
                    best_side = side_opt
    except Exception:
        pass
    
    # Fallback if invalid
    if not _all_separated(best_x):
        d2 = sqrt3 * 1.02
        best_x = np.array([
            [0, 0, 0],
            [d2, 0, 0],
            [d2*0.5, d2*sqrt3*0.5, 0],
            [-d2*0.5, d2*sqrt3*0.5, 0],
            [-d2, 0, 0],
            [-d2*0.5, -d2*sqrt3*0.5, 0],
            [d2*0.5, -d2*sqrt3*0.5, 0],
            [1.5*d2, 0, 0],
            [0.75*d2, 1.5*d2*sqrt3*0.5, 0],
            [-0.75*d2, 1.5*d2*sqrt3*0.5, 0],
            [-1.5*d2, 0, 0],
            [-0.75*d2, -1.5*d2*sqrt3*0.5, 0],
        ])
        best_side = _min_outer_hex_side(best_x)
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    return best_x, outer_hex_data, float(best_side)

# EVOLVE-BLOCK-END
