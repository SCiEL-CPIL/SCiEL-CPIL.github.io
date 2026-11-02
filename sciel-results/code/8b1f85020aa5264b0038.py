"""Hexagonal packing of unit hexagons inside a regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _hex_vertices_local(s=1.0):
    """Vertices of a unit regular hexagon centered at origin, flat-top orientation."""
    angles = np.radians(np.arange(6) * 60.0)
    return np.column_stack([s * np.cos(angles), s * np.sin(angles)])

def _hex_vertices(center, angle_deg, s=1.0):
    """Vertices of a hexagon with given center, rotation, and side length."""
    local = _hex_vertices_local(s)
    theta = np.radians(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return local @ R.T + np.array(center)

def _outer_hex_vertices(center, side, angle_deg):
    """Vertices of outer hexagon."""
    local = _hex_vertices_local(side)
    theta = np.radians(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return local @ R.T + np.array(center)

def _point_in_hexagon(p, center, side, angle_deg):
    """Check if point p is inside hexagon with given center, side, angle."""
    theta = np.radians(angle_deg)
    R_inv = np.array([[np.cos(-theta), -np.sin(-theta)],
                      [np.sin(-theta), np.cos(-theta)]])
    p_local = R_inv @ (np.array(p) - np.array(center))
    x, y = p_local
    s_val = side
    sqrt3 = np.sqrt(3.0)
    if x > s_val + 1e-10:
        return False
    if x < -s_val - 1e-10:
        return False
    if sqrt3 * y - x > s_val * sqrt3 + 1e-10:
        return False
    if sqrt3 * y + x > s_val * sqrt3 + 1e-10:
        return False
    if -sqrt3 * y - x > s_val * sqrt3 + 1e-10:
        return False
    if -sqrt3 * y + x > s_val * sqrt3 + 1e-10:
        return False
    return True

def _hex_in_hexagon(center, angle_deg, outer_center, outer_side, outer_angle):
    """Check if a unit hexagon (with given center and angle) is inside the outer hexagon."""
    verts = _hex_vertices(center, angle_deg, 1.0)
    for v in verts:
        if not _point_in_hexagon(v, outer_center, outer_side, outer_angle):
            return False
    return True

def _hex_overlap(c1, a1, c2, a2, tol=1e-8):
    """Check if two unit hexagons overlap."""
    d = np.linalg.norm(np.array(c1) - np.array(c2))
    return d < np.sqrt(3.0) - tol

def _compute_score(inner, outer_center, outer_side, outer_angle):
    """Compute validity score: 0 if invalid, else outer_side (lower is better)."""
    for (cx, cy, ang) in inner:
        if not _hex_in_hexagon((cx, cy), ang, outer_center, outer_side, outer_angle):
            return None
    n = len(inner)
    for i in range(n):
        for j in range(i+1, n):
            if _hex_overlap(inner[i][0:2], inner[i][2], inner[j][0:2], inner[j][2]):
                return None
    return outer_side

def _min_enclosing_hex_side(positions, angles, outer_center, outer_angle):
    """Compute minimum side length of outer hexagon that encloses all inner hexagons."""
    # Collect all vertices of all inner hexagons
    all_verts = []
    for i in range(len(positions)):
        cx, cy = positions[i]
        ang = angles[i]
        verts = _hex_vertices((cx, cy), ang, 1.0)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # For a regular hexagon with center c, side s, angle a:
    # A point p is inside if for all 6 half-planes: n_i . (p - c) <= s * h_i
    # where n_i are unit normals and h_i are apothem offsets
    # For flat-top hexagon with side s, the 6 half-plane constraints are:
    # n_0 = (1, 0), offset = s
    # n_1 = (1/2, sqrt(3)/2), offset = s*sqrt(3)/2
    # n_2 = (-1/2, sqrt(3)/2), offset = s*sqrt(3)/2
    # n_3 = (-1, 0), offset = s
    # n_4 = (-1/2, -sqrt(3)/2), offset = s*sqrt(3)/2
    # n_5 = (1/2, -sqrt(3)/2), offset = s*sqrt(3)/2
    
    # Rotate these by outer_angle
    theta = np.radians(outer_angle)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Original normals (flat-top)
    normals = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Original offsets (for side length s=1)
    offsets = np.array([1, np.sqrt(3)/2, np.sqrt(3)/2, 1, np.sqrt(3)/2, np.sqrt(3)/2])
    
    # Rotate normals
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    rotated_normals = normals @ R.T
    
    # For each vertex, compute required s such that n_i . (v - c) <= s * offset_i
    # => s >= n_i . (v - c) / offset_i
    
    c = np.array(outer_center)
    max_s = 0.0
    for v in all_verts:
        diffs = rotated_normals.T @ (v - c)
        required_s = diffs / offsets
        max_s = max(max_s, np.max(required_s))
    
    return max_s

def _build_positions(n):
    """Generate positions for n hexagons in a hexagonal lattice pattern."""
    if n == 1:
        return [(0.0, 0.0)]
    
    sqrt3 = np.sqrt(3.0)
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3/2.0, 1.5])
    
    candidates = []
    for i in range(-6, 7):
        for j in range(-6, 7):
            pos = i * v1 + j * v2
            candidates.append(pos)
    
    candidates.sort(key=lambda p: np.linalg.norm(p))
    positions = [tuple(p) for p in candidates[:n]]
    return positions

def construct():
    """Construct N unit hexagons inside a regular hexagon, minimizing outer side length."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Build initial hexagonal lattice configuration
    positions = _build_positions(N)
    positions = np.array(positions)
    angles = np.zeros(N)
    
    # Compute initial outer hexagon
    outer_center = np.mean(positions, axis=0)
    outer_angle = 0.0
    outer_side = _min_enclosing_hex_side(positions, angles, outer_center, outer_angle)
    
    best_side = outer_side
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_outer_center = outer_center.copy()
    best_outer_angle = outer_angle
    
    def objective(params):
        """Objective: outer side length with penalty for invalid configurations."""
        # params: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN, cx, cy, outer_angle]
        positions_p = params[:3*N].reshape(N, 3)
        outer_center_p = params[3*N:3*N+2]
        outer_angle_p = params[3*N+2]
        
        pos = positions_p[:, :2]
        ang = positions_p[:, 2]
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                if _hex_overlap(pos[i], ang[i], pos[j], ang[j]):
                    return 1e10
        
        # Compute outer side
        side = _min_enclosing_hex_side(pos, ang, outer_center_p, outer_angle_p)
        
        # Check all inner hexagons are inside
        for i in range(N):
            if not _hex_in_hexagon(pos[i], np.degrees(ang[i]), outer_center_p, side, np.degrees(outer_angle_p)):
                return 1e10
        
        return side
    
    # Optimize using Nelder-Mead
    x0 = np.concatenate([
        positions.reshape(N, 3).flatten(),
        [outer_center[0], outer_center[1], outer_angle]
    ])
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
        
        if res.fun < best_side and res.fun < 1e9:
            best_side = res.fun
            best_params = res.x
            best_positions = best_params[:3*N].reshape(N, 3)[:, :2]
            best_angles = best_params[:3*N].reshape(N, 3)[:, 2]
            best_outer_center = best_params[3*N:3*N+2]
            best_outer_angle = best_params[3*N+2]
    except Exception:
        pass
    
    # Final verification and cleanup
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i][0]), float(best_positions[i][1]), float(np.degrees(best_angles[i]))])
    
    # Final check: recompute outer side to ensure consistency
    final_side = _min_enclosing_hex_side(best_positions, best_angles, best_outer_center, best_outer_angle)
    
    # Verify validity
    valid = True
    for i in range(N):
        if not _hex_in_hexagon(best_positions[i], best_angles[i], best_outer_center, final_side, best_outer_angle):
            valid = False
            break
    if valid:
        for i in range(N):
            for j in range(i+1, N):
                if _hex_overlap(best_positions[i], best_angles[i], best_positions[j], best_angles[j]):
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        # Fallback: use the initial lattice configuration
        inner = [[p[0], p[1], 0.0] for p in positions]
        final_side = _min_enclosing_hex_side(positions, angles, outer_center, outer_angle)
    
    return inner, best_outer_center.tolist(), float(final_side), float(best_outer_angle)

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
