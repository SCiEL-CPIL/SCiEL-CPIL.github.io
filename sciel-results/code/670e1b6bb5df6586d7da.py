import numpy as np
import time
from scipy.optimize import minimize, LinearProgramming
from scipy.linalg import lstsq

_START = time.time()
_DEADLINE = 600.0  # safety margin before 630s kill

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side s, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles),
                             cy + s * np.sin(angles)])

def _hex_edge_normals(a_deg):
    """Return 12 unit outward edge normals for a regular hexagon at angle a_deg."""
    a = np.deg2rad(a_deg)
    # Edge midpoints are at angles a + 30 + 60*i
    edge_angles = a + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
    # Also include the 6 opposite normals (for the 12 edge normals check)
    all_normals = np.vstack([normals, -normals])
    return all_normals

def _separation_ok(hex1, hex2, eps=1e-6):
    """Check if two unit hexagons are separated (no intersection).
    Each hex: (cx, cy, angle_deg). Side = 1."""
    cx1, cy1, a1 = hex1
    cx2, cy2, a2 = hex2
    
    n1 = _hex_edge_normals(a1)
    n2 = _hex_edge_normals(a2)
    all_normals = np.vstack([n1, n2])
    
    verts1 = _hex_vertices(cx1, cy1, 1.0, a1)
    verts2 = _hex_vertices(cx2, cy2, 1.0, a2)
    
    for i in range(12):
        n = all_normals[i]
        proj1 = verts1 @ n
        proj2 = verts2 @ n
        # Projected intervals must be apart by more than eps
        # Interval 1: [min1, max1], Interval 2: [min2, max2]
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - eps or max2 < min1 - eps:
            return True  # Separated along this normal
    return False  # No separating axis found -> intersecting

def _all_separated(inner_hexes, eps=1e-6):
    """Check all pairs of inner hexagons are separated."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _separation_ok(inner_hexes[i], inner_hexes[j], eps):
                return False
    return True

def _enclosing_hex_side(inner_hexes, outer_center, outer_angle):
    """Compute minimum side length of a regular hexagon (centered at outer_center,
    with angle outer_angle) that contains all inner hexagon vertices."""
    ocx, ocy = outer_center[0], outer_center[1]
    oa = np.deg2rad(outer_angle)
    
    # Outer hexagon edges: for a regular hexagon centered at origin with angle a,
    # the 6 edges have outward normals at angles a + 30 + 60*i
    # A point (px, py) is inside if for all 6 edges:
    #   dot(point - center, normal_i) <= s * apothem = s * sqrt(3)/2
    # So s >= 2/sqrt(3) * max_i dot(point - center, normal_i)
    
    edge_normals = np.column_stack([
        np.cos(oa + np.deg2rad(30.0 + 60.0 * np.arange(6))),
        np.sin(oa + np.deg2rad(30.0 + 60.0 * np.arange(6)))
    ])
    
    max_proj = 0.0
    for hx, hy, ha in inner_hexes:
        verts = _hex_vertices(hx, hy, 1.0, ha)
        shifted = verts - np.array([ocx, ocy])
        projs = shifted @ edge_normals.T  # shape (6, 6)
        max_proj = max(max_proj, projs.max())
    
    side = 2.0 / np.sqrt(3.0) * max_proj
    return side

def _objective(params):
    """Objective: side length of enclosing hexagon given layout parameters.
    params = [r1, r2, angle_deg]
    Layout: center at origin, 6 at radius r1 at 60deg spacing (offset by angle),
    4 at radius r2 at 90, 270, and two more positions.
    """
    r1, r2, ang = params[0], params[1], np.deg2rad(params[2])
    
    # 11 hexagons: 1 center + 6 ring at r1 + 4 outer at r2
    # Ring positions: 60*i + ang
    # Outer positions: 90, 270 (top/bottom) and 210, 330 (diagonal) - need to think about this
    # Actually for 6-fold symmetry with 11: 1+6+4 doesn't have 6-fold symmetry for the 4.
    # Let me use: 1 center, 6 at r1 (60deg spacing), 4 at r2 (90deg spacing offset)
    # This gives 4-fold + 6-fold mixed symmetry.
    
    inner = np.zeros((11, 3))
    inner[0] = [0, 0, 0]  # center
    
    # 6 hexagons at radius r1, 60-degree spacing
    for i in range(6):
        theta = ang + np.deg2rad(60.0 * i)
        inner[i + 1, 0] = r1 * np.cos(theta)
        inner[i + 1, 1] = r1 * np.sin(theta)
        inner[i + 1, 2] = 0  # all aligned
    
    # 4 hexagons at radius r2, at 90-degree spacing (top, right, bottom, left)
    for i in range(4):
        theta = ang + np.deg2rad(90.0 * i)
        inner[i + 7, 0] = r2 * np.cos(theta)
        inner[i + 7, 1] = r2 * np.sin(theta)
        inner[i + 7, 2] = 0
    
    # Check separation
    if not _all_separated(inner):
        return 1e10
    
    # Compute enclosing hexagon side (try several outer angles)
    best_side = np.inf
    for oa in [0, 30, 60, 90]:  # try a few outer angles
        s = _enclosing_hex_side(inner, np.array([0.0, 0.0]), oa)
        best_side = min(best_side, s)
    
    return best_side

def _objective2(params):
    """More flexible: r1, r2, outer_r2_angle_offset, all_hex_angle.
    params = [r1, r2, offset, hex_angle]
    """
    r1, r2, offset, hex_ang = params
    r1 = max(r1, 2.0)  # minimum separation
    r2 = max(r2, 3.0)
    offset = np.deg2rad(offset)
    hex_ang = hex_ang % 360
    
    inner = np.zeros((11, 3))
    inner[0] = [0, 0, hex_ang]
    
    for i in range(6):
        theta = offset + np.deg2rad(60.0 * i)
        inner[i + 1, 0] = r1 * np.cos(theta)
        inner[i + 1, 1] = r1 * np.sin(theta)
        inner[i + 1, 2] = hex_ang
    
    for i in range(4):
        theta = offset + np.deg2rad(90.0 * i)
        inner[i + 7, 0] = r2 * np.cos(theta)
        inner[i + 7, 1] = r2 * np.sin(theta)
        inner[i + 7, 2] = hex_ang
    
    if not _all_separated(inner):
        return 1e10
    
    best_side = np.inf
    for oa in [0, 30, 60, 90, 15, 45, 75]:
        s = _enclosing_hex_side(inner, np.array([0.0, 0.0]), oa)
        best_side = min(best_side, s)
    
    return best_side

def _refine_with_lm(inner_hexes, outer_center, outer_angle):
    """Local refinement using scipy minimize on positions."""
    def obj_flat(flat):
        arr = flat.reshape(11, 3)
        if not _all_separated(arr):
            return 1e10
        return _enclosing_hex_side(arr, outer_center, outer_angle)
    
    x0 = inner_hexes.flatten()
    result = minimize(obj_flat, x0, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9})
    if result.success and result.fun < 1e9:
        return result.x.reshape(11, 3), result.fun
    return inner_hexes, _enclosing_hex_side(inner_hexes, outer_center, outer_angle)

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    best_side = np.inf
    best_inner = None
    best_outer_angle = 0.0
    
    # Initial guess: 6-fold symmetric layout
    # Center + 6 ring + 4 outer
    r1 = 2.5
    r2 = 4.0
    offset = 0.0
    
    inner = np.zeros((11, 3))
    inner[0] = [0, 0, 0]
    for i in range(6):
        theta = np.deg2rad(60.0 * i)
        inner[i + 1] = [r1 * np.cos(theta), r1 * np.sin(theta), 0]
    for i in range(4):
        theta = np.deg2rad(90.0 * i)
        inner[i + 7] = [r2 * np.cos(theta), r2 * np.sin(theta), 0]
    
    # Try multiple outer angles
    for oa in range(0, 90, 15):
        s = _enclosing_hex_side(inner, np.array([0.0, 0.0]), oa)
        if s < best_side:
            best_side = s
            best_inner = inner.copy()
            best_outer_angle = oa
    
    # Nelder-Mead optimization on layout parameters
    if _time_left() > 60:
        x0 = np.array([2.5, 4.0, 0.0, 0.0])
        res = minimize(_objective2, x0, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8,
                                'adaptive': True})
        if res.fun < best_side:
            r1, r2, offset, hex_ang = res.x
            offset = np.deg2rad(offset)
            inner2 = np.zeros((11, 3))
            inner2[0] = [0, 0, hex_ang]
            for i in range(6):
                theta = offset + np.deg2rad(60.0 * i)
                inner2[i + 1] = [r1 * np.cos(theta), r1 * np.sin(theta), hex_ang]
            for i in range(4):
                theta = offset + np.deg2rad(90.0 * i)
                inner2[i + 7] = [r2 * np.cos(theta), r2 * np.sin(theta), hex_ang]
            for oa in range(0, 90, 10):
                s = _enclosing_hex_side(inner2, np.array([0.0, 0.0]), oa)
                if s < best_side:
                    best_side = s
                    best_inner = inner2.copy()
                    best_outer_angle = oa
    
    # Try multiple random starts
    if _time_left() > 30:
        np.random.seed(42)
        for trial in range(200):
            if _time_left() < 10:
                break
            r1 = np.random.uniform(2.0, 3.5)
            r2 = np.random.uniform(3.0, 5.0)
            off = np.random.uniform(0, 60)
            ha = np.random.uniform(0, 60)
            
            inner_t = np.zeros((11, 3))
            inner_t[0] = [0, 0, ha]
            for i in range(6):
                theta = np.deg2rad(off) + np.deg2rad(60.0 * i)
                inner_t[i + 1] = [r1 * np.cos(theta), r1 * np.sin(theta), ha]
            for i in range(4):
                theta = np.deg2rad(off) + np.deg2rad(90.0 * i)
                inner_t[i + 7] = [r2 * np.cos(theta), r2 * np.sin(theta), ha]
            
            if not _all_separated(inner_t):
                continue
            
            for oa in [0, 30, 60]:
                s = _enclosing_hex_side(inner_t, np.array([0.0, 0.0]), oa)
                if s < best_side:
                    best_side = s
                    best_inner = inner_t.copy()
                    best_outer_angle = oa
    
    # Final local refinement on positions
    if _time_left() > 30 and best_inner is not None:
        refined, side_val = _refine_with_lm(best_inner, np.array([0.0, 0.0]), best_outer_angle)
        if side_val < best_side and _all_separated(refined):
            best_inner = refined
            best_side = side_val
    
    # Fallback
    if best_inner is None or best_side == np.inf:
        best_inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0],
            [-1.25, -2.17, 0], [1.25, -2.17, 0],
            [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0],
        ], dtype=float)
        best_outer_angle = 0.0
        best_side = _enclosing_hex_side(best_inner, np.array([0.0, 0.0]), 0.0)
    
    outer_hex_data = np.array([0.0, 0.0, float(best_outer_angle)])
    
    return best_inner, outer_hex_data, float(best_side)
