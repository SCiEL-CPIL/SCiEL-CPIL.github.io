import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3

def _hex_vertex_offsets(angle_deg):
    """6 vertex offsets of unit hexagon centered at origin with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _build_cluster_offsets():
    """
    Build the 66 vertex offsets (relative to cluster center) for 11 hexagons
    in a compact hexagonal lattice arrangement, all with zero relative rotation.
    Returns array of shape (66, 2).
    """
    scale = 1.0000001
    sqrt3 = _SQRT3
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    # Compact arrangement: center + 6 ring + 4 from second ring
    target = [(0, 0)]
    ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
    target.extend(ring1)
    ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)]
    ring2_sorted = sorted(ring2, key=lambda p: p[0]**2 + p[1]**2)
    target.extend(ring2_sorted[:4])
    
    base_verts = _hex_vertex_offsets(0.0)  # 6 vertices at origin
    
    offsets = []
    for ti, tj in target:
        cx = ti * u[0] + tj * v[0]
        cy = ti * u[1] + tj * v[1]
        for vx, vy in base_verts:
            offsets.append((vx + cx, vy + cy))
    
    return np.array(offsets)

# Precompute cluster offsets
_CLUSTER_OFFSETS = _build_cluster_offsets()
_NVERTS = len(_CLUSTER_OFFSETS)

def _solve_rigid_lp(theta_deg, phi_deg):
    """
    Solve the LP for fixed cluster rotation theta and outer hex orientation phi.
    Variables: [tx, ty, cx, cy, R]
    Minimize R.
    Constraints: for each vertex offset o, rotated by theta:
        (o_rot + t - c) . n_k <= R  for all 6 outer normals n_k
        (c - o_rot - t) . n_k <= R
    where o_rot = R(theta) @ o
    """
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg + 30)  # outer normals at phi + 30
    
    # Rotate cluster offsets
    c_th, s_th = math.cos(theta), math.sin(theta)
    rot = np.array([[c_th, -s_th], [s_th, c_th]])
    rotated_offsets = _CLUSTER_OFFSETS @ rot.T  # (66, 2)
    
    # Outer normals
    angles = phi + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)
    
    # For each vertex i and normal k:
    # (o_i + t - c).n_k <= R  =>  t.n_k - c.n_k - R <= -o_i.n_k
    # (c - o_i - t).n_k <= R  =>  -t.n_k + c.n_k - R <= o_i.n_k
    
    # Build A_ub and b_ub
    # Variables: [tx, ty, cx, cy, R]
    A_rows = []
    b_vals = []
    
    for k in range(6):
        nx, ny = normals[k]
        # o_i . n_k for all i
        dots = rotated_offsets @ np.array([nx, ny])  # (66,)
        
        # Row type 1: nx*tx + ny*ty - nx*cx - ny*cy - R <= -dot_i
        A1 = np.column_stack([
            np.full(_NVERTS, nx), np.full(_NVERTS, ny),
            np.full(_NVERTS, -nx), np.full(_NVERTS, -ny),
            np.full(_NVERTS, -1.0)
        ])
        A_rows.append(A1)
        b_vals.append(-dots)
        
        # Row type 2: -nx*tx - ny*ty + nx*cx + ny*cy - R <= dot_i
        A2 = np.column_stack([
            np.full(_NVERTS, -nx), np.full(_NVERTS, -ny),
            np.full(_NVERTS, nx), np.full(_NVERTS, ny),
            np.full(_NVERTS, -1.0)
        ])
        A_rows.append(A2)
        b_vals.append(dots)
    
    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)
    
    # R >= 0
    A = np.vstack([A, np.array([[0.0, 0.0, 0.0, 0.0, -1.0]])])
    b = np.concatenate([b, [0.0]])
    
    c_obj = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (None, None), (None, None), (0, None)],
                      method='highs', options={'presolve': True})
        if res.success:
            tx, ty, cx, cy, R = res.x
            side = R * 2.0 / _SQRT3
            return tx, ty, cx, cy, side
    except Exception:
        pass
    return None

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_tx = 0.0
    best_ty = 0.0
    best_cx = 0.0
    best_cy = 0.0
    best_theta = 0.0
    best_phi = 0.0
    
    # Coarse grid over theta (cluster rotation) and phi (outer orientation)
    # By symmetry, theta in [0, 60) and phi in [0, 30)
    theta_vals = np.arange(0, 60.1, 2.0)
    phi_vals = np.arange(0, 30.1, 2.0)
    
    coarse_results = []
    
    for theta in theta_vals:
        if time.time() > _DEADLINE - 15.0:
            break
        for phi in phi_vals:
            if time.time() > _DEADLINE - 15.0:
                break
            res = _solve_rigid_lp(theta, phi)
            if res:
                tx, ty, cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_tx, best_ty = tx, ty
                    best_cx, best_cy = cx, cy
                    best_theta = theta
                    best_phi = phi
    
    # Local refinement around best (theta, phi)
    if best_side < float('inf') and time.time() < _DEADLINE - 20.0:
        # Fine grid around best
        for theta in np.arange(max(0, best_theta - 2.0), min(60.0, best_theta + 2.0) + 0.01, 0.25):
            if time.time() > _DEADLINE - 15.0:
                break
            for phi in np.arange(max(0, best_phi - 2.0), min(30.0, best_phi + 2.0) + 0.01, 0.25):
                if time.time() > _DEADLINE - 15.0:
                    break
                res = _solve_rigid_lp(theta, phi)
                if res:
                    tx, ty, cx, cy, side = res
                    if side < best_side:
                        best_side = side
                        best_tx, best_ty = tx, ty
                        best_cx, best_cy = cx, cy
                        best_theta = theta
                        best_phi = phi
    
    # Ultra-fine polish
    if best_side < float('inf') and time.time() < _DEADLINE - 10.0:
        for theta in np.arange(max(0, best_theta - 0.5), min(60.0, best_theta + 0.5) + 0.001, 0.05):
            if time.time() > _DEADLINE - 8.0:
                break
            for phi in np.arange(max(0, best_phi - 0.5), min(30.0, best_phi + 0.5) + 0.001, 0.05):
                if time.time() > _DEADLINE - 8.0:
                    break
                res = _solve_rigid_lp(theta, phi)
                if res:
                    tx, ty, cx, cy, side = res
                    if side < best_side:
                        best_side = side
                        best_tx, best_ty = tx, ty
                        best_cx, best_cy = cx, cy
                        best_theta = theta
                        best_phi = phi
    
    # Build inner positions: cluster offsets rotated by best_theta, translated by (best_tx, best_ty)
    theta_rad = math.radians(best_theta)
    c_th, s_th = math.cos(theta_rad), math.sin(theta_rad)
    rot = np.array([[c_th, -s_th], [s_th, c_th]])
    
    scale = 1.0000001
    sqrt3 = _SQRT3
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    target = [(0, 0)]
    ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
    target.extend(ring1)
    ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)]
    ring2_sorted = sorted(ring2, key=lambda p: p[0]**2 + p[1]**2)
    target.extend(ring2_sorted[:4])
    
    inner = []
    for ti, tj in target:
        cx_local = ti * u[0] + tj * v[0]
        cy_local = ti * u[1] + tj * v[1]
        # Rotate and translate
        px = c_th * cx_local - s_th * cy_local + best_tx
        py = s_th * cx_local + c_th * cy_local + best_ty
        inner.append([px, py, best_theta])
    
    if best_inner_is_valid(inner, (best_cx, best_cy), best_side, best_phi):
        return inner, (best_cx, best_cy), best_side, best_phi
    
    # Fallback
    inner_fb = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
    return inner_fb, (0.0, 0.0), 40.0, 0.0

def best_inner_is_valid(inner, center, side, outer_angle):
    """Quick validity check."""
    try:
        cx, cy = center
        verts = []
        for x, y, ang in inner:
            a0 = math.radians(ang)
            angles = a0 + np.arange(6) * math.pi / 3.0
            verts.extend([(x + math.cos(a), y + math.sin(a)) for a in angles])
        verts = np.array(verts)
        
        base_angle = math.radians(outer_angle + 30)
        angles = base_angle + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        R = side * _SQRT3 / 2.0
        for v in verts:
            for n in normals:
                if np.dot(v - np.array(center), n) > R + 1e-6:
                    return False
                if np.dot(np.array(center) - v, n) > R + 1e-6:
                    return False
        return True
    except Exception:
        return False
