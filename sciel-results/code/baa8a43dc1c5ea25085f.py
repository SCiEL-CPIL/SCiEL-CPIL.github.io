import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill

# Precompute unit hexagon vertices and edge normals (angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i + 1) % 6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i + 1) % 6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx * _nx + _ny * _ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

_UX = _UNIT_HEX_VERTS[:, 0]
_UY = _UNIT_HEX_VERTS[:, 1]
_ENX = _EDGE_NORMALS[:, 0]
_ENY = _EDGE_NORMALS[:, 1]
_SQRT3 = math.sqrt(3)

def _get_hex_verts(pos, ang):
    """
    Get vertices of a hexagon.
    pos: (x, y)
    ang: rotation in radians
    Returns: (6, 2) array of vertices
    """
    c, s = math.cos(ang), math.sin(ang)
    # Rotation matrix: [[c, -s], [s, c]]
    # v_rot = R @ v_unit
    # x_rot = x_u * c - y_u * s
    # y_rot = x_u * s + y_u * c
    rx = _UX * c - _UY * s
    ry = _UX * s + _UY * c
    return np.column_stack((rx + pos[0], ry + pos[1]))

def _hexagons_disjoint_fast(h1_pos, h1_ang, h2_pos, h2_ang):
    """
    Check if two hexagons are disjoint.
    """
    dx = h1_pos[0] - h2_pos[0]
    dy = h1_pos[1] - h2_pos[1]
    dist_sq = dx * dx + dy * dy
    if dist_sq > 4.5: # (2*R)^2 approx, R=sqrt(3)/2, D=2*sqrt(3)/2*sqrt(3)=3? No.
        # Max distance between centers for disjointness is not fixed, but if > 2*sqrt(3) ~ 3.46 they are definitely apart.
        # Actually, if distance > 2 * (circumradius) = 2, they might still touch if aligned? No, circumradius is 1.
        # If distance > 2, they are disjoint.
        # Wait, circumradius of unit hex is 1. So if dist > 2, disjoint.
        # Inradius is sqrt(3)/2 ~ 0.866.
        # If dist > 2, definitely disjoint.
        return True
    
    v1 = _get_hex_verts(h1_pos, h1_ang)
    v2 = _get_hex_verts(h2_pos, h2_ang)
    
    # Get normals for v1
    c1, s1 = math.cos(h1_ang), math.sin(h1_ang)
    en1x = _ENX * c1 - _ENY * s1
    en1y = _ENX * s1 + _ENY * c1
    
    # Get normals for v2
    c2, s2 = math.cos(h2_ang), math.sin(h2_ang)
    en2x = _ENX * c2 - _ENY * s2
    en2y = _ENX * s2 + _ENY * c2
    
    # Check separating axes from v1
    for i in range(6):
        nx, ny = en1x[i], en1y[i]
        p1 = v1[:, 0] * nx + v1[:, 1] * ny
        p2 = v2[:, 0] * nx + v2[:, 1] * ny
        if p1.max() < p2.min() or p2.max() < p1.min():
            return True
            
    # Check separating axes from v2
    for i in range(6):
        nx, ny = en2x[i], en2y[i]
        p1 = v1[:, 0] * nx + v1[:, 1] * ny
        p2 = v2[:, 0] * nx + v2[:, 1] * ny
        if p1.max() < p2.min() or p2.max() < p1.min():
            return True
            
    return False

def _is_valid(positions, angles_rad):
    for i in range(N):
        for j in range(i + 1, N):
            if not _hexagons_disjoint_fast(positions[i], angles_rad[i], positions[j], angles_rad[j]):
                return False
    return True

def _compute_min_outer_side(positions, angles_rad):
    """
    Compute the minimum side length of a regular hexagon that encloses all given hexagons.
    Returns (side, phi, cx, cy)
    """
    # Collect all vertices
    all_verts = []
    for i in range(N):
        verts = _get_hex_verts(positions[i], angles_rad[i])
        all_verts.append(verts)
    verts_flat = np.vstack(all_verts)
    
    center = np.mean(verts_flat, axis=0)
    offsets = verts_flat - center
    
    def eval_phi(phi_deg):
        phi = np.radians(phi_deg)
        # Normals of the enclosing hexagon
        # The sides of a regular hexagon are at angles phi, phi+60, ...
        # The normals are at phi+30, phi+90, ...
        normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        # Project all points onto these 6 normals
        # proj[k, j] = dot(offset[k], normal[j])
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        
        # For a regular hexagon with side L, the distance from center to side (apothem) is L * sqrt(3) / 2
        # So max_proj = apothem = L * sqrt(3) / 2
        # L = max_proj * 2 / sqrt(3)
        side_vals = (2.0 / _SQRT3) * max_projs
        return np.max(side_vals)

    # Vectorized coarse scan
    phi_vals = np.radians(np.arange(0, 60, 1))
    normal_angles = phi_vals[:, None] + np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
    abs_proj = np.abs(proj)
    max_projs = np.max(abs_proj, axis=0)  # (60, 6)
    side_vals = (2.0 / _SQRT3) * max_projs
    best_side = np.max(side_vals).min()
    best_idx = np.argmin(np.max(side_vals, axis=1))
    best_phi = float(best_idx)

    # Fine tune around best
    try:
        lo = max(0.0, best_phi - 1.0)
        hi = min(60.0, best_phi + 1.0)
        res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                              options={'xatol': 1e-10, 'maxiter': 100})
        if res.success and res.fun < best_side:
            best_side = float(res.fun)
            best_phi = float(res.x)
    except Exception:
        pass

    return float(best_side), float(best_phi), float(center[0]), float(center[1])

def _kinematic_model(params):
    """
    Parameters:
    0: cx (center x)
    1: cy (center y)
    2: global_angle
    3: scale (spacing factor)
    4: r1 (radius of inner ring, 6 hexes)
    5: r2 (radius of outer ring, 5 hexes? No, 12 total. 1 center? No, 12 is even. Maybe 2 rings of 6? Or 1+5+6? Or 3+9?)
    
    Let's try a "Flower" pattern: 1 center, 6 around, 5 more? 
    Or 2 rings of 6? That's 12.
    Let's use 2 concentric rings of 6 hexagons.
    Inner ring radius r1, Outer ring radius r2.
    Angle offset for outer ring: delta_angle.
    
    Total params:
    cx, cy, global_angle, r1, r2, delta_angle_outer, 
    individual angles? No, keep them aligned for simplicity first (angle = global_angle).
    Maybe add a "wobble" or individual rotation? 
    Let's stick to: cx, cy, global_angle, r1, r2, delta_angle_outer.
    That's 6 parameters. Very low dimensional.
    
    Positions:
    Inner ring (i=0..5):
      theta_i = global_angle + i * 60 deg
      pos_i = center + r1 * (cos(theta_i), sin(theta_i))
      
    Outer ring (i=6..11):
      theta_i = global_angle + delta_angle_outer + (i-6) * 60 deg
      pos_i = center + r2 * (cos(theta_i), sin(theta_i))
      
    Angles:
      All hexes have angle = global_angle (aligned with radial direction? or fixed?)
      Let's say angle = global_angle.
    """
    cx, cy, g_ang, r1, r2, d_ang = params
    
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    
    # Inner ring
    for i in range(6):
        theta = g_ang + i * np.pi / 3
        pos[i, 0] = cx + r1 * np.cos(theta)
        pos[i, 1] = cy + r1 * np.sin(theta)
        ang[i] = g_ang
        
    # Outer ring
    for i in range(6):
        theta = g_ang + d_ang + i * np.pi / 3
        pos[6 + i, 0] = cx + r2 * np.cos(theta)
        pos[6 + i, 1] = cy + r2 * np.sin(theta)
        ang[6 + i] = g_ang
        
    return pos, ang

def _objective_kinematic(params):
    pos, ang = _kinematic_model(params)
    
    # Check validity
    if not _is_valid(pos, ang):
        # Penalize overlaps heavily
        # Calculate sum of penetration depths? Or just return a large number.
        # For Nelder-Mead, a large constant is okay, but gradient-like behavior is better.
        # Let's just return a large penalty.
        return 100.0
        
    side, _, _, _ = _compute_min_outer_side(pos, ang)
    return side

def construct():
    t0 = time.time()
    
    # Initial guesses for the kinematic model
    # r1 approx sqrt(3) ~ 1.732 for touching
    # r2 approx 2 * sqrt(3) ~ 3.464
    # d_ang approx 30 deg (offset by half step) or 0
    initial_params = [
        [0.0, 0.0, 0.0, 1.73, 3.46, 0.0],
        [0.0, 0.0, 0.0, 1.73, 3.46, np.pi/6],
        [0.0, 0.0, np.pi/6, 1.73, 3.46, 0.0],
        [0.0, 0.0, 0.0, 1.8, 3.6, np.pi/6],
    ]
    
    best_side = float('inf')
    best_pos = None
    best_ang = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    for init_params in initial_params:
        if time.time() - t0 > DEADLINE - 10.0:
            break
            
        # Nelder-Mead optimization
        try:
            res = minimize(_objective_kinematic, init_params, method='Nelder-Mead',
                           options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 2000, 'maxfev': 5000})
            
            if res.fun < best_side:
                # Re-evaluate to get precise metrics
                pos, ang = _kinematic_model(res.x)
                if _is_valid(pos, ang):
                    side, phi, cx, cy = _compute_min_outer_side(pos, ang)
                    if side < best_side:
                        best_side = side
                        best_pos = pos
                        best_ang = ang
                        best_phi = phi
                        best_cx = cx
                        best_cy = cy
        except Exception:
            pass

    # If kinematic model failed or found nothing, fallback to a simple valid configuration
    if best_pos is None:
        # Fallback: Honeycomb
        a1 = np.array([_SQRT3, 0.0])
        a2 = np.array([_SQRT3 / 2.0, 1.5])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        best_pos = np.zeros((N, 2))
        best_ang = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            best_pos[idx] = i * a1 + j * a2
            best_ang[idx] = 0.0
            
        # Ensure validity (it should be)
        if not _is_valid(best_pos, best_ang):
             # Last resort: spread them out
             for i in range(N):
                 best_pos[i, 0] = i * 2.0
                 best_pos[i, 1] = 0.0
                 
        best_side, best_phi, best_cx, best_cy = _compute_min_outer_side(best_pos, best_ang)

    inner = []
    for i in range(N):
        inner.append([float(best_pos[i, 0]), float(best_pos[i, 1]), float(best_ang[i])])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle

if __name__ == "__main__":
    construct()
