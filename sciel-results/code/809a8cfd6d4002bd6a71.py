# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 30.0

def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _rigid_cluster_offsets():
    """
    Generate 12 centers for unit hexagons in a compact hexagonal lattice cluster.
    Lattice vectors for non-overlapping unit hexagons (side 1):
    a1 = (3, 0)
    a2 = (1.5, 1.5 * sqrt(3))
    
    Indices (i, j) such that centers = i*a1 + j*a2.
    We pick the 12 points closest to the origin in the hexagonal metric to form a compact cluster.
    """
    a1 = np.array([3.0, 0.0])
    a2 = np.array([1.5, 1.5 * np.sqrt(3)])
    
    # Generate a grid of indices
    indices = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            indices.append((i, j))
            
    # Compute positions and sort by distance from origin
    pos_dist = []
    for i, j in indices:
        p = i * a1 + j * a2
        d = np.linalg.norm(p)
        pos_dist.append((d, i, j))
        
    pos_dist.sort(key=lambda x: x[0])
    
    # Take the first 12
    centers = []
    for d, i, j in pos_dist[:12]:
        centers.append(i * a1 + j * a2)
        
    centers = np.array(centers)
    # Center the cluster at origin for easier rotation logic
    centers -= np.mean(centers, axis=0)
    return centers

_LATTICE_BASE = _rigid_cluster_offsets()

# Precompute unit hexagon vertices (angle 0)
_UNIT_VERTS = _hex_vertices(0, 0, 1.0, 0.0)
# (6, 2) array

def _min_enclosing_hex_side_analytic(centers, theta, tx, ty):
    """
    Compute the minimum side length s of a regular hexagon (centered at origin, angle 0)
    that contains all 12 unit hexagons rotated by theta and translated by (tx, ty).
    
    A regular hexagon centered at origin with side s and angle 0 is defined by:
    max_k | n_k . p | <= s * sqrt(3) / 2
    where n_k are the outward normals to the edges.
    Normals are at angles 30, 90, 150, 210, 270, 330 degrees.
    """
    c = np.cos(theta)
    s_ = np.sin(theta)
    R = np.array([[c, -s_], [s_, c]])
    
    # Rotate centers
    rot_centers = (R @ centers.T).T + np.array([tx, ty])
    
    # All vertices of all hexagons
    # rot_centers is (12, 2), _UNIT_VERTS is (6, 2)
    # We need to add each center to each vertex offset
    # verts[i, j] = rot_centers[i] + _UNIT_VERTS[j]
    # Shape (12, 6, 2)
    verts = rot_centers[:, np.newaxis, :] + _UNIT_VERTS[np.newaxis, :, :]
    verts = verts.reshape(-1, 2) # (72, 2)
    
    # Normals for the outer hexagon (fixed at angle 0)
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)]) # (6, 2)
    
    # Project all vertices onto all normals
    # proj[k, v] = normals[k] . verts[v]
    proj = normals @ verts.T # (6, 72)
    
    # For each normal direction k, we need the hexagon to contain the projection range.
    # The hexagon edge is at distance apothem = s * sqrt(3) / 2 from center.
    # So for direction k, we need:
    # max_v (proj[k, v]) <= s * sqrt(3) / 2
    # min_v (proj[k, v]) >= -s * sqrt(3) / 2
    # This implies:
    # s * sqrt(3) / 2 >= max( max_v(proj[k,v]), -min_v(proj[k,v]) )
    # s >= 2/sqrt(3) * max_k( max(max_v, -min_v) )
    
    max_projs = np.max(proj, axis=1) # (6,)
    min_projs = np.min(proj, axis=1) # (6,)
    
    # Required apothem for each direction
    req_apothems = np.maximum(max_projs, -min_projs)
    
    max_req_apothem = np.max(req_apothems)
    
    s = 2.0 * max_req_apothem / np.sqrt(3.0)
    
    return s

def _objective(params):
    """
    Objective function for optimization.
    params = [theta, tx, ty]
    """
    theta, tx, ty = params
    return _min_enclosing_hex_side_analytic(_LATTICE_BASE, theta, tx, ty)

def _check_separation(centers, tol=1e-6):
    """
    Check if all pairs of unit hexagons are separated using SAT.
    Since we use a rigid lattice, this should always be true, but we verify.
    """
    n = len(centers)
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)
    
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    all_verts = np.zeros((n, 6, 2))
    for i in range(n):
        all_verts[i, :, 0] = centers[i, 0] + base_cos
        all_verts[i, :, 1] = centers[i, 1] + base_sin
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]
            
            proj_i = vi @ normals.T
            proj_j = vj @ normals.T
            
            min_i = np.min(proj_i, axis=1)
            max_i = np.max(proj_i, axis=1)
            min_j = np.min(proj_j, axis=1)
            max_j = np.max(proj_j, axis=1)
            
            separated = np.any(max_i < min_j - tol) or np.any(max_j < min_i - tol)
            if not separated:
                return False
    return True

def _check_containment(centers, s, tol=1e-6):
    """
    Check all inner vertices are inside the outer hexagon of side s centered at origin.
    """
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)
    
    # Outer hexagon vertices
    outer_verts = np.column_stack([s * np.cos(base_angles), s * np.sin(base_angles)])
    
    # For each inner vertex, check it's inside the outer hexagon
    for i in range(len(centers)):
        vx = centers[i, 0] + base_cos
        vy = centers[i, 1] + base_sin
        for k in range(6):
            v1 = outer_verts[k]
            v2 = outer_verts[(k + 1) % 6]
            edge = v2 - v1
            # cross(edge, point - v1) should be >= -tol for CCW polygon
            cross_val = edge[0] * (vy[k] - v1[1]) - edge[1] * (vx[k] - v1[0])
            if cross_val < -tol:
                return False
    return True

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    """
    best_s = np.inf
    best_params = None
    best_centers = None
    
    # 1. Global Scan over theta
    # The function s(theta, t_opt) is periodic with period pi/3 (60 deg).
    # We scan theta in [0, pi/3) with fine steps to find a good starting point.
    
    n_scan = 1000
    thetas = np.linspace(0, np.pi / 3, n_scan, endpoint=False)
    
    best_scan_s = np.inf
    best_scan_theta = 0
    best_scan_t = np.array([0, 0])
    
    for th in thetas:
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            break
            
        # For a fixed theta, find optimal translation t to minimize s.
        # s(t) = max_k ( max_i (n_k . (R p_i + t)) , -min_i (n_k . (R p_i + t)) ) * 2/sqrt(3)
        # Let u_i = R p_i.
        # For each k, let m_k = max_i (n_k . u_i), M_k = min_i (n_k . u_i).
        # Projections become n_k . u_i + n_k . t.
        # Max proj: m_k + n_k . t. Min proj: M_k + n_k . t.
        # We want to minimize max_k ( max(m_k + n_k . t, -(M_k + n_k . t)) )
        # This is a convex optimization problem in t (2 variables).
        # We can solve it using scipy.optimize.minimize or linear programming.
        # Given the small dimension, we can use a simple local optimizer starting from center of mass.
        
        c = np.cos(th)
        s_ = np.sin(th)
        R = np.array([[c, -s_], [s_, c]])
        rot_centers = (R @ _LATTICE_BASE.T).T
        com = np.mean(rot_centers, axis=0)
        
        def _obj_t(params_t):
            tx, ty = params_t
            return _min_enclosing_hex_side_analytic(_LATTICE_BASE, th, tx, ty)
        
        # Use Nelder-Mead for the 2D translation optimization
        try:
            res_t = minimize(_obj_t, [-com[0], -com[1]], method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            s_val = res_t.fun
            t_val = res_t.x
        except Exception:
            s_val = _min_enclosing_hex_side_analytic(_LATTICE_BASE, th, -com[0], -com[1])
            t_val = np.array([-com[0], -com[1]])
            
        if s_val < best_scan_s:
            best_scan_s = s_val
            best_scan_theta = th
            best_scan_t = t_val.copy()
            
    # 2. Local Refinement over theta, tx, ty
    # Start from the best scan result
    
    x0 = np.array([best_scan_theta, best_scan_t[0], best_scan_t[1]])
    
    try:
        res = minimize(_objective, x0, method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
        if res.fun < best_scan_s:
            best_scan_s = res.fun
            best_scan_theta = res.x[0]
            best_scan_t = res.x[1:3]
    except Exception:
        pass
        
    # 3. Construct final result
    theta = best_scan_theta
    tx, ty = best_scan_t
    
    c = np.cos(theta)
    s_ = np.sin(theta)
    R = np.array([[c, -s_], [s_, c]])
    best_centers = (R @ _LATTICE_BASE.T).T + np.array([tx, ty])
    best_s = _min_enclosing_hex_side_analytic(_LATTICE_BASE, theta, tx, ty)
    
    # Verify separation
    if not _check_separation(best_centers):
        # Fallback to unrotated lattice if something went wrong (shouldn't happen)
        best_centers = _LATTICE_BASE.copy()
        # Optimize translation for unrotated
        def _obj_t_fallback(params_t):
            tx, ty = params_t
            return _min_enclosing_hex_side_analytic(_LATTICE_BASE, 0.0, tx, ty)
        try:
            res_t = minimize(_obj_t_fallback, [0, 0], method='Nelder-Mead', options={'maxiter': 2000})
            best_s = res_t.fun
            best_centers = _LATTICE_BASE + res_t.x
        except Exception:
            best_s = _min_enclosing_hex_side_analytic(_LATTICE_BASE, 0.0, 0.0, 0.0)
            
    # Verify containment
    if not _check_containment(best_centers, best_s):
        # Increase s slightly to ensure validity
        best_s *= 1.001
        
    # Build output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_centers[i, 0]
        inner_hex_data[i, 1] = best_centers[i, 1]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = best_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
