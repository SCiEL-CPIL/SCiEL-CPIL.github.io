# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


def _hex_vertices(cx, cy, d, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with side d and angle angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _compute_outer_hex(centers, rot_deg):
    """
    Given inner hex centers (unit hexagons, all axis-aligned),
    compute the minimum side length s of a regular hexagon (rotated by rot_deg)
    that contains all 12 inner hexagons, and the optimal center offset.
    Uses LP for optimal center placement.
    Returns (s, ox, oy) or (inf, 0, 0) if infeasible.
    """
    rot = np.radians(rot_deg)
    
    # Unit hexagon vertices (side 1, angle 0)
    unit_verts = np.array([[np.cos(np.radians(60*i)), np.sin(np.radians(60*i))] for i in range(6)])
    
    # All inner vertices
    all_verts = []
    for c in centers:
        for u in unit_verts:
            all_verts.append(c + u)
    all_verts = np.array(all_verts)  # shape (72, 2)
    
    from scipy.optimize import linprog
    
    n_constraints = 6 * len(all_verts)
    A = np.zeros((n_constraints, 3))  # variables: [ox, oy, s]
    b = np.zeros(n_constraints)
    
    idx = 0
    for i in range(6):
        angle = rot + np.radians(60*i + 30)
        ci = np.cos(angle)
        si = np.sin(angle)
        for j in range(len(all_verts)):
            A[idx, 0] = -ci
            A[idx, 1] = -si
            A[idx, 2] = -np.sqrt(3)/2
            b[idx] = -(ci * all_verts[j, 0] + si * all_verts[j, 1])
            idx += 1
    
    c_obj = np.array([0, 0, 1])
    bounds = [(None, None), (None, None), (0, None)]
    
    res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    
    if res.status == 0:
        return res.x[2], res.x[0], res.x[1]
    else:
        return np.inf, 0.0, 0.0


def _check_separation(centers, eps=1e-5):
    """
    Check that all pairs of unit hexagons centered at given positions are separated.
    Uses separating axis test with the 6 edge normals of a unit hexagon.
    Returns True if all pairs are separated by more than eps.
    """
    # Edge normals for a unit hexagon (angle 0)
    normal_angles = np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    n = len(centers)
    for i in range(n):
        for j in range(i+1, n):
            diff = centers[j] - centers[i]
            for k in range(6):
                proj_diff = np.dot(diff, normals[k])
                if abs(proj_diff) > np.sqrt(3) + 2*eps:
                    break
            else:
                return False
    return True


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses Nelder-Mead optimization on lattice parameters.
    """
    start_time = time.time()
    deadline = start_time + 570.0
    
    sqrt3 = np.sqrt(3)
    
    lattice_indices = np.array([
        [0, 0], [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
        [2, 0], [1, 1], [-1, 2], [-2, 1], [-2, 0]
    ])
    
    def objective(params):
        s_lat, rot_deg = params
        if s_lat < 1.8:
            return 1e10
        if time.time() > deadline:
            return 1e10
            
        a1 = np.array([s_lat, 0.0])
        a2 = np.array([s_lat/2, s_lat*sqrt3/2])
        
        # Fixed broadcasting: (12,1) * (2,) -> (12,2)
        centers_lat = lattice_indices[:, 0:1] * a1 + lattice_indices[:, 1:2] * a2
        
        rot = np.radians(rot_deg)
        cos_r, sin_r = np.cos(rot), np.sin(rot)
        rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        centers = centers_lat @ rot_matrix.T
        
        if not _check_separation(centers, eps=1e-5):
            return 1e10
            
        s_out, ox, oy = _compute_outer_hex(centers, rot_deg)
        
        if not np.isfinite(s_out) or s_out < np.sqrt(12) - 1e-6:
            return 1e10
            
        # Maximize 1/s, so minimize -1/s
        return -1.0 / s_out

    best_val = 1e10
    best_params = None
    
    # Multiple starts for Nelder-Mead
    starts = [
        [2.0, 0.0],
        [2.1, 30.0],
        [2.2, 15.0],
        [1.9, 45.0]
    ]
    
    for sp in starts:
        if time.time() > deadline - 30:
            break
        try:
            res = scipy.optimize.minimize(objective, sp, method='Nelder-Mead', 
                                          options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-7})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            continue
            
    if best_params is None:
        # Fallback
        best_params = [2.0, 0.0]
        
    s_lat, rot_deg = best_params
    a1 = np.array([s_lat, 0.0])
    a2 = np.array([s_lat/2, s_lat*sqrt3/2])
    centers_lat = lattice_indices[:, 0:1] * a1 + lattice_indices[:, 1:2] * a2
    rot = np.radians(rot_deg)
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    centers = centers_lat @ rot_matrix.T
    
    s_out, ox, oy = _compute_outer_hex(centers, rot_deg)
    
    inner_hex_data = np.column_stack([centers, np.full(12, rot_deg)])
    outer_hex_data = np.array([ox, oy, rot_deg])
    
    return inner_hex_data, outer_hex_data, s_out


# EVOLVE-BLOCK-END
