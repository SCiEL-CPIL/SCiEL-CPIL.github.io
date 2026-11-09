# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _hex_vertices(cx, cy, d, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with side d and angle angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _min_outer_side(centers, angles, rot_deg):
    """
    Compute minimum side length s of outer hexagon (rotated by rot_deg) containing all inner hexagons.
    Returns (s, ox, oy).
    """
    sqrt3 = np.sqrt(3)
    rot = np.radians(rot_deg)
    
    # Collect all vertices of all inner hexagons
    n_inner = len(centers)
    all_verts = np.zeros((n_inner * 6, 2))
    for i in range(n_inner):
        cx, cy = centers[i]
        all_verts[6*i:6*(i+1)] = _hex_vertices(cx, cy, 1.0, angles[i])
    
    # For each of the 6 edge normal directions of the outer hexagon,
    # compute the required range and center projection
    s_min = 0.0
    A = np.zeros((6, 2))
    b = np.zeros(6)
    
    for i in range(6):
        # Outward normal direction for edge i of outer hexagon
        n_angle = rot + np.radians(60 * i + 30)
        nx = np.cos(n_angle)
        ny = np.sin(n_angle)
        A[i] = [nx, ny]
        
        projections = all_verts[:, 0] * nx + all_verts[:, 1] * ny
        p_min = projections.min()
        p_max = projections.max()
        
        required_s = (p_max - p_min) / sqrt3
        s_min = max(s_min, required_s)
        
        # Optimal center projection in this direction
        b[i] = (p_min + p_max) / 2
    
    # Solve for center (ox, oy) using least squares
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    ox, oy = c
    
    # Recompute s with the actual center
    s_final = 0.0
    center_proj = ox * A[:, 0] + oy * A[:, 1]
    for i in range(6):
        nx, ny = A[i]
        d_proj = all_verts[:, 0] * nx + all_verts[:, 1] * ny - center_proj[i]
        s_final = max(s_final, np.abs(d_proj).max() * 2 / sqrt3)
    
    return s_final, ox, oy


def _separation_check(c1, a1, c2, a2, eps=1e-6):
    """
    Check if two unit hexagons are separated using SAT.
    Returns True if separated.
    """
    d = c2 - c1
    
    # Edge normals for hexagon 1
    n1 = np.radians(a1)
    normals1 = np.column_stack([np.cos(n1 + np.radians(60 * np.arange(6))),
                                  np.sin(n1 + np.radians(60 * np.arange(6)))])
    
    # Edge normals for hexagon 2
    n2 = np.radians(a2)
    normals2 = np.column_stack([np.cos(n2 + np.radians(60 * np.arange(6))),
                                  np.sin(n2 + np.radians(60 * np.arange(6)))])
    
    all_normals = np.vstack([normals1, normals2])
    
    verts1 = _hex_vertices(c1[0], c1[1], 1.0, a1)
    verts2 = _hex_vertices(c2[0], c2[1], 1.0, a2)
    
    for n in all_normals:
        proj1 = verts1 @ n
        proj2 = verts2 @ n
        if proj1.max() + eps < proj2.min() or proj2.max() + eps < proj1.min():
            return True
    
    return False


def _validate(inner, outer, s, eps=1e-6):
    """Full validation of packing."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    ox, oy, oang = outer
    outer_verts = _hex_vertices(ox, oy, s, oang)
    
    # Check outer vertices distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    
    # Check separation
    for i in range(12):
        for j in range(i+1, 12):
            if not _separation_check(inner[i, :2], inner[i, 2], inner[j, :2], inner[j, 2], eps):
                return False
    
    # Check containment
    for i in range(12):
        cx, cy, aang = inner[i]
        verts = _hex_vertices(cx, cy, 1.0, aang)
        for v in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1) % 6]
                cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                if cross < -1e-6:
                    return False
    
    return True


def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon, minimizing outer side length.
    """
    start_time = time.time()
    deadline = start_time + 620.0
    sqrt3 = np.sqrt(3)
    
    best_s = np.inf
    best_inner = None
    best_outer = None
    
    def try_config(centers, angles, rot_deg):
        nonlocal best_s, best_inner, best_outer
        s, ox, oy = _min_outer_side(centers, angles, rot_deg)
        if s < best_s:
            inner = np.column_stack([centers, angles])
            outer = np.array([ox, oy, rot_deg])
            if _validate(inner, outer, s):
                best_s = s
                best_inner = inner
                best_outer = outer
    
    # Standard hexagonal lattice: 12 points
    # e1 = (2, 0), e2 = (1, sqrt(3))
    lattice_points = np.array([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (-1, 2), (-2, 1), (2, -1), (1, -2), (0, 2),
    ])
    
    e1 = np.array([2.0, 0.0])
    e2 = np.array([1.0, sqrt3])
    centers = np.array([p[0]*e1 + p[1]*e2 for p in lattice_points])
    angles = np.zeros(12)
    
    # Coarse rotation search
    for rot in np.linspace(0, 60, 61):
        if time.time() > deadline - 60:
            break
        try_config(centers, angles, rot)
    
    # Fine rotation search around best
    if best_s < np.inf:
        # Find the best rotation from coarse search
        best_rot = 0
        best_rot_s = np.inf
        for rot in np.linspace(0, 60, 61):
            s, _, _ = _min_outer_side(centers, angles, rot)
            if s < best_rot_s:
                best_rot_s = s
                best_rot = rot
        
        # Fine search around best_rot
        for rot in np.linspace(best_rot - 3, best_rot + 3, 121):
            if time.time() > deadline - 30:
                break
            try_config(centers, angles, rot % 60)
    
    # Try slightly expanded lattices for strict separation
    if time.time() < deadline - 30:
        for scale in [1.0005, 1.001, 1.0015, 1.002]:
            if time.time() > deadline - 20:
                break
            e1_s = e1 * scale
            e2_s = e2 * scale
            centers_s = np.array([p[0]*e1_s + p[1]*e2_s for p in lattice_points])
            angles_s = np.zeros(12)
            
            for rot in np.linspace(0, 60, 31):
                if time.time() > deadline - 10:
                    break
                try_config(centers_s, angles_s, rot)
    
    # Optimize lattice parameters
    if time.time() < deadline - 60:
        def obj(params):
            if time.time() > deadline - 50:
                return 1e10
            e1_len, e2_len, angle_deg = params
            e1_v = np.array([e1_len, 0.0])
            a_rad = np.radians(angle_deg)
            e2_v = np.array([e2_len * np.cos(a_rad), e2_len * np.sin(a_rad)])
            
            c = np.array([p[0]*e1_v + p[1]*e2_v for p in lattice_points])
            ang = np.zeros(12)
            
            # Quick check for overlaps
            for i in range(12):
                for j in range(i+1, 12):
                    if not _separation_check(c[i], ang[i], c[j], ang[j], 1e-7):
                        return 1e10
            
            # Find best rotation
            best_local = 1e10
            for rot in np.linspace(0, 60, 13):
                s, _, _ = _min_outer_side(c, ang, rot)
                best_local = min(best_local, s)
            
            return best_local
        
        try:
            res = minimize(obj, [2.0, 2.0, 60.0], method='Nelder-Mead',
                          options={'maxiter': 100, 'xatol': 1e-7, 'fatol': 1e-9})
            if res.fun < 1e9:
                e1_len, e2_len, angle_deg = res.x
                e1_v = np.array([e1_len, 0.0])
                a_rad = np.radians(angle_deg)
                e2_v = np.array([e2_len * np.cos(a_rad), e2_len * np.sin(a_rad)])
                c_opt = np.array([p[0]*e1_v + p[1]*e2_v for p in lattice_points])
                ang_opt = np.zeros(12)
                
                for rot in np.linspace(0, 60, 61):
                    if time.time() > deadline - 20:
                        break
                    try_config(c_opt, ang_opt, rot)
        except Exception:
            pass
    
    # Fallback
    if best_inner is None:
        centers_fb = np.array([p[0]*e1 + p[1]*e2 for p in lattice_points])
        angles_fb = np.zeros(12)
        s, ox, oy = _min_outer_side(centers_fb, angles_fb, 0)
        best_s = s
        best_inner = np.column_stack([centers_fb, angles_fb])
        best_outer = np.array([ox, oy, 0.0])
    
    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END
