import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side_lp(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with flat-top orientation offset by outer_angle_deg)
    that contains all points in verts. Uses LP to find optimal center and inradius.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    A_rows = []
    b_vals = []
    
    for i in range(M):
        vi = verts[i]
        for k in range(6):
            nk = normals[k]
            A_rows.append([-nk[0], -nk[1], -1.0])
            b_vals.append(-nk[0]*vi[0] - nk[1]*vi[1])
            A_rows.append([nk[0], nk[1], -1.0])
            b_vals.append(nk[0]*vi[0] + nk[1]*vi[1])
    
    A = np.array(A_rows)
    b = np.array(b_vals)
    c_obj = np.array([0.0, 0.0, 1.0])
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.status == 0:
            cx, cy, r = res.x
            side = r * 2.0 / _SQRT3
            return cx, cy, side
    except Exception:
        pass
    
    # Fallback: centroid-based
    cx, cy = np.mean(verts, axis=0)
    rel = verts - np.array([cx, cy])
    projections = rel @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    r = np.max(np.maximum(max_proj, -min_proj))
    if r < 0: r = 0
    side = r * 2.0 / _SQRT3
    return cx, cy, side

def _get_cluster_vertices(cluster_rotation_deg):
    """
    Get all vertices of 11 unit hexagons arranged in a compact hexagonal lattice,
    rotated by cluster_rotation_deg about the origin.
    Layout: center + 6 ring + 4 selected from ring 2 (symmetric)
    """
    sqrt3 = _SQRT3
    # Hexagonal lattice positions (for flat-top hexagons with 30° rotation)
    # Center at origin
    positions = [(0.0, 0.0)]
    # Ring 1: 6 neighbors
    for i in range(6):
        angle = math.radians(60 * i)
        x = sqrt3 * math.cos(angle)
        y = sqrt3 * math.sin(angle) * 0.8660254037844386  # * sqrt(3)/2
        # Actually for hexagonal lattice of flat-top hexagons:
        # spacing is sqrt(3) horizontally, 1.5 vertically
        # Let me use the proper hex grid coordinates
        pass
    
    # Use axial coordinates for hex grid
    # For flat-top hexagons with side 1, the centers are at:
    # (sqrt(3)*col + sqrt(3)/2*(row%2), 1.5*row) for pointy-top
    # For flat-top: (sqrt(3)/2*(col%2), sqrt(3)/2*col + 1.5*row) ... let me just use the known positions
    
    # Known compact arrangement of 11 hexagons:
    # Center + 6 surrounding + 4 more
    sqrt3 = _SQRT3
    pos = []
    # Center
    pos.append((0.0, 0.0))
    # Ring 1 (6 hexagons)
    for i in range(6):
        a = math.radians(60*i)
        pos.append((sqrt3*math.cos(a), sqrt3*math.sin(a)*0.5 + 0.0))
    # Wait, let me use the standard hex grid
    # For flat-top hexagons of side 1:
    # Horizontal distance between centers in same row: sqrt(3)
    # Vertical distance between rows: 1.5
    # Odd rows offset by sqrt(3)/2
    
    # Let me just hardcode the 11 positions for a compact cluster
    # Using the layout from the original that works well
    u = (sqrt3, 0.0)
    v = (sqrt3/2, 1.5)
    
    # Center + 6 ring + 4 from ring 2
    coords = [(0,0)]
    ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
    coords.extend(ring1)
    ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)]
    # Pick 4 symmetric ones from ring 2
    ring2_sel = [(2,0),(-2,0),(1,1),(-1,-1)]
    coords.extend(ring2_sel)
    
    positions = []
    for ci, cj in coords[:11]:
        x = ci*u[0] + cj*v[0]
        y = ci*u[1] + cj*v[1]
        positions.append((x, y))
    
    # Rotate all positions by cluster_rotation_deg
    theta = math.radians(cluster_rotation_deg)
    ct, st = math.cos(theta), math.sin(theta)
    rotated = []
    for x, y in positions:
        rx = x*ct - y*st
        ry = x*st + y*ct
        rotated.append((rx, ry))
    
    # Each hexagon has the same rotation as the cluster (no relative rotation)
    inner_angle = 30.0 + cluster_rotation_deg
    all_verts = []
    for px, py in rotated:
        verts = _hex_vertices(px, py, inner_angle)
        all_verts.append(verts)
    
    return np.vstack(all_verts)

def _objective(params):
    """
    Objective: minimize enclosing hexagon side length.
    params = [cluster_rotation_deg, outer_rotation_deg]
    """
    if time.time() > _DEADLINE - 5.0:
        return 1e10
    
    cluster_rot = params[0]
    outer_rot = params[1]
    
    verts = _get_cluster_vertices(cluster_rot)
    result = _min_enclosing_hex_side_lp(verts, outer_rot)
    if result is None:
        return 1e10
    cx, cy, side = result
    return side

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_params = None
    best_center = (0.0, 0.0)
    
    # Multi-start Nelder-Mead over (cluster_rotation, outer_rotation)
    # Due to 60° symmetry of hexagons, we only need to search [0, 60) for each
    # But the interaction means we search a 2D space
    
    initial_points = [
        (0.0, 0.0),
        (5.0, 0.0),
        (10.0, 5.0),
        (15.0, 10.0),
        (20.0, 15.0),
        (25.0, 20.0),
        (30.0, 0.0),
        (30.0, 15.0),
        (30.0, 30.0),
        (15.0, 30.0),
        (45.0, 15.0),
        (10.0, 25.0),
        (20.0, 35.0),
        (5.0, 40.0),
        (25.0, 5.0),
    ]
    
    for x0 in initial_points:
        if time.time() > _DEADLINE - 15.0:
            break
        
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8,
                                   'adaptive': True})
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Also do a fine grid search as backup
    for cluster_rot in np.arange(0, 30.0, 2.0):
        if time.time() > _DEADLINE - 10.0:
            break
        for outer_rot in np.arange(0, 30.0, 2.0):
            if time.time() > _DEADLINE - 10.0:
                break
            verts = _get_cluster_vertices(cluster_rot)
            result = _min_enclosing_hex_side_lp(verts, outer_rot)
            if result:
                cx, cy, side = result
                if side < best_side:
                    best_side = side
                    best_params = (cluster_rot, outer_rot)
    
    # Refine with finer Nelder-Mead if we have time
    if best_params is not None and time.time() < _DEADLINE - 20.0:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10,
                                   'adaptive': True})
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        # Fallback: use original best layout
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    cluster_rot, outer_rot = best_params
    verts = _get_cluster_vertices(cluster_rot)
    result = _min_enclosing_hex_side_lp(verts, outer_rot)
    if result:
        cx, cy, side = result
        best_center = (cx, cy)
        best_side = side
    else:
        cx, cy = 0.0, 0.0
        best_center = (cx, cy)
    
    # Build inner hexagon positions
    sqrt3 = _SQRT3
    u = (sqrt3, 0.0)
    v = (sqrt3/2, 1.5)
    
    coords = [(0,0)]
    ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
    coords.extend(ring1)
    ring2_sel = [(2,0),(-2,0),(1,1),(-1,-1)]
    coords.extend(ring2_sel)
    
    theta = math.radians(cluster_rot)
    ct, st = math.cos(theta), math.sin(theta)
    inner_angle = 30.0 + cluster_rot
    
    inner = []
    for ci, cj in coords[:11]:
        x = ci*u[0] + cj*v[0]
        y = ci*u[1] + cj*v[1]
        rx = x*ct - y*st
        ry = x*st + y*ct
        inner.append([rx, ry, inner_angle])
    
    return inner, best_center, best_side, outer_rot
