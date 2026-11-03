"""Direct hexagon packing: 11 unit hexagons in a hexagonal lattice pattern with optimized outer hexagon.
Optimized for speed using NumPy vectorization and robust multi-start optimization."""
import math
import time
import itertools
import numpy as np
from scipy.optimize import minimize

N = 11

# Precompute constants
SQRT3 = math.sqrt(3)
# The 6 normal directions for a hexagon with "flat top" or "pointy top"?
# The code uses: theta = 30 + 60*k degrees.
# Vertices are at angles angle + 60*k.
# Normals are at angle + 30 + 60*k.
# If inner hexagons are at angle 0, their vertices are at 0, 60, ...
# Normals are at 30, 90, ...
# Let's define the 6 unit normal vectors for the outer hexagon's constraints.
# These are fixed relative to the outer hexagon's local frame.
# When we rotate the outer hexagon by `angle`, these normals rotate by `angle`.
# Alternatively, we can rotate the points by `-angle` and project onto fixed axes.
# Fixed axes (normals) for a hexagon centered at origin with angle 0:
# Normal 1: (cos(30), sin(30)) = (SQRT3/2, 0.5)
# Normal 2: (cos(90), sin(90)) = (0, 1)
# Normal 3: (cos(150), sin(150)) = (-SQRT3/2, 0.5)
# Normal 4: (cos(210), sin(210)) = (-SQRT3/2, -0.5)
# Normal 5: (cos(270), sin(270)) = (0, -1)
# Normal 6: (cos(330), sin(330)) = (SQRT3/2, -0.5)

NORMALS = np.array([
    [SQRT3 / 2, 0.5],
    [0.0, 1.0],
    [-SQRT3 / 2, 0.5],
    [-SQRT3 / 2, -0.5],
    [0.0, -1.0],
    [SQRT3 / 2, -0.5]
])

def _get_hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    a = math.radians(angle_deg)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    # Base vertices for side=1, angle=0
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    base_x = side * np.cos(base_angles)
    base_y = side * np.sin(base_angles)
    
    # Rotate
    rot_x = base_x * cos_a - base_y * sin_a
    rot_y = base_x * sin_a + base_y * cos_a
    
    # Translate
    x = cx + rot_x
    y = cy + rot_y
    return np.column_stack((x, y))

def _compute_min_outer_side_vectorized(inner_hexes, ox, oy, oangle_deg):
    """
    Compute the minimum side length of an outer hexagon centered at (ox, oy) 
    with rotation oangle_deg that contains all inner hexagons.
    
    inner_hexes: list of (cx, cy, side, angle_deg)
    """
    # Collect all vertices of all inner hexagons
    all_verts = []
    for cx, cy, s, ang in inner_hexes:
        verts = _get_hex_vertices(cx, cy, s, ang)
        all_verts.append(verts)
    
    if not all_verts:
        return 0.0
        
    all_verts = np.vstack(all_verts)
    
    # Rotate points by -oangle to align outer hexagon normals with fixed NORMALS
    a = -math.radians(oangle_deg)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    
    # Translate points relative to outer center
    dx = all_verts[:, 0] - ox
    dy = all_verts[:, 1] - oy
    
    # Rotate
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a
    
    # Project onto the 6 normal directions
    # NORMALS shape (6, 2)
    # rx, ry shape (N,)
    # Projections shape (N, 6)
    projs = np.column_stack((
        rx * NORMALS[:, 0] + ry * NORMALS[:, 1] # This is wrong, need to multiply correctly
    ))
    
    # Correct projection calculation:
    # proj_i = dot(v_i, n_j)
    # v_i is (rx, ry), n_j is (nx, ny)
    # projs = np.einsum('ni,ji->nj', np.column_stack((rx, ry)), NORMALS)
    # Or simply:
    v = np.column_stack((rx, ry))
    projs = v @ NORMALS.T # Shape (N, 6)
    
    # For each of the 6 directions, we need the max projection.
    # The outer hexagon contains the points if max(proj_j) <= apothem_outer for all j.
    # Apothem_outer = side_outer * sqrt(3) / 2
    # So side_outer >= max_j(max_i(proj_ij)) * 2 / sqrt(3)
    
    max_projs = np.max(projs, axis=0) # Shape (6,)
    global_max = np.max(max_projs)
    
    return global_max * 2.0 / SQRT3

def _optimize_outer_hex_fast(inner_hexes, initial_cx, initial_cy, initial_angle, deadline):
    """
    Optimize outer hexagon center and angle.
    Uses Nelder-Mead on the vectorized objective.
    """
    # Precompute vertices for speed? 
    # _compute_min_outer_side_vectorized recomputes them every time.
    # We can cache them if inner_hexes don't change.
    cached_verts = []
    for cx, cy, s, ang in inner_hexes:
        cached_verts.append(_get_hex_vertices(cx, cy, s, ang))
    all_verts = np.vstack(cached_verts)
    
    def objective(params):
        cx, cy, angle = params
        a = -math.radians(angle)
        cos_a = math.cos(a)
        sin_a = math.sin(a)
        
        dx = all_verts[:, 0] - cx
        dy = all_verts[:, 1] - cy
        
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        
        v = np.column_stack((rx, ry))
        projs = v @ NORMALS.T
        max_projs = np.max(projs, axis=0)
        global_max = np.max(max_projs)
        return global_max * 2.0 / SQRT3

    x0 = np.array([initial_cx, initial_cy, initial_angle])
    
    # Check time
    if time.time() > deadline:
        return initial_cx, initial_cy, initial_angle, objective(x0)

    # Multi-start with a few angles to avoid local minima in angle
    best_x = x0
    best_f = objective(x0)
    
    # Try a few perturbations in angle
    angles_to_try = [initial_angle, initial_angle + 5, initial_angle - 5, initial_angle + 15, initial_angle - 15]
    
    for ang in angles_to_try:
        if time.time() > deadline:
            break
        x_try = np.array([initial_cx, initial_cy, ang])
        f_try = objective(x_try)
        if f_try < best_f:
            best_f = f_try
            best_x = x_try
            
        # Run a quick Nelder-Mead
        try:
            res = minimize(objective, x_try, method='Nelder-Mead', 
                           options={'xatol': 1e-8, 'fatol': 1e-11, 'maxiter': 1000, 'maxfev': 2000})
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        except Exception:
            pass
            
    # Final polish
    try:
        res = minimize(objective, best_x, method='Nelder-Mead', 
                       options={'xatol': 1e-10, 'fatol': 1e-13, 'maxiter': 500, 'maxfev': 1000})
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
    except Exception:
        pass
        
    return best_x[0], best_x[1], best_x[2], best_f

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin

    sqrt3 = math.sqrt(3)

    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    # Positions for 11 hexagons
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)
    
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
             
    best_side = float('inf')
    best_positions = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # We will test a subset of combinations if time is tight, but with vectorized opt,
    # we can likely test all 1001 quickly.
    # C(14, 4) = 1001
    combos = list(itertools.combinations(range(len(ring2)), 4))
    
    # Shuffle to find good ones early? Or just iterate.
    # The "most symmetric" ones might be best.
    # Let's prioritize combinations that are "balanced".
    
    for combo in combos:
        if time.time() > deadline:
            break
            
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0
        
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
        
        # Initial guess for outer
        # Try a few angles
        initial_angles = [0.0, 30.0, 15.0, 45.0]
        
        for oang in initial_angles:
            if time.time() > deadline:
                break
            # Quick estimate first?
            # The optimizer is fast enough, just run it.
            opt_cx, opt_cy, opt_angle, opt_side = _optimize_outer_hex_fast(
                inner_hexes, cx_mean, cy_mean, oang, deadline
            )
            
            if opt_side < best_side:
                best_side = opt_side
                best_positions = full_xy
                best_center = (opt_cx, opt_cy)
                best_angle = opt_angle

    if best_positions is None:
        # Fallback
        best_positions = [axial_to_xy(q, r) for q, r in pos_axial]
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]
        cx_mean = sum(x for x, y in best_positions) / 7.0
        cy_mean = sum(y for x, y in best_positions) / 7.0
        best_cx, best_cy, best_angle, best_side = _optimize_outer_hex_fast(
            inner_hexes, cx_mean, cy_mean, 0.0, deadline
        )
        best_center = (best_cx, best_cy)

    # Final verification and formatting
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]
    
    # Ensure containment with a tiny buffer just in case of numerical errors
    # The optimizer minimizes the side, so it should be tight.
    # Adding a tiny epsilon ensures validity.
    best_side *= 1.000001

    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_center[0], best_center[1]), best_side, best_angle
