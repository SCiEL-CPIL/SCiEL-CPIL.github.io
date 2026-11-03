import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg):
    """Generate vertices of a unit hexagon centered at (cx, cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side_lp(verts, outer_angle_deg):
    """
    Solve the LP to find the minimum side length of a regular hexagon 
    with orientation outer_angle_deg that encloses the given vertices.
    Returns (cx, cy, side_length).
    """
    # Normals to the sides of the outer hexagon
    # The sides are perpendicular to the normals.
    # For a regular hexagon, normals are at angles 0, 60, ..., 300 + offset.
    # Standard orientation (flat top): normals are at 90, 150, 210, 270, 330, 30.
    # Let's define normals based on outer_angle_deg.
    # If outer_angle_deg is 0, vertices are at 0, 60, ...
    # Normals are at 30, 90, 150, 210, 270, 330.
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    # v_dot_n[i, k] = verts[i] dot normals[k]
    v_dot_n = verts @ normals.T  # (M, 6)
    
    # Variables: cx, cy, R
    # Constraints:
    # For each vertex i and normal k:
    # n_k dot (p_i - c) <= R  =>  -n_k dot c + R >= -n_k dot p_i
    # -n_k dot (p_i - c) <= R =>  n_k dot c + R >= n_k dot p_i
    
    A_rows = []
    b_vals = []
    
    for k in range(6):
        nx, ny = normals[k]
        v_dots = v_dot_n[:, k]
        
        # -nx*cx - ny*cy + R >= -v_dots  =>  nx*cx + ny*cy - R <= v_dots
        A_rows.append(np.column_stack([np.full(M, nx), np.full(M, ny), np.full(M, -1.0)]))
        b_vals.append(v_dots)
        
        # nx*cx + ny*cy + R >= v_dots  =>  -nx*cx - ny*cy - R <= -v_dots
        A_rows.append(np.column_stack([np.full(M, -nx), np.full(M, -ny), np.full(M, -1.0)]))
        b_vals.append(-v_dots)
    
    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)
    
    c_obj = np.array([0.0, 0.0, 1.0])
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R * 2.0 / _SQRT3
            return cx, cy, side
    except Exception:
        pass
    
    # Fallback: Centroid based estimate
    cx, cy = np.mean(verts, axis=0)
    rel = verts - np.array([cx, cy])
    projections = rel @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    r = np.max(np.maximum(max_proj, -min_proj))
    if r < 0: r = 0
    side = r * 2.0 / _SQRT3
    return cx, cy, side

def _build_rigid_layouts():
    """
    Generate a list of rigid layouts (list of (x,y) positions) for 11 unit hexagons.
    These are subsets of the triangular lattice.
    """
    layouts = []
    scale = 1.0
    u = (_SQRT3 * scale, 0.0)
    v = (_SQRT3 / 2 * scale, 1.5 * scale)
    
    # Helper to convert axial coords to cartesian
    def axial_to_cartesian(q, r):
        x = u[0] * q + v[0] * r
        y = u[1] * q + v[1] * r
        return (x, y)

    # Layout 0: Hex(3) minus 8 from outer ring.
    # Hex(3) has 19 hexagons.
    # Center (0,0).
    # Ring 1 (6 hexagons): (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 (12 hexagons): (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2) -- wait, standard ring 2 has 12.
    # Let's list all 19 in Hex(3):
    # q, r such that max(|q|, |r|, |q+r|) <= 2
    coords = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q+r)) <= 2:
                coords.append((q, r))
    # Sort by distance from center to ensure consistent selection
    coords.sort(key=lambda p: p[0]**2 + p[1]**2 + p[0]*p[1])
    
    # We need 11. The center and ring 1 are 7 hexagons.
    # We need 4 more from ring 2.
    # To maintain symmetry, pick 4 from the 12 in ring 2.
    # Options:
    # 1. 4 corners? (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2) are the 6 "corners" of the hexagon?
    #    Actually, the vertices of the hexagon are at (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2).
    #    If we pick 4 of these 6, we break 6-fold symmetry but might maintain 2-fold or 3-fold.
    #    Let's try picking (2,0), (0,2), (-2,0), (0,-2). This maintains 4-fold? No, hexagons don't have 4-fold.
    #    Let's try picking (2,0), (-2,0), (0,2), (0,-2). This is symmetric wrt x and y axes (if rotated).
    #    Let's try picking (2,0), (1,1), (-1,2), (-2,1). This is a "chunk".
    
    # Let's generate a few variations.
    ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
    ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1),(2,-2),(-2,2)]
    # Wait, (-2,2) is in ring 2? max(2,2,0)=2. Yes.
    # Let's verify the 12 points in ring 2.
    # (2,0), (2,-1), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-1,2), (0,2), (1,1), (2,0)... 
    # Let's just filter the 19 coords.
    ring2_coords = [c for c in coords if max(abs(c[0]), abs(c[1]), abs(c[0]+c[1])) == 2]
    
    # Variation 1: Pick 4 "corner" points from ring 2.
    # Corners are (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2).
    # Let's pick (2,0), (0,2), (-2,0), (0,-2).
    selected_r2_v1 = [(2,0), (0,2), (-2,0), (0,-2)]
    layout_v1 = [(0,0)] + ring1 + selected_r2_v1
    layouts.append([axial_to_cartesian(q,r) for q,r in layout_v1])
    
    # Variation 2: Pick 4 adjacent points from ring 2.
    # (2,0), (1,1), (0,2), (-1,2)
    selected_r2_v2 = [(2,0), (1,1), (0,2), (-1,2)]
    layout_v2 = [(0,0)] + ring1 + selected_r2_v2
    layouts.append([axial_to_cartesian(q,r) for q,r in layout_v2])
    
    # Variation 3: Pick 4 points to make a rectangle-ish shape?
    # (2,0), (-2,0), (1,1), (-1,-1)
    selected_r2_v3 = [(2,0), (-2,0), (1,1), (-1,-1)]
    layout_v3 = [(0,0)] + ring1 + selected_r2_v3
    layouts.append([axial_to_cartesian(q,r) for q,r in layout_v3])
    
    # Variation 4: A different base. Maybe a line of 5, line of 3, line of 3?
    # Row 1 (y=0): 5 hexagons. x = -2*sqrt3, -1*sqrt3, 0, 1*sqrt3, 2*sqrt3.
    # Row 2 (y=1.5): 3 hexagons. x = -1.5*sqrt3, -0.5*sqrt3, 0.5*sqrt3.
    # Row 3 (y=3.0): 3 hexagons. x = -1.5*sqrt3, -0.5*sqrt3, 0.5*sqrt3.
    # This is 11 hexagons.
    # Let's check overlaps.
    # Distance between (0,0) and (1.5*sqrt3, 1.5) is sqrt(3*1.5^2 + 1.5^2) = 1.5*sqrt(4) = 3. OK.
    # Distance between (0,0) and (sqrt3, 0) is sqrt3 > 2. OK.
    # This layout is "taller" but might fit better in a hexagon?
    pos_v4 = []
    for i in range(-2, 3):
        pos_v4.append((i * _SQRT3, 0.0))
    for i in range(-1, 2):
        pos_v4.append((i * _SQRT3 + _SQRT3/2, 1.5))
    for i in range(-1, 2):
        pos_v4.append((i * _SQRT3 + _SQRT3/2, 3.0))
    layouts.append(pos_v4)
    
    # Variation 5: Same as V4 but rotated 90 degrees?
    pos_v5 = []
    for i in range(-2, 3):
        pos_v5.append((0.0, i * _SQRT3))
    for i in range(-1, 2):
        pos_v5.append((-_SQRT3/2 - i*_SQRT3, 1.5)) # Rotation
    # Let's just rotate V4 by 90 deg programmatically later if needed, or just add another symmetric one.
    # Let's try a "T" shape or something else.
    # For now, V1-V4 should be enough.
    
    return layouts

def construct():
    if time.time() > _DEADLINE - 5.0:
        # Emergency fallback
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_layout_idx = -1
    best_rel_angle = 0.0

    layouts = _build_rigid_layouts()
    
    # For each rigid layout, we optimize the relative angle between the cluster and the outer hexagon.
    # The outer hexagon orientation is phi. The cluster orientation is theta.
    # Due to symmetry, we only need to search phi - theta in [0, 60).
    # We can fix theta = 0 and search phi in [0, 60).
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break
            
        # Pre-compute vertices for theta=0
        # We need to test different relative angles.
        # Let's sweep phi (outer angle) from 0 to 60.
        # For each phi, we compute the min enclosing hex side.
        
        coarse_best_side = float('inf')
        coarse_best_phi = 0.0
        
        # Coarse sweep
        for phi_deg in np.arange(0, 60.0, 0.5):
            if time.time() > _DEADLINE - 10.0:
                break
            # Vertices of all 11 hexagons (rotated by 0, translated by 0)
            # We need to pass the vertices to the LP solver.
            # The LP solver finds the center and side for a FIXED outer orientation.
            # So we just need the vertices of the inner hexagons.
            # Since the cluster is rigid and we fixed its orientation to 0, the vertices are static.
            # We only change the outer hexagon orientation phi.
            
            all_vertices = np.vstack([_hex_vertices(px, py, 0.0) for px, py in positions])
            
            cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, phi_deg)
            if side < coarse_best_side:
                coarse_best_side = side
                coarse_best_phi = phi_deg
        
        if coarse_best_side < best_side:
            # Refine around coarse_best_phi
            best_side = coarse_best_side
            best_outer_angle = coarse_best_phi
            best_layout_idx = layout_idx
            best_center, best_side, best_outer_angle = _min_enclosing_hex_side_lp(
                np.vstack([_hex_vertices(px, py, 0.0) for px, py in positions]), 
                coarse_best_phi
            )
            best_inner = [[px, py, 0.0] for px, py in positions]
            
            # Fine sweep
            for phi_deg in np.arange(max(0, coarse_best_phi - 0.5), min(60.0, coarse_best_phi + 0.5), 0.01):
                if time.time() > _DEADLINE - 10.0:
                    break
                all_vertices = np.vstack([_hex_vertices(px, py, 0.0) for px, py in positions])
                cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, phi_deg)
                if side < best_side:
                    best_side = side
                    best_outer_angle = phi_deg
                    best_center = (cx, cy)
                    best_inner = [[px, py, 0.0] for px, py in positions]
                    best_layout_idx = layout_idx

    # If we found a good rigid solution, we can try to "relax" it slightly?
    # No, the directive says keep the design family (rigid body).
    # But we can try different global rotations of the cluster?
    # No, relative angle covers that.
    
    # One more check: What if the optimal cluster orientation is not 0?
    # It doesn't matter, because we are sweeping the relative angle.
    # If the optimal is cluster at 10 deg and outer at 40 deg, 
    # it is equivalent to cluster at 0 deg and outer at 30 deg (relative 30).
    # So fixing cluster to 0 is sufficient.

    # Safety check
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Verify no overlaps (should be fine for rigid lattice layouts)
    # Just in case, check distances.
    positions = [(p[0], p[1]) for p in best_inner]
    for i in range(N):
        for j in range(i+1, N):
            d = math.dist(positions[i], positions[j])
            if d < 2.0 - 1e-5:
                # Overlap! This shouldn't happen with valid lattice layouts.
                # Fallback
                inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
                return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_outer_angle
