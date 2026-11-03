"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import linprog, minimize
import time

N = 11

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy), side length, rotated by angle_deg."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def min_enclosing_hexagon(inner_pts, inner_angles, deadline):
    """
    Find the minimum side length regular hexagon that encloses all inner hexagons.
    inner_pts: (n, 2) centers
    inner_angles: (n,) angles in degrees
    Returns (side_length, (cx, cy), angle_deg) of enclosing hexagon.
    """
    n_inner = len(inner_pts)
    sqrt3_2 = np.sqrt(3) / 2.0
    
    # Collect all inner hexagon vertices
    all_verts = []
    for k in range(n_inner):
        verts = hex_vertices(inner_pts[k, 0], inner_pts[k, 1], 1.0, inner_angles[k])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)  # (6*n_inner, 2)
    m = len(all_verts)
    
    best_side = np.inf
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try a range of outer angles (0 to 60 degrees, since hexagon has 6-fold symmetry)
    for outer_angle_deg in range(0, 60, 3):  # finer grid
        if time.time() > deadline:
            break
        outer_angle = np.radians(outer_angle_deg)
        
        # 6 outward normals for the enclosing hexagon
        normals = np.array([[np.cos(outer_angle + i * np.pi / 3), 
                            np.sin(outer_angle + i * np.pi / 3)] for i in range(6)])
        
        # LP: minimize s
        # Variables: x = [cx, cy, s]
        # Constraints: for each vertex v_i and each normal n_j:
        #   n_j · (v_i - c) <= s * sqrt(3)/2
        #   => -n_j[0]*cx - n_j[1]*cy + (sqrt(3)/2)*s >= -n_j · v_i
        #   => n_j[0]*cx + n_j[1]*cy - (sqrt(3)/2)*s <= n_j · v_i
        #
        # A_ub * x <= b_ub
        # A_ub row: [n_j[0], n_j[1], -sqrt3_2]
        # b_ub: n_j · v_i
        
        A_ub = np.zeros((m * 6, 3))
        b_ub = np.zeros(m * 6)
        
        for i in range(m):
            vx, vy = all_verts[i]
            for j in range(6):
                row = i * 6 + j
                A_ub[row, 0] = normals[j, 0]
                A_ub[row, 1] = normals[j, 1]
                A_ub[row, 2] = -sqrt3_2
                b_ub[row] = vx * normals[j, 0] + vy * normals[j, 1]
        
        c = np.array([0.0, 0.0, 1.0])  # minimize s
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs', 
                           options={'time_limit': 5.0})
            if result.success and result.x[2] < best_side:
                best_side = result.x[2]
                best_center = (result.x[0], result.x[1])
                best_angle = outer_angle_deg
        except Exception:
            pass
    
    if not np.isfinite(best_side):
        # Fallback
        xs = all_verts[:, 0]
        ys = all_verts[:, 1]
        best_side = max(xs.max() - xs.min(), ys.max() - ys.min()) / np.sqrt(3)
        best_center = ((xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2)
        best_angle = 0.0
    
    return best_side, best_center, best_angle

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    sqrt3 = np.sqrt(3)
    
    # Initial configuration: compact hexagonal packing
    # For unit regular hexagons, minimum center distance for non-overlap is sqrt(3)
    # (when sharing an edge, apothem = sqrt(3)/2, so center-to-center = 2*sqrt(3)/2 = sqrt(3))
    
    spacing = sqrt3  # exact touching
    
    # Place 11 hexagons: center + 6 in first ring + 4 in second ring
    pos = []
    pos.append((0.0, 0.0))
    
    # First ring: 6 hexagons at distance spacing, angles 0, 60, 120, 180, 240, 300
    for i in range(6):
        angle = i * np.pi / 3
        x = spacing * np.cos(angle)
        y = spacing * np.sin(angle)
        pos.append((x, y))
    
    # Second ring: 4 more hexagons
    # Place at the "corner" positions of the hexagonal lattice
    # In a hex lattice, second ring positions are at distance 2*spacing
    # at angles 30, 90, 150, 210, 270, 330 (6 positions)
    # We only need 4, so pick 4 of them
    for i in [0, 1, 2, 3]:
        angle = (2*i + 1) * np.pi / 6  # 30, 90, 150, 210 degrees
        r = 2 * spacing
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        pos.append((x, y))
    
    # That's 1 + 6 + 4 = 11
    assert len(pos) == N, f"Expected {N}, got {len(pos)}"
    
    inner_angles = np.zeros(N)  # all axis-aligned initially
    
    initial_params = np.array(pos).flatten()
    
    # Compute initial enclosing hexagon
    init_pts = np.array(pos)
    best_side, best_center, best_angle = min_enclosing_hexagon(
        init_pts, inner_angles, deadline
    )
    
    best_params = initial_params.copy()
    best_angles = inner_angles.copy()
    best_result = (best_side, best_center, best_angle)
    
    # Multi-start L-BFGS-B optimization
    n_restarts = 8
    np.random.seed(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            start_params = initial_params.copy()
            start_angles = inner_angles.copy()
        else:
            # Perturb
            perturbation = 0.3 + 0.1 * restart
            start_params = initial_params + np.random.randn(N * 2) * perturbation
            start_angles = np.mod(inner_angles + np.random.randn(N) * 20.0, 360.0)
        
        # Optimize positions with fixed angles
        def obj_pos(p):
            if time.time() > deadline:
                return 1e10
            pts = p.reshape(N, 2)
            s, c, a = min_enclosing_hexagon(pts, start_angles, deadline)
            return s
        
        try:
            result = minimize(obj_pos, start_params, method='L-BFGS-B', 
                             options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5})
            if result.fun < best_result[0]:
                # Get the enclosing hexagon for this configuration
                pts = result.x.reshape(N, 2)
                s, c, a = min_enclosing_hexagon(pts, start_angles, deadline)
                best_result = (s, c, a)
                best_params = result.x.copy()
                best_angles = start_angles.copy()
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    # Also try optimizing angles for the best position configuration
    if time.time() < deadline - 20:
        def obj_angles(params_flat):
            if time.time() > deadline:
                return 1e10
            angles = params_flat
            pts = best_params.reshape(N, 2)
            s, c, a = min_enclosing_hexagon(pts, angles, deadline)
            return s
        
        try:
            result = minimize(obj_angles, best_angles.copy(), method='L-BFGS-B',
                             options={'maxiter': 30, 'ftol': 1e-8, 'gtol': 1e-5})
            if result.fun < best_result[0]:
                angles = np.mod(result.x, 360.0)
                pts = best_params.reshape(N, 2)
                s, c, a = min_enclosing_hexagon(pts, angles, deadline)
                best_result = (s, c, a)
                best_angles = angles.copy()
        except Exception:
            pass
    
    # Build final result
    final_side = best_result[0]
    final_center = best_result[1]
    final_angle = best_result[2]
    
    inner = []
    for k in range(N):
        x = best_params[k * 2]
        y = best_params[k * 2 + 1]
        a = best_angles[k]
        inner.append([float(x), float(y), float(a)])
    
    return inner, (float(final_center[0]), float(final_center[1])), float(final_side), float(final_angle)

# EVOLVE-BLOCK-END
