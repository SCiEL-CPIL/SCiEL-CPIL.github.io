import math
import time
import numpy as np
from scipy.optimize import linprog, minimize_scalar, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with vertex at outer_angle_deg)
    that contains all points in verts (numpy array of shape (M,2)).
    Returns (cx, cy, side) or None.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    vdots = verts @ normals.T  # shape (M, 6)
    
    A_rows = []
    b_vals = []
    
    for j in range(6):
        nx, ny = normals[j]
        A_rows.append(np.tile([-nx, -ny, -1.0], (M, 1)))
        b_vals.append(-vdots[:, j])
        A_rows.append(np.tile([nx, ny, -1.0], (M, 1)))
        b_vals.append(vdots[:, j])
    
    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)
    
    A = np.vstack([A, np.array([[0.0, 0.0, -1.0]])])
    b = np.concatenate([b, [0.0]])
    
    c_obj = np.array([0.0, 0.0, 1.0])
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs', options={'presolve': True, 'dual_feasibility_tolerance': 1e-12, 'primal_feasibility_tolerance': 1e-12})
        if res.success:
            cx, cy, R = res.x
            side = R * _INVSQRT3 * 2.0
            return cx, cy, side
    except Exception:
        pass
    return None

def _objective_for_angle(verts, outer_angle_deg):
    """Return the side length for a given outer angle, or a large penalty."""
    res = _min_enclosing_hex_side(verts, outer_angle_deg)
    if res is None:
        return 1e6
    return res[2]

def _smooth_objective_for_angle(verts, outer_angle_deg, beta=50.0):
    """
    Smooth surrogate for the outer side length using log-sum-exp.
    For a given outer angle, the minimum enclosing hex side is determined by
    the max over all vertex-normal pairs of (dot - center_dot). We approximate
    this max with a soft-max to get a differentiable objective.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    vdots = verts @ normals.T  # shape (M, 6)
    
    # The constraint is: for each face j, max_i(vdots[i,j]) - min_i(vdots[i,j]) <= 2*R
    # where R is the inradius. Side = 2*R/sqrt(3).
    # Actually, the LP finds cx, cy, R such that:
    #   nx_j * cx + ny_j * cy - R <= vdots[i,j] for all i
    #  -nx_j * cx - ny_j * cy - R <= -vdots[i,j] for all i
    # This means: R >= |nx_j*cx + ny_j*cy - vdots[i,j]| for all i,j
    # The optimal R for a given (cx,cy) is max_{i,j} |nx_j*cx + ny_j*cy - vdots[i,j]|
    # And we also minimize over cx,cy.
    #
    # For the smooth surrogate, we fix the center at the centroid and compute
    # a soft-max of the distances.
    
    # Better approach: use the LP result but with a smooth approximation.
    # Actually, let's just use the exact LP but call it from minimize.
    # The issue is that minimize needs a smooth function.
    # 
    # Alternative: parameterize by the 6 face offsets and use soft-max.
    # For a regular hexagon with fixed rotation, the side length is determined by
    # the maximum extent in each of the 6 normal directions.
    # side = (2/sqrt(3)) * max_j (max_i vdot_ij - min_i vdot_ij) / 2 * ... 
    # Actually: the width in direction j is max_i(vdots[:,j]) - min_i(vdots[:,j])
    # The inradius R must satisfy R >= width_j / 2 for all j
    # So R = max_j(width_j / 2), and side = 2*R/sqrt(3)
    
    widths = np.max(vdots, axis=0) - np.min(vdots, axis=0)  # shape (6,)
    R_vals = widths / 2.0
    
    # Soft-max approximation of max(R_vals)
    max_R = np.max(R_vals)
    shifted = R_vals - max_R
    log_sum_exp = max_R + np.log(np.sum(np.exp(beta * shifted))) / beta
    
    side = 2.0 * log_sum_exp * _INVSQRT3
    return side

def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    sqrt3 = _SQRT3
    scale = 1.0000001
    u = (sqrt3*scale, 0.0)
    v = (sqrt3/2*scale, 1.5*scale)
    
    coords = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i*u[0] + j*v[0]
            y = i*u[1] + j*v[1]
            coords.append((x, y, i, j))
    
    if layout_idx == 0:
        target = [(0,0)]
        ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        target.extend(ring1)
        ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)]
        ring2_sorted = sorted(ring2, key=lambda p: p[0]**2 + p[1]**2)
        target.extend(ring2_sorted[:4])
        positions = []
        for ti, tj in target:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x,y))
        return positions[:N]
    elif layout_idx == 1:
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 3):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 2:
        positions = []
        y1 = -1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 3:
        positions = []
        y1 = -3.0*scale
        for k in range(0, 2):
            x = k*sqrt3*scale - sqrt3/2*scale
            positions.append((x, y1))
        y2 = -1.5*scale
        for k in range(-1, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 1):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        y4 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 4:
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        return positions[:N]
    else:
        coords_sorted = sorted(coords, key=lambda t: t[0]**2 + t[1]**2)
        return [(x,y) for x,y,i,j in coords_sorted[:N]]

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_inner_angle = 0.0
    
    num_layouts = 6
    layouts = [_build_positions(i) for i in range(num_layouts)]
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        for inner_angle in [0.0, 30.0, 15.0, -15.0]:
            if time.time() > _DEADLINE - 10.0:
                break
            
            all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
            
            # Coarse grid search to find good starting points
            coarse_results = []
            for outer_angle_deg in np.arange(0, 30.25, 0.5):
                if time.time() > _DEADLINE - 10.0:
                    break
                res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                if res:
                    coarse_results.append((res[2], outer_angle_deg))
            
            if not coarse_results:
                continue
            
            # Find the best coarse result and its neighbors for local optimization
            coarse_results.sort(key=lambda x: x[0])
            best_coarse_side, best_coarse_angle = coarse_results[0]
            
            # Use smooth surrogate + L-BFGS-B to refine the outer angle
            best_local_side = best_coarse_side
            best_local_angle = best_coarse_angle
            best_local_center = None
            
            # Try smooth optimization from multiple starting points
            candidate_angles = [best_coarse_angle]
            for side, angle in coarse_results[:3]:
                if abs(angle - best_coarse_angle) > 1e-6:
                    candidate_angles.append(angle)
            
            for start_angle in candidate_angles:
                if time.time() > _DEADLINE - 15.0:
                    break
                
                # Use smooth surrogate with L-BFGS-B
                try:
                    res_opt = minimize(
                        lambda a: _smooth_objective_for_angle(all_vertices, a[0], beta=80.0),
                        x0=[start_angle],
                        method='L-BFGS-B',
                        bounds=[(0.0, 30.0)],
                        options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
                    )
                    if res_opt.success:
                        opt_angle = res_opt.x[0]
                        # Evaluate exact objective at the smooth optimum
                        opt_side_exact = _objective_for_angle(all_vertices, opt_angle)
                        if opt_side_exact < best_local_side:
                            best_local_side = opt_side_exact
                            best_local_angle = opt_angle
                except Exception:
                    pass
                
                # Also try minimize_scalar with the exact objective for comparison
                lo = max(0.0, start_angle - 5.0)
                hi = min(30.0, start_angle + 5.0)
                try:
                    res_opt2 = minimize_scalar(
                        lambda a: _objective_for_angle(all_vertices, a),
                        bounds=(lo, hi),
                        method='bounded',
                        options={'xatol': 1e-14, 'maxiter': 300}
                    )
                    if res_opt2.success:
                        opt_angle2 = res_opt2.x
                        opt_side2 = res_opt2.fun
                        if opt_side2 < best_local_side:
                            best_local_side = opt_side2
                            best_local_angle = opt_angle2
                except Exception:
                    pass
            
            # Final high-precision polish: very fine grid around the best angle
            if best_local_side < best_side and time.time() < _DEADLINE - 20.0:
                # Multi-scale refinement
                for delta_max, step in [(0.01, 0.001), (0.002, 0.0001), (0.0005, 0.00002)]:
                    if time.time() > _DEADLINE - 15.0:
                        break
                    lo = max(0.0, best_local_angle - delta_max)
                    hi = min(30.0, best_local_angle + delta_max)
                    for outer_angle_deg in np.arange(lo, hi + step/2, step):
                        if time.time() > _DEADLINE - 15.0:
                            break
                        res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                        if res:
                            cx, cy, side = res
                            if side < best_local_side:
                                best_local_side = side
                                best_local_angle = outer_angle_deg
                                best_local_center = (cx, cy)
            
            # Get the center for the best angle
            if best_local_side < best_side:
                res_final = _min_enclosing_hex_side(all_vertices, best_local_angle)
                if res_final:
                    cx, cy, side = res_final
                    if side < best_side:
                        best_side = side
                        best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                        best_center = (cx, cy)
                        best_outer_angle = best_local_angle
                        best_inner_angle = inner_angle
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
