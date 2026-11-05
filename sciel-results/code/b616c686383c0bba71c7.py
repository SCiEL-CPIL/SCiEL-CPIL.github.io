import numpy as np
import time
import math
import random
from scipy.optimize import minimize, linprog

N = 12
DEADLINE = 140.0  # Safety margin below 150s kill

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with rotation rot_deg."""
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def sat_overlap(v1, v2):
    """
    Returns separation distance (positive) if not overlapping, or negative penetration if overlapping.
    Uses Separating Axis Theorem for convex polygons.
    """
    axes = []
    for v in (v1, v2):
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i+1) % len(v)]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                axes.append(n)
    
    min_overlap = float('inf')
    for n in axes:
        proj1 = v1 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        proj2 = v2 @ n
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap  # positive = separated
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    return -min_overlap  # negative = overlapping

def outer_hex_side(points, outer_rot_deg):
    """
    Minimum side length of regular hexagon centered at origin with given rotation
    that encloses all points.
    """
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    # Apothem = (sqrt(3)/2) * side
    # max_proj is the max distance from center to a side (apothem)
    # So side = 2 * apothem / sqrt(3)
    return (2.0 / np.sqrt(3.0)) * max_proj

def outer_hex_side_linprog(params):
    """
    Compute outer hexagon side length using LP for optimal centering.
    params: (N,3) array of [x, y, rot] for each hexagon + [outer_rot]
    Returns: (L, cx, cy, outer_rot) where L is side length
    """
    xys_rots = params[:3*N].reshape(N, 3)
    outer_rot = params[3*N] if len(params) > 3*N else 0.0
    
    # Get all vertices
    all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
    n_pts = len(all_points)
    
    # Outer hexagon normals (6 directions)
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # For each point and each normal, compute projection
    # proj_i_k = p_i . n_k
    # We need: -t <= cx*n_kx + cy*n_ky + proj_i_k <= t for all i,k
    # This gives 2*n_pts*6 linear constraints on (cx, cy, t)
    # Minimize t, then L = 2*t/sqrt(3)
    
    # A matrix: each row is [n_kx, n_ky, -1] for constraint cx*n_kx + cy*n_ky - t <= -proj_i_k
    # and [n_kx, n_ky, 1] for constraint cx*n_kx + cy*n_ky + t >= proj_i_k => -cx*n_kx - cy*n_ky - t <= -proj_i_k
    
    A_rows = []
    b_vals = []
    
    for k in range(6):
        nx, ny = normals[k]
        for i in range(n_pts):
            px, py = all_points[i]
            proj = px * nx + py * ny
            # cx*nx + cy*ny - t <= -proj
            A_rows.append([nx, ny, -1.0])
            b_vals.append(-proj)
            # -cx*nx - cy*ny - t <= -proj  (from cx*nx + cy*ny + t >= proj)
            A_rows.append([-nx, -ny, -1.0])
            b_vals.append(-proj)
    
    A = np.array(A_rows)
    b = np.array(b_vals)
    c = np.array([0.0, 0.0, 1.0])  # minimize t
    
    res = linprog(c, A_ub=A, b_ub=b, method='highs')
    
    if res.success:
        t = res.x[2]
        L = (2.0 / np.sqrt(3.0)) * t
        cx, cy = res.x[0], res.x[1]
        return L, cx, cy, outer_rot
    else:
        # Fallback to centered version
        L = outer_hex_side(all_points, outer_rot)
        return L, 0.0, 0.0, outer_rot

def smooth_overlap_penalty(params, beta=50.0):
    """
    Compute smooth overlap penalty using log-sum-exp for numerical stability.
    Returns a smooth approximation of the sum of squared penetrations.
    """
    xys_rots = params[:3*N].reshape(N, 3)
    centers = xys_rots[:, :2]
    
    # Precompute vertices for all hexagons
    vertices_list = []
    for i in range(N):
        cx, cy, rot = xys_rots[i]
        v = get_hex_vertices(cx, cy, 1.0, rot)
        vertices_list.append(v)
    
    # Compute pairwise penetrations
    penetrations = []
    for i in range(N):
        for j in range(i+1, N):
            # Quick distance check
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 2.0:  # Only compute SAT if close enough
                d_sat = sat_overlap(vertices_list[i], vertices_list[j])
                if d_sat < 0:
                    penetrations.append(-d_sat)
    
    if len(penetrations) == 0:
        return 0.0
    
    # Use log-sum-exp for smooth approximation of sum of squares
    p_arr = np.array(penetrations)
    scaled = 2.0 * p_arr / beta
    max_s = np.max(scaled)
    if max_s > 500:  # Avoid overflow
        return np.sum(p_arr**2)
    lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    return beta * lse

def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)

    def honeycomb_rigid_search():
        """Search over rigid honeycomb cluster orientation and translation."""
        # Define honeycomb topology: 12 hexagons in a compact arrangement
        # Using a 3x4 grid with hex offset pattern
        sqrt3 = np.sqrt(3)
        base_positions = []
        # 3 rows of 4, with offset on odd rows
        for r in range(3):
            for c in range(4):
                x = c * sqrt3
                y = r * 1.5
                if r % 2 == 1:
                    x += sqrt3 / 2.0
                base_positions.append([x, y])
        base_positions = np.array(base_positions)
        # Center the cluster
        base_positions -= np.mean(base_positions, axis=0)
        
        best_L_local = float('inf')
        best_params_local = None
        
        # Search over rotation angles
        for rot_deg in np.arange(0, 60, 2.0):
            if time.time() - start_time > DEADLINE - 35:
                break
            
            # Rotate base positions
            rot_rad = np.radians(rot_deg)
            cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
            rotated = np.column_stack((
                base_positions[:, 0] * cos_r - base_positions[:, 1] * sin_r,
                base_positions[:, 0] * sin_r + base_positions[:, 1] * cos_r
            ))
            
            # All inner hexagons share the same rotation
            inner_rots = np.full(N, rot_deg)
            
            # Try several translations to center the cluster optimally
            for tx in np.arange(-1.0, 1.0, 0.5):
                for ty in np.arange(-1.0, 1.0, 0.5):
                    if time.time() - start_time > DEADLINE - 35:
                        break
                    
                    centers = rotated + np.array([tx, ty])
                    xys_rots = np.column_stack((centers[:, 0], centers[:, 1], inner_rots))
                    
                    # Check validity
                    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
                    valid = True
                    for i in range(N):
                        for j in range(i+1, N):
                            d = sat_overlap(vertices_list[i], vertices_list[j])
                            if d < -1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if not valid:
                        continue
                    
                    # Try several outer rotations
                    for outer_rot in [0.0, 15.0, 30.0, 45.0]:
                        all_points = np.vstack(vertices_list)
                        L = outer_hex_side(all_points, outer_rot)
                        if L < best_L_local:
                            best_L_local = L
                            best_params_local = np.concatenate((xys_rots.flatten(), [outer_rot]))
        
        return best_L_local, best_params_local
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def objective(params):
        if time.time() - start_time > DEADLINE - 5:
            return 1e6
        
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        
        # Use smooth penalty with adaptive beta for better gradient information
        # Start with higher beta for smoother landscape, reduce for tighter fit
        elapsed = time.time() - start_time
        if elapsed < 60:
            beta = 80.0
        elif elapsed < 100:
            beta = 40.0
        else:
            beta = 20.0
        penalty = smooth_overlap_penalty(params, beta=beta)
        
        # Compute outer side length
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
        L = outer_hex_side(all_points, outer_rot)
        
        # Weight penalty more heavily to ensure validity during optimization
        penalty_weight = 50.0 if elapsed < 80 else 100.0
        return L + penalty_weight * penalty
    
    def make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
        """Create a hex grid of unit hexagons with optional gap."""
        pts = []
        sqrt3 = np.sqrt(3)
        spacing_x = sqrt3 + gap
        spacing_y = 1.5 + gap * 0.5
        for r in range(rows):
            y = r * spacing_y
            for c in range(cols):
                x = c * spacing_x
                if r % 2 == 1:
                    x += spacing_x / 2.0
                pts.append([x, y, rot_offset])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    def make_ring(gap=0.0):
        """1 center + 6 ring + 5 outer with gap."""
        pts = []
        sqrt3 = np.sqrt(3)
        d1 = sqrt3 + gap
        d2 = 2 * sqrt3 + gap
        pts.append([0, 0, 0.0])
        for k in range(6):
            ang = np.radians(60 * k)
            pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
        for k in range(5):
            ang = np.radians(60 * k + 30)
            pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    def make_ring12(gap=0.0):
        """12 hexagons in a ring around center."""
        pts = []
        R = 2.5 + gap
        for i in range(N):
            theta = 2 * np.pi * i / N
            x = R * np.cos(theta)
            y = R * np.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    def is_valid(params, tol=1e-6):
        """Check if configuration has no overlaps."""
        xys_rots = params[:3*N].reshape(N, 3)
        vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
        
        for i in range(N):
            for j in range(i+1, N):
                d = sat_overlap(vertices_list[i], vertices_list[j])
                if d < -tol:
                    return False
        return True
    
    def get_L(params):
        """Compute outer side length for a valid configuration."""
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
        return outer_hex_side(all_points, outer_rot)
    
    def full_score(cfg, angle):
        """Return outer side if valid, else None."""
        params = np.concatenate((cfg.flatten(), [angle]))
        if not is_valid(params):
            return None
        return get_L(params)
    
    best_L = float('inf')
    best_params = None
    
    # Generate diverse initial configurations with a small gap to ensure validity
    configs = []
    gap = 0.05
    configs.append(make_hex_grid(4, 3, gap=gap))
    configs.append(make_hex_grid(3, 4, gap=gap))
    configs.append(make_hex_grid(2, 6, gap=gap))
    configs.append(make_hex_grid(6, 2, gap=gap))
    configs.append(make_ring(gap=gap))
    configs.append(make_ring12(gap=gap))
    
    # Add rotated variants
    for base_cfg in configs[:4]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)
    
    outer_rots = [0.0, 30.0, 15.0, 45.0]
    
    # Phase 0: Honeycomb rigid body search
    if time.time() - start_time < DEADLINE - 35:
        hcb_L, hcb_params = honeycomb_rigid_search()
        if hcb_params is not None and hcb_L < best_L:
            best_L = hcb_L
            best_params = hcb_params.copy()
    
    # Phase 1: Evaluate initial configurations
    for cfg in configs:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        for orot in outer_rots:
            if time.time() - start_time > DEADLINE - 30:
                break
            
            init_params = np.concatenate((cfg.flatten(), [orot]))
            
            if is_valid(init_params):
                L_val = get_L(init_params)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()
    
    # Phase 2: Simulated annealing for global exploration
    if time.time() - start_time < DEADLINE - 40:
        # Start from best initial config
        if best_params is not None:
            cfg = best_params[:3*N].reshape(N, 3).copy()
            angle = best_params[3*N]
            cur_score = best_L
            
            T = 0.8
            T_end = 0.003
            alpha = 0.9997
            
            while time.time() - start_time < DEADLINE - 40:
                r = random.random()
                
                if r < 0.50:
                    # Move one hexagon
                    idx = random.randrange(N)
                    old_x, old_y = cfg[idx][0], cfg[idx][1]
                    step = 0.2 * (1.0 + 0.5 * T)
                    cfg[idx][0] += random.gauss(0, step)
                    cfg[idx][1] += random.gauss(0, step)
                elif r < 0.70:
                    # Rotate one hexagon
                    idx = random.randrange(N)
                    old_ang = cfg[idx][2]
                    cfg[idx][2] += random.gauss(0, 5.0 + 3.0 * T)
                elif r < 0.80:
                    # Rotate outer
                    old_angle = angle
                    angle += random.gauss(0, 2.0 + 2.0 * T)
                else:
                    # Perturb two hexagons
                    idx1, idx2 = random.sample(range(N), 2)
                    old_x1, old_y1 = cfg[idx1][0], cfg[idx1][1]
                    old_x2, old_y2 = cfg[idx2][0], cfg[idx2][1]
                    step = 0.15 * (1.0 + 0.5 * T)
                    cfg[idx1][0] += random.gauss(0, step)
                    cfg[idx1][1] += random.gauss(0, step)
                    cfg[idx2][0] += random.gauss(0, step)
                    cfg[idx2][1] += random.gauss(0, step)
                
                new_score = full_score(cfg, angle)
                
                if new_score is not None:
                    if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                        cur_score = new_score
                        if new_score < best_L:
                            best_L = new_score
                            best_params = np.concatenate((cfg.flatten(), [angle])).copy()
                    else:
                        # Revert
                        if r < 0.50:
                            cfg[idx][0], cfg[idx][1] = old_x, old_y
                        elif r < 0.70:
                            cfg[idx][2] = old_ang
                        elif r < 0.80:
                            angle = old_angle
                        else:
                            cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                            cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
                else:
                    # Invalid: revert
                    if r < 0.50:
                        cfg[idx][0], cfg[idx][1] = old_x, old_y
                    elif r < 0.70:
                        cfg[idx][2] = old_ang
                    elif r < 0.80:
                        angle = old_angle
                    else:
                        cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                        cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
                
                T *= alpha
                if T < T_end:
                    break
    
    # Phase 3: Gradient-based refinement from best SA result
    if best_params is not None and time.time() - start_time < DEADLINE - 25:
        try:
            # First pass: L-BFGS-B with moderate settings
            res_lbgfs = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                options={
                    'maxiter': 800,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxfun': 2000
                }
            )
            
            if is_valid(res_lbgfs.x):
                L_val = get_L(res_lbgfs.x)
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_lbgfs.x.copy()
        except Exception:
            pass
        
        # Second pass: tighter refinement if time permits
        if best_params is not None and time.time() - start_time < DEADLINE - 15:
            try:
                res_lbgfs2 = minimize(
                    objective,
                    best_params,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 1000,
                        'ftol': 1e-15,
                        'gtol': 1e-13,
                        'maxfun': 3000
                    }
                )
                
                if is_valid(res_lbgfs2.x):
                    L_val = get_L(res_lbgfs2.x)
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res_lbgfs2.x.copy()
            except Exception:
                pass
        
        # Phase 3.5: LP-based centering optimization
        # For the current best configuration, use LP to find optimal outer hexagon center
        if best_params is not None and time.time() - start_time < DEADLINE - 10:
            try:
                xys_rots = best_params[:3*N].reshape(N, 3)
                outer_rot = best_params[3*N]
                
                # Try several outer rotations with LP centering
                for orot_candidate in [outer_rot, outer_rot + 15.0, outer_rot - 15.0, 0.0, 30.0]:
                    test_params = np.concatenate((xys_rots.flatten(), [orot_candidate]))
                    L_lp, cx, cy, _ = outer_hex_side_linprog(test_params)
                    
                    if L_lp < best_L:
                        best_L = L_lp
                        best_params = test_params.copy()
            except Exception:
                pass
    
    # Phase 4: COBYLA for unconstrained polish
    if best_params is not None and time.time() - start_time < DEADLINE - 12:
        try:
            res_cobyla = minimize(
                objective,
                best_params,
                method='COBYLA',
                options={
                    'maxiter': 3000,
                    'rhobeg': 0.005,
                    'tol': 1e-12
                }
            )
            
            if is_valid(res_cobyla.x):
                L_val = get_L(res_cobyla.x)
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_cobyla.x.copy()
        except Exception:
            pass
    
    # Phase 5: Nelder-Mead for final polish
    if best_params is not None and time.time() - start_time < DEADLINE - 5:
        try:
            res_nm = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 8000,
                    'xatol': 1e-11,
                    'fatol': 1e-11,
                    'adaptive': True
                }
            )
            
            if is_valid(res_nm.x):
                L_val = get_L(res_nm.x)
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_nm.x.copy()
        except Exception:
            pass
        
        # Phase 6: Final exact local solve with very tight tolerances
        if best_params is not None and time.time() - start_time < DEADLINE - 2:
            try:
                res_final = minimize(
                    objective,
                    best_params,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 2000,
                        'ftol': 1e-16,
                        'gtol': 1e-14,
                        'maxfun': 5000
                    }
                )
                
                if is_valid(res_final.x):
                    L_val = get_L(res_final.x)
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res_final.x.copy()
            except Exception:
                pass
    
    if best_params is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Extract final result
    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    
    # Use LP to find optimal center for final configuration
    try:
        L_final, cx_final, cy_final, _ = outer_hex_side_linprog(best_params)
        if L_final <= best_L + 1e-10:
            best_L = L_final
    except Exception:
        cx_final, cy_final = 0.0, 0.0
    
    inner = []
    for x, y, rot in xys_rots:
        inner.append([x - cx_final, y - cy_final, rot])
    
    return inner, (cx_final, cy_final), best_L, outer_rot

# EVOLVE-BLOCK-END
