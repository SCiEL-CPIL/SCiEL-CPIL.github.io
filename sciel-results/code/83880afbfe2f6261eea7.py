import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, linprog, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 12.0

def hex_support(alpha, direction_angle):
    """
    Compute the support function of a unit regular hexagon with rotation alpha
    in the given direction.
    For a regular hexagon of side 1 (circumradius 1), vertices are at angles
    alpha + k*pi/3 for k=0..5.
    Support in direction theta = max_k cos(theta - alpha - k*pi/3)
    """
    diff = (direction_angle - alpha) % (math.pi / 3.0)
    # The max of cos over the 6 directions
    # cos is symmetric, so we can reduce to [0, pi/6] range
    d = min(diff, math.pi / 3.0 - diff)
    return math.cos(d)

def hex_support_vec(alpha, dir_angles):
    """Vectorized support function for multiple directions."""
    diffs = (dir_angles - alpha) % (math.pi / 3.0)
    d = np.minimum(diffs, math.pi / 3.0 - diffs)
    return np.cos(d)

def construct():
    # Check if we are already close to the time limit
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Generate 11 points on a triangular lattice
    # Axial coordinates (q, r)
    # This is a compact "flower" shape: center + 6 neighbors + 4 from the next ring
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    # Use a very small expansion to ensure strict disjointness (no touching)
    # while keeping the size minimal.
    SCALE = 1.0 + 1e-6
    
    pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0) * SCALE
        y = 1.5 * r * SCALE
        pts.append((x, y))
        
    pts = np.array(pts)
    n_pts = len(pts)
    
    # Precompute the 12 constraint rows per inner hex per outer normal
    # For each inner hex i and outer normal k, we have 2 constraints
    # We'll precompute the coefficient structure
    
    def solve_lp(beta, inner_angles, time_limit=TIME_LIMIT - SAFETY_MARGIN):
        """
        Solves the LP for a given outer rotation beta and inner hexagon rotations.
        Returns (cx, cy, S) or (None, None, inf)
        """
        # Check time limit
        if time.time() - START_TIME > time_limit:
            return None, None, float('inf')
        
        # Normals of the outer hexagon
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # Precompute direction angles for all normals
        normal_angles = [math.atan2(ny, nx) for nx, ny in normals]
        
        # Variables: cx, cy, D
        # D is the max over i,k of |(p_i - c) . n_k| + h_i(n_k)
        # where h_i(n_k) is the support of inner hex i in direction n_k
        # Constraint: (p_i - c) . n_k + h_i(n_k) <= S * APOTHEM
        # and -(p_i - c) . n_k + h_i(n_k) <= S * APOTHEM
        # Let D = S * APOTHEM
        # Then: (p_i - c) . n_k + h_i(n_k) <= D
        #       -(p_i - c) . n_k + h_i(n_k) <= D
        
        c_obj = np.array([0.0, 0.0, 1.0]) # Minimize D
        
        A_ub = []
        b_ub = []
        
        for i, (px, py) in enumerate(pts):
            alpha_i = inner_angles[i]
            # Vectorized support computation for all 6 normals
            h_vals = hex_support_vec(alpha_i, np.array(normal_angles))
            
            for k in range(6):
                nx, ny = normals[k]
                h_i = h_vals[k]
                
                # (px - cx)nx + (py - cy)ny + h_i <= D
                # -nx*cx - ny*cy - D <= -(px*nx + py*ny + h_i)
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny + h_i))
                
                # -(px - cx)nx - (py - cy)ny + h_i <= D
                # nx*cx + ny*cy - D <= px*nx + py*ny + h_i
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny + h_i)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                cx = res.x[0]
                cy = res.x[1]
                D = res.x[2]
                S = D / APOTHEM
                return cx, cy, S
            else:
                return None, None, float('inf')
        except Exception:
            return None, None, float('inf')

    def smooth_solve_lp(beta, inner_angles, temperature=1.0):
        """
        Smooth approximation of the LP using log-sum-exp.
        Returns (cx, cy, S_smooth) where S is an estimate of the optimal value.
        """
        # Normals of the outer hexagon
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # Variables: cx, cy, D
        # Use smooth max approximation
        # D_smooth = (1/temp) * log(sum(exp(temp * f_j)))
        # where f_j are the constraint values
        
        def objective(x):
            cx, cy, D = x
            # Compute all constraint violations
            violations = []
            for i, (px, py) in enumerate(pts):
                alpha_i = inner_angles[i]
                for nx, ny in normals:
                    dir_ang = math.atan2(ny, nx)
                    h_i = hex_support(alpha_i, dir_ang)
                    
                    # (px - cx)nx + (py - cy)ny + h_i - D
                    v1 = (px - cx) * nx + (py - cy) * ny + h_i - D
                    # -(px - cx)nx - (py - cy)ny + h_i - D
                    v2 = -(px - cx) * nx - (py - cy) * ny + h_i - D
                    
                    violations.append(v1)
                    violations.append(v2)
            
            # Smooth max of violations
            violations = np.array(violations)
            # Clip to prevent overflow
            violations = np.clip(violations, -50, 50)
            max_v = np.max(violations)
            if max_v > 50:
                smooth_max = max_v
            else:
                smooth_max = max_v + (1.0 / temperature) * math.log(np.sum(np.exp(temperature * (violations - max_v))))
            
            return D + smooth_max
        
        # Initial guess
        cx0, cy0, D0 = 0.0, 0.0, 3.0
        x0 = [cx0, cy0, D0]
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.success:
                cx, cy, D = res.x
                S = D / APOTHEM
                return cx, cy, S
            else:
                # Use the best point found
                cx, cy, D = res.x
                S = D / APOTHEM
                return cx, cy, S
        except Exception:
            return None, None, float('inf')

    # Start with all inner hexagons aligned with outer (angle = beta)
    # First, optimize beta with aligned inner hexagons
    def obj_aligned(beta):
        inner_angles = [beta] * n_pts
        _, _, S = solve_lp(beta, inner_angles)
        return S

    # Initial guess: beta = 0
    best_beta = 0.0
    best_cx, best_cy, best_S = solve_lp(best_beta, [best_beta] * n_pts)
    
    # If initial solve failed, try a small offset or return fallback
    if best_S == float('inf'):
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Minimize over beta in [0, pi/6] with multi-start
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            # Coarse grid search first
            for beta_init in np.linspace(0, math.pi / 6, 7):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                res = minimize_scalar(obj_aligned, bounds=(max(0, beta_init - math.pi/12), 
                                                         min(math.pi/6, beta_init + math.pi/12)), 
                                     method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-8})
                if res.fun < best_S:
                    best_beta = res.x
                    best_cx, best_cy, best_S = solve_lp(best_beta, [best_beta] * n_pts)
            
            # Fine refinement around best
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                local_bounds = (max(0, best_beta - 0.005), min(math.pi / 6, best_beta + 0.005))
                res2 = minimize_scalar(obj_aligned, bounds=local_bounds, method='bounded',
                                      options={'maxiter': 500, 'xatol': 1e-12})
                if res2.fun < best_S:
                    best_beta = res2.x
                    best_cx, best_cy, best_S = solve_lp(best_beta, [best_beta] * n_pts)
    except Exception:
        pass

    # Now optimize inner hexagon rotations independently
    # For each inner hex, try rotating it to minimize the required S
    # We'll do a local search on each inner hex angle
    
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        # Optimize each inner hex angle one at a time (coordinate descent)
        inner_angles = [best_beta] * n_pts
        
        # Multiple passes of coordinate descent
        for pass_num in range(8):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            improved = False
            for i in range(n_pts):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                
                def obj_angle_i(theta_i):
                    angles = inner_angles[:]
                    angles[i] = theta_i
                    _, _, S = solve_lp(best_beta, angles)
                    return S
                
                # Try a few candidate angles around current best
                current_angle = inner_angles[i]
                best_angle_i = current_angle
                best_S_i = obj_angle_i(current_angle)
                
                # Adaptive search: start wide, narrow down
                if pass_num < 2:
                    # Coarse: pi/6 range with 13 points
                    candidates = np.linspace(current_angle - math.pi/6, current_angle + math.pi/6, 13)
                elif pass_num < 4:
                    # Medium: pi/12 range with 17 points
                    candidates = np.linspace(current_angle - math.pi/12, current_angle + math.pi/12, 17)
                else:
                    # Fine: pi/24 range with 25 points
                    candidates = np.linspace(current_angle - math.pi/24, current_angle + math.pi/24, 25)
                
                # Also try canonical angles
                for base in [0, math.pi/6, math.pi/3, math.pi/2, 2*math.pi/3, 5*math.pi/6]:
                    candidates = np.append(candidates, base)
                
                for cand in candidates:
                    S_cand = obj_angle_i(cand)
                    if S_cand < best_S_i:
                        best_S_i = S_cand
                        best_angle_i = cand
                
                if best_S_i < best_S - 1e-14:
                    inner_angles[i] = best_angle_i
                    best_S = best_S_i
                    best_cx, best_cy, _ = solve_lp(best_beta, inner_angles)
                    improved = True
            
            # If no improvement in a full pass, stop early
            if not improved and pass_num >= 2:
                break
        
        # Final polish: optimize beta again with optimized inner angles
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            def obj_beta_final(beta):
                _, _, S = solve_lp(beta, inner_angles)
                return S
            
            # Multi-start for beta refinement
            for beta_init in [best_beta, 0, math.pi/12, math.pi/6]:
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                res = minimize_scalar(obj_beta_final, 
                                     bounds=(max(0, beta_init - 0.01), min(math.pi/6, beta_init + 0.01)),
                                     method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-11})
                if res.fun < best_S:
                    best_beta = res.x
                    best_cx, best_cy, best_S = solve_lp(best_beta, inner_angles)
            
            # Polish inner angles again with tighter search
            for i in range(n_pts):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                def obj_angle_i(theta_i):
                    angles = inner_angles[:]
                    angles[i] = theta_i
                    _, _, S = solve_lp(best_beta, angles)
                    return S
                
                current_angle = inner_angles[i]
                best_angle_i = current_angle
                best_S_i = obj_angle_i(current_angle)
                
                # Very fine search
                candidates = np.linspace(current_angle - math.pi/48, current_angle + math.pi/48, 33)
                for cand in candidates:
                    S_cand = obj_angle_i(cand)
                    if S_cand < best_S_i:
                        best_S_i = S_cand
                        best_angle_i = cand
                
                if best_S_i < best_S - 1e-14:
                    inner_angles[i] = best_angle_i
                    best_S = best_S_i
                    best_cx, best_cy, _ = solve_lp(best_beta, inner_angles)

    # Final check
    if best_S == float('inf'):
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # High-precision polish: jointly optimize (cx, cy, beta, inner_angles) 
    # using a smooth surrogate with log-sum-exp for the max constraint
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        n_vars = 2 + 1 + n_pts  # cx, cy, beta, inner_angles[0..n_pts-1]
        
        def smooth_objective(x):
            cx, cy = x[0], x[1]
            beta = x[2]
            angles = x[3:]
            
            # Normals of outer hexagon
            normal_angles = np.array([beta + math.pi / 6.0 + k * math.pi / 3.0 for k in range(6)])
            normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
            
            # Compute all constraint values vectorized
            constraints = []
            for i in range(n_pts):
                px, py = pts[i]
                alpha_i = angles[i]
                h_vals = hex_support_vec(alpha_i, normal_angles)
                
                for k in range(6):
                    nx, ny = normals[k]
                    h_i = h_vals[k]
                    v1 = (px - cx) * nx + (py - cy) * ny + h_i
                    v2 = -(px - cx) * nx - (py - cy) * ny + h_i
                    constraints.append(v1)
                    constraints.append(v2)
            
            constraints = np.array(constraints)
            max_c = np.max(constraints)
            
            # Smooth max via log-sum-exp with adaptive temperature
            temp = 100.0  # higher for sharper approximation
            shifted = constraints - max_c
            shifted = np.clip(shifted, -30, 30)
            smooth_max = max_c + (1.0 / temp) * math.log(np.sum(np.exp(temp * shifted)))
            
            S = smooth_max / APOTHEM
            return S
        
        # Initial point
        x0 = np.array([best_cx, best_cy, best_beta] + list(inner_angles))
        
        try:
            # First pass: Nelder-Mead with moderate settings
            res = minimize(smooth_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
            
            if res.fun < best_S:
                best_cx, best_cy = float(res.x[0]), float(res.x[1])
                best_beta = float(res.x[2])
                inner_angles = [float(a) for a in res.x[3:]]
                # Verify with exact LP
                _, _, S_exact = solve_lp(best_beta, inner_angles)
                if S_exact < best_S:
                    best_S = S_exact
                    best_cx, best_cy, _ = solve_lp(best_beta, inner_angles)
            
            # Second pass: finer Nelder-Mead from improved point
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                x1 = np.array([best_cx, best_cy, best_beta] + list(inner_angles))
                res2 = minimize(smooth_objective, x1, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True})
                
                if res2.fun < best_S:
                    best_cx, best_cy = float(res2.x[0]), float(res2.x[1])
                    best_beta = float(res2.x[2])
                    inner_angles = [float(a) for a in res2.x[3:]]
                    # Verify with exact LP
                    _, _, S_exact = solve_lp(best_beta, inner_angles)
                    if S_exact < best_S:
                        best_S = S_exact
                        best_cx, best_cy, _ = solve_lp(best_beta, inner_angles)
        except Exception:
            pass
    
    # Final exact LP verification and center refinement
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        _, _, S_final = solve_lp(best_beta, inner_angles)
        if S_final < best_S:
            best_S = S_final
            best_cx, best_cy, _ = solve_lp(best_beta, inner_angles)

    # Prepare output
    inner = []
    for i, (x, y) in enumerate(pts):
        angle_deg = math.degrees(inner_angles[i]) % 360.0
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(best_cx), float(best_cy)]
    side = float(best_S)
    rotation = math.degrees(best_beta) % 360.0
    
    return inner, center, side, float(rotation)
