import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, linprog, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def hex_support(alpha, direction_angle):
    """
    Compute the support function of a unit regular hexagon with rotation alpha
    in the given direction.
    For a regular hexagon of side 1, the support in direction theta is:
    h(theta) = APOTHEM * max over vertices of cos(theta - vertex_angle)
    But more efficiently, for a regular hexagon, the support is piecewise.
    Actually, for a regular hexagon with apothem a, the support in any direction
    is a * |cos(theta - alpha - k*pi/3)| for the appropriate k... 
    
    More precisely: the support function of a regular hexagon centered at origin,
    with one vertex at angle alpha, in direction theta is:
    h(theta) = APOTHEM / cos(min_k |theta - alpha - k*pi/3| mod pi/3)
    
    Wait, that's not right either. Let me think again.
    
    A regular hexagon of side 1 has apothem APOTHEM = sqrt(3)/2.
    The vertices are at distance 1 from center, at angles alpha + k*pi/3.
    The support in direction theta is max_i (v_i . d) where d = (cos theta, sin theta).
    = max_k cos(theta - alpha - k*pi/3)
    
    Since cos is periodic with period 2*pi and symmetric, we can reduce theta - alpha
    modulo pi/3 and take the max.
    """
    diff = (direction_angle - alpha) % (math.pi / 3.0)
    # The maximum of cos over the 6 vertices occurs when diff is closest to 0 or pi/3
    # Actually, cos(theta - alpha - k*pi/3) for k=0..5
    # The max is cos(min over k of |theta - alpha - k*pi/3|) but with periodicity
    # Let's just compute all 6
    max_val = -1.0
    for k in range(6):
        ang = direction_angle - alpha - k * math.pi / 3.0
        c = math.cos(ang)
        if c > max_val:
            max_val = c
    return max_val  # This is the support of a unit-circumradius hexagon
    # But our hexagon has side 1, so circumradius = 1, apothem = sqrt(3)/2
    # The support in direction theta is indeed max_k cos(theta - alpha - k*pi/3)
    # since vertices are at unit distance.

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
    
    def solve_lp(beta, inner_angles):
        """
        Solves the LP for a given outer rotation beta and inner hexagon rotations.
        Returns (cx, cy, S) or (None, None, inf)
        """
        # Check time limit
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return None, None, float('inf')
        
        # Normals of the outer hexagon
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
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
            for nx, ny in normals:
                # Support of inner hex i in direction (nx, ny)
                dir_ang = math.atan2(ny, nx)
                h_i = hex_support(alpha_i, dir_ang)
                
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

    # Minimize over beta in [0, pi/6]
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            res = minimize_scalar(obj_aligned, bounds=(0, math.pi / 6), method='bounded',
                                 options={'maxiter': 500, 'xatol': 1e-10})
            if res.fun < best_S:
                best_beta = res.x
                best_cx, best_cy, best_S = solve_lp(best_beta, [best_beta] * n_pts)
                
                # Second pass with even tighter tolerance around the found minimum
                if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                    local_bounds = (max(0, best_beta - 0.01), min(math.pi / 6, best_beta + 0.01))
                    res2 = minimize_scalar(obj_aligned, bounds=local_bounds, method='bounded',
                                          options={'maxiter': 1000, 'xatol': 1e-12})
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
        for pass_num in range(3):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
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
                
                # Try angles in [current - pi/6, current + pi/6] with some candidates
                candidates = []
                for delta in np.linspace(-math.pi/6, math.pi/6, 7):
                    candidates.append(current_angle + delta)
                # Also try 0, pi/6, pi/3, etc.
                for base in [0, math.pi/6, math.pi/3, math.pi/2, 2*math.pi/3, 5*math.pi/6]:
                    candidates.append(base)
                
                for cand in candidates:
                    S_cand = obj_angle_i(cand)
                    if S_cand < best_S_i:
                        best_S_i = S_cand
                        best_angle_i = cand
                
                if best_S_i < best_S - 1e-12:
                    inner_angles[i] = best_angle_i
                    best_S = best_S_i
                    best_cx, best_cy, _ = solve_lp(best_beta, inner_angles)
        
        # Final polish: optimize beta again with optimized inner angles
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            def obj_beta_final(beta):
                _, _, S = solve_lp(beta, inner_angles)
                return S
            
            res = minimize_scalar(obj_beta_final, bounds=(0, math.pi / 6), method='bounded',
                                 options={'maxiter': 500, 'xatol': 1e-10})
            if res.fun < best_S:
                best_beta = res.x
                best_cx, best_cy, best_S = solve_lp(best_beta, inner_angles)
                
                # Polish inner angles again
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
                    
                    candidates = np.linspace(current_angle - math.pi/12, current_angle + math.pi/12, 13)
                    for cand in candidates:
                        S_cand = obj_angle_i(cand)
                        if S_cand < best_S_i:
                            best_S_i = S_cand
                            best_angle_i = cand
                    
                    if best_S_i < best_S - 1e-12:
                        inner_angles[i] = best_angle_i
                        best_S = best_S_i
                        best_cx, best_cy, _ = solve_lp(best_beta, inner_angles)

    # Final check
    if best_S == float('inf'):
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Prepare output
    inner = []
    for i, (x, y) in enumerate(pts):
        angle_deg = math.degrees(inner_angles[i])
        inner.append([float(x), float(y), angle_deg])
        
    center = [float(best_cx), float(best_cy)]
    side = float(best_S)
    rotation = math.degrees(best_beta)
    
    return inner, center, side, rotation
