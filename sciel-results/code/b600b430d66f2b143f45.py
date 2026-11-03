import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)
TIME_LIMIT = 145.0  # Safety margin before 150s kill

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute unit hexagon vertices for angle 0 (flat-top)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0)
        with rotation outer_angle that contains all inner hexagons.
        Uses soft-max for smoothness.
        """
        theta_rad = math.radians(outer_angle)
        normals = np.array([[math.cos(theta_rad + math.pi/6 + k * math.pi/3),
                             math.sin(theta_rad + math.pi/6 + k * math.pi/3)]
                            for k in range(6)])
        
        # Collect all vertex projections
        all_projs = []
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            all_projs.append(projs)
        
        all_projs = np.concatenate(all_projs, axis=0)
        
        # Use log-sum-exp for smooth max
        max_val = np.max(all_projs)
        if max_val > 0:
            log_sum_exp = max_val + np.log(np.sum(np.exp(all_projs - max_val)))
        else:
            log_sum_exp = 0.0
        
        # Convert to side length: for a regular hexagon, apothem = side * sqrt(3)/2
        # So side = 2 * apothem / sqrt(3)
        return log_sum_exp * 2.0 / math.sqrt(3)

    def sat_gap(poly1, poly2):
        """
        Returns the minimum separation distance between two convex polygons.
        If they overlap, returns a negative value (penetration depth approximation).
        If they are separated, returns the positive distance.
        """
        min_gap = float('inf')
        
        # Axes from poly1 edges
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            norm = np.linalg.norm(normal1)
            if norm < 1e-9: continue
            axis = normal1 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Gap on this axis
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                # Overlapping on this axis
                min_gap = min(min_gap, gap)
            else:
                # Separated on this axis -> disjoint
                return gap
                
        # Axes from poly2 edges
        for i in range(6):
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            norm = np.linalg.norm(normal2)
            if norm < 1e-9: continue
            axis = normal2 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
                
        return min_gap

    def objective(params, penalty_weight=1000.0):
        if time.time() > deadline - 5:
            return 1000.0
            
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        side_len = calc_min_outer_side(positions, angles, 0.0)
        
        penalty = 0.0
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    # Use exponential penalty for stronger separation
                    penalty += penalty_weight * (1.0 - math.exp(gap * 10.0))
                    
        return side_len + penalty

    def check_validity(positions, angles, tol=1e-5):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    # Initial guess: Honeycomb subset - use a better symmetric configuration
    # Place hexagons in a ring pattern around center
    init_positions = np.zeros((N, 2))
    init_angles = np.zeros(N)
    
    # Center hexagon
    init_positions[0] = [0, 0]
    
    # Ring of 6 hexagons at distance sqrt(3) (touching)
    for i in range(1, 7):
        angle = (i-1) * 60 * math.pi / 180
        init_positions[i] = [SQRT3 * math.cos(angle), SQRT3 * math.sin(angle)]
    
    # Outer ring of 5 hexagons
    for i in range(7, 12):
        angle = (i-7) * 72 * math.pi / 180 + 30 * math.pi / 180
        dist = SQRT3 * 2.0
        init_positions[i] = [dist * math.cos(angle), dist * math.sin(angle)]
    
    x0 = np.concatenate([init_positions.flatten(), init_angles])
    
    bounds = []
    for i in range(N):
        bounds.append((-10, 10)) # x
        bounds.append((-10, 10)) # y
    for i in range(N):
        bounds.append((0, 360)) # angle

    best_score = float('inf')
    best_result = None

    # Strategy 1: Penalty Method with L-BFGS-B
    try:
        res = minimize(
            objective, 
            x0, 
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-8, 'maxls': 30, 'disp': False}
        )
        
        pos = res.x[:2*N].reshape(N, 2)
        ang = res.x[2*N:]
        
        if check_validity(pos, ang):
            best_score = calc_min_outer_side(pos, ang, 0.0)
            best_result = res.x
    except Exception as e:
        pass

    # Strategy 2: Nelder-Mead if LBFGS failed or score is poor
    if (best_result is None or best_score > 4.5) and time.time() < deadline - 60:
        try:
            def obj_nm(params):
                if time.time() > deadline - 5:
                    return 1000.0
                positions = params[:2*N].reshape(N, 2)
                angles = params[2*N:]
                side_len = calc_min_outer_side(positions, angles, 0.0)
                penalty = 0.0
                polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
                for i in range(N):
                    for j in range(i+1, N):
                        gap = sat_gap(polys[i], polys[j])
                        if gap < 0:
                            penalty += 500.0 * (1.0 - math.exp(gap * 10.0))
                return side_len + penalty

            start_x = best_result if best_result is not None else x0
            
            res2 = minimize(
                obj_nm, 
                start_x, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True, 'disp': False}
            )
            
            pos = res2.x[:2*N].reshape(N, 2)
            ang = res2.x[2*N:]
            
            if check_validity(pos, ang):
                score2 = calc_min_outer_side(pos, ang, 0.0)
                if score2 < best_score:
                    best_score = score2
                    best_result = res2.x
        except Exception:
            pass

    # Strategy 3: Gradient-based local search with stronger penalties
    if best_result is not None and time.time() < deadline - 40:
        try:
            def obj_strong(params):
                if time.time() > deadline - 5:
                    return 1000.0
                positions = params[:2*N].reshape(N, 2)
                angles = params[2*N:]
                side_len = calc_min_outer_side(positions, angles, 0.0)
                penalty = 0.0
                polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
                for i in range(N):
                    for j in range(i+1, N):
                        gap = sat_gap(polys[i], polys[j])
                        if gap < 0:
                            penalty += 10000.0 * (1.0 - math.exp(gap * 20.0))
                return side_len + penalty

            res3 = minimize(
                obj_strong,
                best_result,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'maxls': 30, 'disp': False}
            )
            
            pos = res3.x[:2*N].reshape(N, 2)
            ang = res3.x[2*N:]
            
            if check_validity(pos, ang):
                score3 = calc_min_outer_side(pos, ang, 0.0)
                if score3 < best_score:
                    best_score = score3
                    best_result = res3.x
        except Exception:
            pass

    # Strategy 4: Final Polish with very tight tolerances
    if best_result is not None and time.time() < deadline - 10:
        try:
            def obj_final(params):
                if time.time() > deadline - 2:
                    return 1000.0
                positions = params[:2*N].reshape(N, 2)
                angles = params[2*N:]
                side_len = calc_min_outer_side(positions, angles, 0.0)
                penalty = 0.0
                polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
                for i in range(N):
                    for j in range(i+1, N):
                        gap = sat_gap(polys[i], polys[j])
                        if gap < 0:
                            penalty += 50000.0 * (1.0 - math.exp(gap * 50.0))
                return side_len + penalty

            res4 = minimize(
                obj_final, 
                best_result, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14, 'maxls': 20, 'disp': False}
            )
            
            pos = res4.x[:2*N].reshape(N, 2)
            ang = res4.x[2*N:]
            
            if check_validity(pos, ang, tol=1e-6):
                score4 = calc_min_outer_side(pos, ang, 0.0)
                if score4 < best_score:
                    best_score = score4
                    best_result = res4.x
        except Exception:
            pass

    # Finalize
    if best_result is not None:
        final_positions = best_result[:2*N].reshape(N, 2)
        final_angles = best_result[2*N:]
        final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    else:
        # Fallback
        final_positions = np.array([
            [0, 0], [2.1, 0], [-2.1, 0],
            [1.05, 1.81], [-1.05, 1.81], [0, 3.62],
            [1.05, -1.81], [-1.05, -1.81], [0, -3.62],
            [3.15, 1.81], [-3.15, 1.81], [3.15, -1.81]
        ])
        final_angles = np.zeros(N)
        final_side = calc_min_outer_side(final_positions, final_angles, 0.0)

    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0
