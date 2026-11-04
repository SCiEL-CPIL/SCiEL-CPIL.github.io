import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute unit hexagon vertices for angle 0 (flat-top)
    # Vertices at 0, 60, ..., 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    
    # Precompute normals for the 6 sides of a unit hexagon
    unit_normals_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(30, 360, 60)])

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
        """
        theta_rad = math.radians(outer_angle)
        # Normals for the outer hexagon sides: 6 directions
        normal_angles = theta_rad + math.pi/6 + np.arange(6) * math.pi/3
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj_all = 0.0
        for i in range(N):
            cx, cy = positions[i]
            ang_deg = angles[i]
            ang_rad = math.radians(ang_deg)
            c, s = math.cos(ang_rad), math.sin(ang_rad)
            rot = np.array([[c, -s], [s, c]])
            verts = unit_verts_0 @ rot.T + np.array([cx, cy])
            projs = verts @ normals.T  # (6, 6)
            max_proj_i = np.max(projs, axis=0)  # (6,)
            m = np.max(max_proj_i)
            if m > max_proj_all:
                max_proj_all = m
        
        return max_proj_all * 2.0 / math.sqrt(3)

    def sat_gap(poly1, poly2):
        """
        Returns the minimum separation distance between two convex polygons.
        If they overlap, returns a negative value (approximate penetration depth).
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

    def is_valid(positions, angles, tol=1e-5):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    # --- PHASE 1: CONSTRUCT KNOWN GOOD TOPOLOGY (3x4 BRICK WALL) ---
    # This configuration is known to be valid and yields a side length of ~5.
    sqrt3 = math.sqrt(3)
    
    init_positions = []
    init_angles = []
    
    # 3 columns, 4 rows
    # x = 2*i + (j % 2)
    # y = j * sqrt(3)
    for j in range(4):
        for i in range(3):
            x = 2.0 * i + (j % 2)
            y = j * sqrt3
            init_positions.append([x, y])
            init_angles.append(0.0)
            
    # Center the cluster at origin
    init_positions = np.array(init_positions)
    centroid = np.mean(init_positions, axis=0)
    init_positions -= centroid
    init_angles = np.array(init_angles)
    
    # Check validity and score
    if is_valid(init_positions, init_angles, tol=1e-6):
        best_valid_score = calc_min_outer_side(init_positions, init_angles, 0.0)
        best_valid_config = (init_positions.copy(), init_angles.copy())
    else:
        # Fallback if brick wall fails (should not happen)
        print("Brick wall invalid, using random")
        init_positions = np.random.rand(N, 2) * 10 - 5
        init_angles = np.zeros(N)
        best_valid_score = float('inf')
        best_valid_config = (init_positions, init_angles)

    init_positions, init_angles = best_valid_config
    x0 = np.concatenate([init_positions.flatten(), init_angles, [0.0]])
    
    # --- PHASE 2: SLSQP REFINEMENT ---
    
    def objective(params):
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N]
        outer_angle = params[3*N]
        return calc_min_outer_side(positions, angles, outer_angle)
    
    def make_constraint(i, j):
        def constraint(params):
            pos_i = params[2*i:2*i+2]
            pos_j = params[2*j:2*j+2]
            ang_i = params[2*N + i]
            ang_j = params[2*N + j]
            
            poly_i = get_hex_verts(pos_i[0], pos_i[1], 1.0, ang_i)
            poly_j = get_hex_verts(pos_j[0], pos_j[1], 1.0, ang_j)
            
            return sat_gap(poly_i, poly_j)
        return constraint

    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
    
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
    for i in range(N):
        bounds.append((0, 360))
    bounds.append((0, 60))  # outer angle

    best_score = best_valid_score
    best_result = x0.copy()

    # Run SLSQP
    try:
        res = minimize(
            objective, 
            x0, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 400, 'ftol': 1e-10, 'disp': False}
        )
        
        if res.success or res.fun < best_score:
            pos = res.x[:2*N].reshape(N, 2)
            ang = res.x[2*N:3*N]
            outer_ang = res.x[3*N]
            
            # Verify validity strictly
            if is_valid(pos, ang, tol=1e-4):
                score = calc_min_outer_side(pos, ang, outer_ang)
                if score < best_score:
                    best_score = score
                    best_result = res.x.copy()
    except Exception as e:
        print(f"SLSQP error: {e}")

    # --- PHASE 3: LOCAL POLISHING ---
    if time.time() < deadline - 10:
        def soft_min_penalty(polys):
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    axes = []
                    for k in range(6):
                        e1 = polys[i][(k+1)%6] - polys[i][k]
                        n1 = np.array([-e1[1], e1[0]])
                        nm = np.linalg.norm(n1)
                        if nm > 1e-9:
                            axes.append(n1 / nm)
                    for k in range(6):
                        e2 = polys[j][(k+1)%6] - polys[j][k]
                        n2 = np.array([-e2[1], e2[0]])
                        nm = np.linalg.norm(n2)
                        if nm > 1e-9:
                            axes.append(n2 / nm)
                    
                    gaps = []
                    for ax in axes:
                        p1 = polys[i] @ ax
                        p2 = polys[j] @ ax
                        gap = max(np.min(p1) - np.max(p2), np.min(p2) - np.max(p1))
                        gaps.append(gap)
                    gaps = np.array(gaps)
                    min_gap = np.min(gaps)
                    if min_gap < 0:
                        penalty += 500.0 * min_gap * min_gap
            return penalty

        def obj_penalty(params):
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:3*N]
            outer_angle = params[3*N]
            side_len = calc_min_outer_side(positions, angles, outer_angle)
            polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
            penalty = soft_min_penalty(polys)
            return side_len + penalty

        try:
            res2 = minimize(
                obj_penalty, 
                best_result, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True, 'disp': False}
            )
            
            pos = res2.x[:2*N].reshape(N, 2)
            ang = res2.x[2*N:3*N]
            outer_ang = res2.x[3*N]
            
            if is_valid(pos, ang, tol=1e-5):
                score2 = calc_min_outer_side(pos, ang, outer_ang)
                if score2 < best_score:
                    best_score = score2
                    best_result = res2.x.copy()
        except Exception:
            pass

    # Finalize
    final_positions = best_result[:2*N].reshape(N, 2)
    final_angles = best_result[2*N:3*N]
    final_outer_angle = best_result[3*N]
    final_side = calc_min_outer_side(final_positions, final_angles, final_outer_angle)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, final_outer_angle

if __name__ == "__main__":
    construct()
