import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

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
        """
        theta_rad = math.radians(outer_angle)
        normal_angles = theta_rad + math.pi/6 + np.arange(6) * math.pi/3
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj_all = 0.0
        for i in range(N):
            cx, cy = positions[i]
            ang_deg = angles[i]
            ang_rad = math.radians(ang_deg)
            c, s = math.cos(ang_rad), math.sin(ang_rad)
            # Rotate unit hex vertices
            ux = unit_verts_0[:, 0]
            uy = unit_verts_0[:, 1]
            rx = c * ux - s * uy
            ry = s * ux + c * uy
            verts = np.column_stack([rx + cx, ry + cy])
            projs = verts @ normals.T  # (6, 6)
            max_proj_i = np.max(projs, axis=0)  # (6,)
            m = np.max(max_proj_i)
            if m > max_proj_all:
                max_proj_all = m
        
        # The apothem of the outer hexagon is max_proj_all
        # Side length = 2 * apothem / sqrt(3)
        return max_proj_all * 2.0 / math.sqrt(3)

    def sat_gap(poly1, poly2):
        """
        Returns the minimum separation distance between two convex hexagons.
        """
        axes1 = []
        for i in range(6):
            edge = poly1[(i+1)%6] - poly1[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                axes1.append(n / norm)
        
        axes2 = []
        for i in range(6):
            edge = poly2[(i+1)%6] - poly2[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                axes2.append(n / norm)
        
        min_gap = float('inf')
        for axis in axes1:
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            gap = max(np.min(proj1) - np.max(proj2), np.min(proj2) - np.max(proj1))
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
        
        for axis in axes2:
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            gap = max(np.min(proj1) - np.max(proj2), np.min(proj2) - np.max(proj1))
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

    # --- PHASE 1: INITIAL TOPOLOGY ---
    sqrt3 = math.sqrt(3)
    best_valid_config = None
    best_valid_score = float('inf')
    
    base_v1 = np.array([1.5, sqrt3/2])
    base_v2 = np.array([0, sqrt3])
    
    for scale in [0.95, 1.0, 1.05]:
        v1 = base_v1 * scale
        v2 = base_v2 * scale
        points = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                p = i * v1 + j * v2
                points.append(p)
        
        points.sort(key=lambda p: np.dot(p, p))
        selected = np.array(points[:12])
        centroid = np.mean(selected, axis=0)
        selected -= centroid
        
        for rot_deg in [0, 30, 60, 15, 45]:
            rot_rad = math.radians(rot_deg)
            rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)], 
                                [math.sin(rot_rad), math.cos(rot_rad)]])
            rotated_selected = selected @ rot_mat.T
            angles = np.full(N, rot_deg)
            
            if is_valid(rotated_selected, angles, tol=1e-6):
                for oa in [0, 5, 10, 15, 20, 25, 30]:
                    score = calc_min_outer_side(rotated_selected, angles, oa)
                    if score < best_valid_score:
                        best_valid_score = score
                        best_valid_config = (rotated_selected.copy(), angles.copy(), oa)
            
            if time.time() > deadline - 60:
                break
        if time.time() > deadline - 60:
            break

    if best_valid_config is None:
        lattice_points = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * 1.5
                y = i * sqrt3 / 2.0 + j * sqrt3
                lattice_points.append((x, y))
        lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
        init_positions = np.array(lattice_points[:12])
        init_positions -= np.mean(init_positions, axis=0)
        init_angles = np.zeros(N)
        best_valid_config = (init_positions, init_angles, 0.0)
        best_valid_score = calc_min_outer_side(init_positions, init_angles, 0.0)

    init_positions, init_angles, init_outer_angle = best_valid_config
    x0 = np.concatenate([init_positions.flatten(), init_angles, [init_outer_angle]])
    
    best_score = best_valid_score
    best_result = x0.copy()

    # --- PHASE 2: HOMOTOPY CONTINUATION ---
    # We track the solution branch by solving a sequence of problems where the
    # objective is to minimize the outer side length, subject to a constraint
    # that the outer side length must be at least the previous step's value.
    # This effectively forces the optimizer to explore the "boundary" of the
    # feasible region, pushing it towards the minimum.
    
    current_side = best_valid_score
    num_steps = 25
    step_size = 0.05 # 5% reduction per step
    
    def make_side_constraint(side_limit):
        def constraint(params):
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:3*N]
            outer_angle = params[3*N]
            return side_limit - calc_min_outer_side(positions, angles, outer_angle)
        return constraint

    def make_collision_constraint(i, j):
        def constraint(params):
            pos_i = params[2*i:2*i+2]
            pos_j = params[2*j:2*j+2]
            ang_i = params[2*N + i]
            ang_j = params[2*N + j]
            
            poly_i = get_hex_verts(pos_i[0], pos_i[1], 1.0, ang_i)
            poly_j = get_hex_verts(pos_j[0], pos_j[1], 1.0, ang_j)
            
            return sat_gap(poly_i, poly_j)
        return constraint

    collision_constraints = []
    for i in range(N):
        for j in range(i+1, N):
            collision_constraints.append({'type': 'ineq', 'fun': make_collision_constraint(i, j)})
            
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
    for i in range(N):
        bounds.append((0, 360))
    bounds.append((0, 60))

    for step in range(num_steps):
        if time.time() > deadline - 20:
            break
            
        target_side = current_side * (1.0 - step_size)
        if target_side < 3.0: # Safety floor
            target_side = 3.0
            
        side_constraint = {'type': 'ineq', 'fun': make_side_constraint(target_side)}
        constraints = collision_constraints + [side_constraint]
        
        def objective(params):
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:3*N]
            outer_angle = params[3*N]
            return calc_min_outer_side(positions, angles, outer_angle)
        
        try:
            res = minimize(
                objective, 
                best_result, 
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
            )
            
            if res.success or res.fun < current_side:
                pos = res.x[:2*N].reshape(N, 2)
                ang = res.x[2*N:3*N]
                outer_ang = res.x[3*N]
                
                if is_valid(pos, ang, tol=1e-4):
                    new_side = calc_min_outer_side(pos, ang, outer_ang)
                    if new_side < best_score:
                        best_score = new_side
                        best_result = res.x.copy()
                    # Update current side for next step
                    # We use the actual achieved side, not the target, to stay on the branch
                    current_side = new_side
                else:
                    # If invalid, we failed to track the branch, break
                    break
            else:
                # If optimization failed to find a feasible point for this target, 
                # we might be stuck or the target is too aggressive.
                # We can try a smaller step or break.
                if step > 5:
                    break
        except Exception:
            break

    # --- PHASE 3: FINAL LOCAL POLISH ---
    # After homotopy, we might be at a local minimum. Do a final unconstrained 
    # (except collisions) polish to squeeze out any remaining slack.
    if time.time() < deadline - 5:
        try:
            # Remove the side constraint, just minimize side subject to collisions
            res_final = minimize(
                objective,
                best_result,
                method='SLSQP',
                bounds=bounds,
                constraints=collision_constraints,
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            
            pos = res_final.x[:2*N].reshape(N, 2)
            ang = res_final.x[2*N:3*N]
            outer_ang = res_final.x[3*N]
            
            if is_valid(pos, ang, tol=1e-5):
                score_final = calc_min_outer_side(pos, ang, outer_ang)
                if score_final < best_score:
                    best_score = score_final
                    best_result = res_final.x.copy()
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
