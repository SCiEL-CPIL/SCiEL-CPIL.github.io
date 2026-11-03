import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top)
    # Vertices at angles 0, 60, ..., 300 degrees
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
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def check_disjoint_fast(poly1, poly2):
        """
        Fast check for disjointness using SAT.
        Returns True if disjoint (or touching), False if overlapping.
        """
        axes = []
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            axes.append(normal1)
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            axes.append(normal2)
            
        for axis in axes:
            norm = np.linalg.norm(axis)
            if norm < 1e-9: continue
            axis = axis / norm
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 or max2 < min1:
                return True
                
        return False

    def is_valid(positions, angles):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if not check_disjoint_fast(polys[i], polys[j]):
                    return False
        return True

    def compute_overlap_penalty_and_grad(params):
        """
        Compute a smooth penalty for overlaps and its gradient.
        Uses the minimum distance between vertices/edges as a proxy for separation.
        To keep it simple and robust, we use the sum of squared distances of closest points.
        However, SAT is not differentiable. We use a heuristic:
        For each pair, find the minimum distance between any vertex of i and any vertex of j.
        If dist < 0 (overlap), penalty = (dist)^2.
        Actually, vertices don't capture edge-edge overlap well.
        
        Alternative: Use the "soft" SAT penalty.
        For each axis, if projections overlap, penalty += overlap_depth^2.
        This is differentiable almost everywhere.
        """
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        
        penalty = 0.0
        grad_pos = np.zeros((N, 2))
        grad_ang = np.zeros(N)
        
        for i in range(N):
            for j in range(i+1, N):
                p1 = polys[i]
                p2 = polys[j]
                
                # SAT axes
                axes = []
                for k in range(6):
                    e1 = p1[(k+1)%6] - p1[k]
                    n1 = np.array([-e1[1], e1[0]])
                    axes.append(n1)
                    e2 = p2[(k+1)%6] - p2[k]
                    n2 = np.array([-e2[1], e2[0]])
                    axes.append(n2)
                    
                for axis in axes:
                    norm = np.linalg.norm(axis)
                    if norm < 1e-9: continue
                    axis = axis / norm
                    
                    proj1 = p1 @ axis
                    proj2 = p2 @ axis
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    
                    # Overlap depth
                    overlap = min(max1, max2) - max(min1, min2)
                    if overlap > 0:
                        # Penalty is overlap^2
                        # Gradient w.r.t position:
                        # d(overlap)/d(pos) depends on which vertex is max/min
                        # For max1: vertex v1_max. d(proj)/d(pos) = axis.
                        # So d(overlap)/d(pos_i) = axis * sign(max1 is limiting)
                        # This is complex. Let's use a simpler heuristic for gradient:
                        # Push centers apart along the axis if overlapping.
                        
                        # Determine if p1 is to the left or right of p2
                        center1 = np.mean(p1, axis=0)
                        center2 = np.mean(p2, axis=0)
                        vec = center2 - center1
                        proj_vec = np.dot(vec, axis)
                        
                        if proj_vec > 0:
                            # p2 is in direction of axis from p1
                            # Increase pos2, decrease pos1 to separate
                            # d(overlap)/d(pos2) ~ -axis (moving pos2 along axis reduces overlap)
                            # Wait, if p2 moves along +axis, min2 increases, max2 increases.
                            # Overlap = min(max1, max2) - max(min1, min2).
                            # This derivative is piecewise constant.
                            # Approximation: grad_pos[i] += axis * overlap * 2.0
                            # grad_pos[j] -= axis * overlap * 2.0
                            pass
                        
                        # Simple heuristic: push centers apart
                        dist_centers = np.linalg.norm(center2 - center1)
                        if dist_centers > 1e-6:
                            dir_vec = (center2 - center1) / dist_centers
                            # Gradient of penalty (overlap^2) w.r.t separation is ~ overlap
                            # We want to increase separation.
                            # So gradient of penalty w.r.t pos_i is -dir_vec * overlap * 2
                            # Gradient w.r.t pos_j is +dir_vec * overlap * 2
                            # But this ignores the specific axis.
                            # Let's just accumulate the axis-based push.
                            # If p2 is "ahead" on axis, push p2 forward, p1 backward.
                            if proj_vec > 0:
                                grad_pos[i] += axis * (2.0 * overlap)
                                grad_pos[j] -= axis * (2.0 * overlap)
                            else:
                                grad_pos[i] -= axis * (2.0 * overlap)
                                grad_pos[j] += axis * (2.0 * overlap)
                                
                        penalty += overlap * overlap
                        
        return penalty, grad_pos, grad_ang

    def objective_with_grad(params):
        """
        Objective function for L-BFGS-B.
        Returns (value, gradient)
        """
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        side = calc_min_outer_side(positions, angles, 0.0)
        
        # Get overlap penalty and gradient
        pen, grad_pen_pos, grad_pen_ang = compute_overlap_penalty_and_grad(params)
        
        # Total objective
        # We want to minimize side + lambda * pen
        # Lambda increases if pen is high
        lambda_val = 1.0 + 10.0 * (pen > 0.01)
        
        value = side + lambda_val * pen
        
        # Gradient
        # Approximate gradient of side
        # side = max_proj * 2 / sqrt(3)
        # max_proj = max over all vertices v of dot(v, normal_k)
        # This is also piecewise linear.
        # Approximation: Use finite difference or ignore gradient for side initially?
        # No, L-BFGS-B needs gradient.
        # Let's use a simple finite difference for the side part? Too slow.
        # Let's use a heuristic gradient for side:
        # If a vertex is the max projector, moving that hexagon in direction of normal increases side.
        # So grad_side_pos[i] = normal_k * (2/sqrt(3)) if hex i provides the max.
        
        theta_rad = 0.0
        normals = np.array([[math.cos(theta_rad + math.pi/6 + k * math.pi/3), 
                             math.sin(theta_rad + math.pi/6 + k * math.pi/3)] for k in range(6)])
        
        max_proj = 0.0
        max_idx = -1
        max_vert_idx = -1
        max_normal_idx = -1
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            local_max = np.max(projs)
            if local_max > max_proj:
                max_proj = local_max
                max_idx = i
                max_vert_idx = np.argmax(projs)
                max_normal_idx = np.argmax(projs[max_vert_idx]) # This is wrong, projs is (6,6)
                
        # Recalculate max normal index properly
        if max_idx >= 0:
            cx, cy = positions[max_idx]
            ang = angles[max_idx]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T # shape (6, 6)
            # Find global max
            global_max_val = np.max(projs)
            max_v_idx, max_n_idx = np.unravel_index(np.argmax(projs), projs.shape)
            
            # Gradient of side w.r.t position of hex max_idx
            # side = (max_proj * 2) / sqrt(3)
            # d(side)/d(pos) = (2/sqrt(3)) * d(max_proj)/d(pos)
            # max_proj = dot(vert, normal)
            # vert = R * v0 + pos
            # d(vert)/d(pos) = I
            # d(max_proj)/d(pos) = normal
            # So grad_side_pos[max_idx] = (2/sqrt(3)) * normals[max_n_idx]
            
            grad_side_pos = np.zeros((N, 2))
            grad_side_pos[max_idx] = (2.0 / math.sqrt(3)) * normals[max_n_idx]
            
            # Gradient w.r.t angle is more complex, approximate as 0 for now or small random
            grad_side_ang = np.zeros(N)
            
        else:
            grad_side_pos = np.zeros((N, 2))
            grad_side_ang = np.zeros(N)
            
        total_grad_pos = grad_side_pos + lambda_val * grad_pen_pos
        total_grad_ang = grad_side_ang + lambda_val * grad_pen_ang
        
        grad = np.concatenate([total_grad_pos.flatten(), total_grad_ang])
        
        return value, grad

    best_score = float('inf')
    best_params = None

    # Multi-start optimization
    # Start 1: Honeycomb subset
    sqrt3 = math.sqrt(3)
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
    
    x0_list = [np.concatenate([init_positions.flatten(), init_angles])]
    
    # Start 2: Random
    np.random.seed(42)
    for _ in range(5):
        pos = np.random.uniform(-5, 5, (N, 2))
        ang = np.random.uniform(0, 360, N)
        x0_list.append(np.concatenate([pos.flatten(), ang]))
        
    # Start 3: Two rings
    r1, r2 = 2.0, 4.0
    pos_rings = []
    ang_rings = []
    for k in range(6):
        theta = k * math.pi / 3
        pos_rings.append([r1 * math.cos(theta), r1 * math.sin(theta)])
        ang_rings.append(math.degrees(theta))
    for k in range(6):
        theta = k * math.pi / 3 + math.pi/6
        pos_rings.append([r2 * math.cos(theta), r2 * math.sin(theta)])
        ang_rings.append(math.degrees(theta))
    x0_list.append(np.concatenate([np.array(pos_rings).flatten(), np.array(ang_rings)]))

    bounds = [(-10, 10)] * (2*N) + [(0, 360)] * N

    for idx, x0 in enumerate(x0_list):
        if time.time() > deadline - 20:
            break
            
        try:
            res = minimize(
                objective_with_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8, 'maxls': 20}
            )
            
            # Check validity and score
            if res.fun < best_score:
                # Verify validity
                final_positions = res.x[:2*N].reshape(N, 2)
                final_angles = res.x[2*N:]
                if is_valid(final_positions, final_angles):
                    score = calc_min_outer_side(final_positions, final_angles, 0.0)
                    if score < best_score:
                        best_score = score
                        best_params = res.x
        except Exception:
            pass

    # If L-BFGS-B failed to find a valid tight packing, try Nelder-Mead on best found or fallback
    if best_params is None:
        # Fallback to a safe, valid configuration
        # Use the honeycomb subset
        final_positions = init_positions
        final_angles = init_angles
        # Verify
        if not is_valid(final_positions, final_angles):
            # Force separation
            # This is a rough fallback
            final_positions = np.array([
                [0, 0], [2.1, 0], [-2.1, 0],
                [1.05, 1.81], [-1.05, 1.81], [0, 3.62],
                [1.05, -1.81], [-1.05, -1.81], [0, -3.62],
                [3.15, 1.81], [-3.15, 1.81], [3.15, -1.81]
            ])
            final_angles = np.zeros(N)
        best_score = calc_min_outer_side(final_positions, final_angles, 0.0)
    else:
        final_positions = best_params[:2*N].reshape(N, 2)
        final_angles = best_params[2*N:]
        
        # Final polish with Nelder-Mead if time permits
        if time.time() < deadline - 30:
            try:
                def obj_nm(params):
                    positions = params[:2*N].reshape(N, 2)
                    angles = params[2*N:]
                    if not is_valid(positions, angles):
                        return 100.0
                    return calc_min_outer_side(positions, angles, 0.0)
                
                res_nm = minimize(
                    obj_nm,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                
                if res_nm.fun < best_score and is_valid(res_nm.x[:2*N].reshape(N,2), res_nm.x[2*N:]):
                    best_score = res_nm.fun
                    best_params = res_nm.x
                    final_positions = best_params[:2*N].reshape(N, 2)
                    final_angles = best_params[2*N:]
            except Exception:
                pass

    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0

if __name__ == "__main__":
    construct()
