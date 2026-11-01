import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def polygons_disjoint_sat(p1, p2, eps=1e-5):
        """
        Returns True if two convex polygons are disjoint (allowing touching with eps tolerance).
        Uses Separating Axis Theorem (SAT).
        """
        for i in range(len(p1)):
            j = (i + 1) % len(p1)
            edge = p1[j] - p1[i]
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) < 1e-10:
                continue
            normal /= np.linalg.norm(normal)
            
            proj1 = p1.dot(normal)
            proj2 = p2.dot(normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - eps or max2 < min1 - eps:
                return True
        
        for i in range(len(p2)):
            j = (i + 1) % len(p2)
            edge = p2[j] - p2[i]
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) < 1e-10:
                continue
            normal /= np.linalg.norm(normal)
            
            proj1 = p1.dot(normal)
            proj2 = p2.dot(normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - eps or max2 < min1 - eps:
                return True
        
        return False

    def check_validity(positions, angles_deg):
        """
        positions: (12, 2) array of (x, y) for unit hexagons.
        angles_deg: (12,) array of rotation angles in degrees.
        Returns True if all are pairwise disjoint.
        """
        vertices_list = []
        for i in range(N):
            x, y = positions[i]
            ang = angles_deg[i]
            verts = hexagon_vertices(x, y, 1.0, ang)
            vertices_list.append(verts)
        
        for i in range(N):
            for j in range(i + 1, N):
                if not polygons_disjoint_sat(vertices_list[i], vertices_list[j]):
                    return False
        return True

    def objective_smooth(params):
        """
        Smooth objective function.
        params: (12, 3) flattened -> (36,). Each row is (x, y, angle_rad).
        We optimize positions and angles.
        """
        if time.time() > deadline:
            return 1e10
            
        params = params.reshape((N, 3))
        positions = params[:, :2]
        angles = params[:, 2]
        
        # Calculate vertices for all hexagons
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, np.degrees(angles[i]))
            
        verts_flat = verts.reshape(-1, 2)
        
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        L1 = x + y / sqrt3
        L2 = y * 2.0 / sqrt3
        L3 = -x + y / sqrt3
        L4 = -x - y / sqrt3
        L5 = -y * 2.0 / sqrt3
        L6 = x - y / sqrt3
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1) # (72, 6)
        
        # Log-sum-exp for S_total
        beta = 50.0
        max_L = np.max(all_L)
        log_sum_exp = max_L + np.log(np.sum(np.exp(all_L - max_L)))
        S_total_smooth = log_sum_exp / beta
        
        # Overlap penalty
        # We need a smooth penalty for overlap of rotated hexagons.
        # Approximate using distance between centers and a "min distance" that depends on relative angle.
        # For two unit hexagons with relative angle delta, the minimum distance for non-overlap varies.
        # Max min-distance is sqrt(3) (when aligned).
        # We can use a conservative bound: d_min = sqrt(3) * (0.8 + 0.2 * abs(cos(3*delta)))? 
        # Actually, let's just use d_min = sqrt(3) as a hard constraint penalty, which is safe but maybe too restrictive for rotated ones?
        # If we use d_min = sqrt(3), we might miss configurations where rotated hexagons can be closer.
        # However, for unit regular hexagons, the minimal distance between centers for non-overlap is indeed bounded by the inradius sum?
        # Inradius r = sqrt(3)/2.
        # If they touch sides, distance is 2*r = sqrt(3).
        # If they touch vertices, distance can be larger.
        # So sqrt(3) is the global minimum distance for ANY orientation.
        # Therefore, using d_min = sqrt(3) is a valid (though potentially non-tight for some orientations) constraint.
        # It forces centers to be at least sqrt(3) apart.
        
        d_min = np.sqrt(3.0)
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                diff = positions[i] - positions[j]
                d = np.linalg.norm(diff)
                val = d_min - d
                sp = np.logaddexp(0, val)
                penalty += sp * sp
                
        lambda_penalty = 100.0
        
        return S_total_smooth + lambda_penalty * penalty

    def objective_grad(params):
        """
        Analytical gradient.
        """
        params = params.reshape((N, 3))
        positions = params[:, :2]
        angles = params[:, 2]
        
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, np.degrees(angles[i]))
            
        verts_flat = verts.reshape(-1, 2)
        
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        L1 = x + y / sqrt3
        L2 = y * 2.0 / sqrt3
        L3 = -x + y / sqrt3
        L4 = -x - y / sqrt3
        L5 = -y * 2.0 / sqrt3
        L6 = x - y / sqrt3
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        
        beta = 50.0
        max_L = np.max(all_L)
        exp_terms = np.exp(all_L - max_L)
        sum_exp = np.sum(exp_terms)
        
        probs = exp_terms / sum_exp
        
        # Gradient w.r.t positions
        # We need d verts / d params.
        # verts[i, k] = positions[i] + R(angles[i]) * local_verts[k]
        # d verts[i, k] / d positions[i] = I
        # d verts[i, k] / d angles[i] = d/dtheta (R(theta) v) = R'(theta) v
        # R'(theta) = [[-sin, -cos], [cos, -sin]]
        
        local_verts = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3.0)/2.0],
            [-0.5, np.sqrt(3.0)/2.0],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3.0)/2.0],
            [0.5, -np.sqrt(3.0)/2.0]
        ])
        
        dL_dx = np.array([1.0, 0.0, -1.0, -1.0, 0.0, 1.0])
        dL_dy = np.array([1.0/sqrt3, 2.0/sqrt3, 1.0/sqrt3, -1.0/sqrt3, -2.0/sqrt3, -1.0/sqrt3])
        
        grad = np.zeros((N, 3))
        
        for i in range(N):
            idx = slice(6*i, 6*(i+1))
            p_i_probs = probs[idx] # (6, )
            
            # Contribution to position gradient
            gx = np.sum(p_i_probs * dL_dx)
            gy = np.sum(p_i_probs * dL_dy)
            grad[i, 0] = gx / beta
            grad[i, 1] = gy / beta
            
            # Contribution to angle gradient
            # d verts / d theta = R'(theta) * local_verts
            # R'(theta) = [[-sin, -cos], [cos, -sin]]
            theta = angles[i]
            s, c = np.sin(theta), np.cos(theta)
            R_prime = np.array([[-s, -c], [c, -s]])
            
            # d verts[i, k] / d theta = R_prime @ local_verts[k]
            # We need sum_k probs[k] * (dL/dx * dx/dtheta + dL/dy * dy/dtheta)
            # dx/dtheta = (R_prime @ local_verts[k])[0]
            # dy/dtheta = (R_prime @ local_verts[k])[1]
            
            # Pre-compute R_prime @ local_verts.T -> (2, 6)
            dv_dtheta = R_prime @ local_verts.T # (2, 6)
            
            gx_dtheta = np.sum(p_i_probs * dL_dx * dv_dtheta[0])
            gy_dtheta = np.sum(p_i_probs * dL_dy * dv_dtheta[1])
            
            grad[i, 2] = (gx_dtheta + gy_dtheta) / beta
            
        # Penalty gradient
        d_min = np.sqrt(3.0)
        lambda_penalty = 100.0
        
        for i in range(N):
            for j in range(i + 1, N):
                diff = positions[i] - positions[j]
                d = np.linalg.norm(diff)
                if d < 1e-8:
                    continue
                unit_diff = diff / d
                val = d_min - d
                if val > 20:
                    sig = 1.0
                elif val < -20:
                    sig = 0.0
                else:
                    sig = 1.0 / (1.0 + np.exp(-val))
                
                sp = np.logaddexp(0, val)
                d_penalty_dd = 2.0 * sp * sig
                
                grad[i, 0] += -lambda_penalty * d_penalty_dd * unit_diff[0]
                grad[i, 1] += -lambda_penalty * d_penalty_dd * unit_diff[1]
                grad[j, 0] += lambda_penalty * d_penalty_dd * unit_diff[0]
                grad[j, 1] += lambda_penalty * d_penalty_dd * unit_diff[1]
                
        return grad.flatten()

    def find_best_orientation(positions):
        """
        Given fixed positions, find the best outer hexagon orientation to minimize side length.
        This is a 1D optimization over the outer hexagon angle.
        """
        def outer_side_for_angle(angle_deg):
            # Outer hexagon has angle angle_deg.
            # We need to find S such that all inner vertices are inside.
            # Inner vertices are fixed.
            # Condition: for each vertex v, and each normal n_k of outer hexagon, n_k . (v - center) <= S.
            # So S >= max_k (n_k . v).
            # Center is assumed to be (0,0) for now? 
            # The task allows outer hexagon to have a center.
            # To minimize S, we should also optimize the center.
            # But for now, let's assume center is (0,0) or centroid of vertices.
            # Actually, the minimal enclosing hexagon of a set of points with fixed orientation 
            # is determined by the support function.
            # Let's just compute S for center (0,0) and outer angle angle_deg.
            
            verts = np.zeros((N, 6, 2))
            for i in range(N):
                verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0)
            verts_flat = verts.reshape(-1, 2)
            
            # Outer hexagon normals
            angle_rad = np.radians(angle_deg)
            # Normals are at angle_rad + 30, 90, ...
            normal_angles = angle_rad + np.radians(np.array([30, 90, 150, 210, 270, 330]))
            normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
            
            # Project all vertices onto normals
            # (72, 2) @ (2, 6).T -> (72, 6)
            projections = verts_flat @ normals.T
            max_proj = np.max(projections, axis=0)
            
            # The side length S is the max of these projections (assuming center 0,0).
            # Wait, if center is not 0,0, we subtract center.
            # Let's assume we will center the configuration later.
            return np.max(max_proj)
        
        # Minimize outer_side_for_angle
        # Use golden section or similar, or just check a few angles.
        # The function is periodic with period 60 degrees.
        # Let's sample 60 angles and take the min.
        angles = np.linspace(0, 60, 61, endpoint=False)
        sides = [outer_side_for_angle(a) for a in angles]
        return min(sides), angles[np.argmin(sides)]

    # Initial configurations
    configs = []
    
    # 1. Hexagonal lattice
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    init1 = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        init1.append([x, y, 0.0])
    configs.append(np.array(init1).flatten())
    
    # 2. Circular arrangement
    init2 = np.zeros(N * 3)
    for i in range(N):
        angle = 2 * np.pi * i / N
        r = 2.0
        init2[3*i] = r * np.cos(angle)
        init2[3*i+1] = r * np.sin(angle)
        init2[3*i+2] = 0.0
    configs.append(init2)
    
    # 3. Random
    np.random.seed(42)
    init3 = np.random.uniform(-5, 5, N * 3)
    configs.append(init3)
    
    best_val = float('inf')
    best_params = None
    
    for init in configs:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                objective_smooth,
                init,
                jac=objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            val = res.fun
            if val < best_val:
                best_val = val
                best_params = res.x
        except Exception as e:
            pass

    # Fallback
    if best_params is None:
        best_params = configs[0]
        
    # Final validation and exact calculation
    params = best_params.reshape((N, 3))
    positions = params[:, :2]
    angles_deg = np.degrees(params[:, 2])
    
    if not check_validity(positions, angles_deg):
        # If invalid, try to find a valid one from initial configs
        for init in configs:
            p = init.reshape((N, 3))
            if check_validity(p[:, :2], np.degrees(p[:, 2])):
                best_params = init
                break
        else:
            # Last resort: spread out
            best_params = np.array([i * 3.0 for i in range(N)] + [0.0 for i in range(N)] + [0.0 for i in range(N)])
    
    params = best_params.reshape((N, 3))
    positions = params[:, :2]
    angles_deg = np.degrees(params[:, 2])
    
    # Calculate exact outer hexagon side length
    # We need to find the best outer hexagon orientation and center.
    # For now, let's assume the optimal outer hexagon is aligned with the "average" orientation or we can just scan.
    # Scanning 60 orientations is fast.
    
    def get_outer_side(positions, angles_deg, outer_angle_deg):
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
        verts_flat = verts.reshape(-1, 2)
        
        angle_rad = np.radians(outer_angle_deg)
        normal_angles = angle_rad + np.radians(np.array([30, 90, 150, 210, 270, 330]))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        projections = verts_flat @ normals.T
        max_proj = np.max(projections, axis=0)
        return np.max(max_proj)
    
    # Scan outer angles
    outer_angles = np.linspace(0, 60, 61, endpoint=False)
    sides = [get_outer_side(positions, angles_deg, a) for a in outer_angles]
    exact_side = min(sides)
    
    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), float(angles_deg[i])])
        
    return inner, (0.0, 0.0), float(exact_side), 0.0

# EVOLVE-BLOCK-END
