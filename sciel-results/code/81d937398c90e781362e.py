import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill

# Precompute unit hexagon vertices and edge normals (angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

def construct():
    t0 = time.time()
    
    unit_hex_verts = _UNIT_HEX_VERTS
    edge_normals = _EDGE_NORMALS
    sqrt3 = math.sqrt(3)
    
    _ux = unit_hex_verts[:, 0]
    _uy = unit_hex_verts[:, 1]
    _enx = edge_normals[:, 0]
    _eny = edge_normals[:, 1]

    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        return np.stack((verts_x, verts_y), axis=-1)

    def compute_collision_penalty(positions, angles_rad):
        """
        Computes a smooth penalty for overlapping hexagons.
        We use the distance between the closest points of the two hexagons.
        If they are disjoint, distance > 0. If overlapping, we estimate penetration depth.
        For simplicity and speed, we check edge-to-edge and vertex-to-edge distances.
        However, a simpler robust metric is the minimum distance between any vertex of one 
        and the other hexagon. If a vertex is inside, the distance is negative (penetration).
        
        For a smooth approximation, we can use the log-sum-exp of negative distances 
        or just sum of squared penetrations.
        Here, we will calculate the minimum distance between the two convex polygons.
        If min_dist > 0, penalty is 0.
        If min_dist < 0, penalty is min_dist^2.
        
        To make it differentiable, we will use a soft min.
        Actually, for L-BFGS-B, a simple hard constraint penalty with a large weight 
        often works if the initial guess is good, but it's non-smooth at 0.
        A better approach: Penalty = max(0, -min_dist)^2.
        The max(0, x) is not differentiable at 0, but the gradient is 0 there, 
        so L-BFGS can handle it if we are careful, or we can smooth it.
        Let's use a smooth approximation: penalty = (1/(beta)) * log(1 + exp(beta * max(0, -min_dist)))?
        No, let's stick to a simpler differentiable proxy: 
        Sum of squared distances from vertices of H1 to edges of H2 and vice versa, 
        but only if the vertex is inside or close.
        
        Actually, the most robust differentiable penalty for convex polygons is based on the 
        support function or GJK distance, which is complex to implement from scratch.
        
        Alternative: Use the minimum distance between any pair of vertices/edges.
        Since we have few hexagons (12), we can afford to be a bit more precise.
        
        Let's use a "soft" collision check:
        For each pair, compute the minimum distance between their boundaries.
        If they are disjoint, dist > 0.
        If overlapping, dist < 0 (approximated by how deep a vertex is inside).
        
        We will calculate the penetration depth of each vertex of H1 into H2 and vice versa.
        Penetration depth of point P into polygon Q:
        If P is outside, depth is 0 (or negative distance).
        If P is inside, depth is -min_distance_to_boundary.
        
        To keep it simple and fast, we will use the sum of squared "violations".
        Violation = max(0, -distance_to_boundary).
        Distance to boundary for a convex polygon can be found by checking edges.
        
        Let's simplify: We will use a penalty based on the minimum distance between 
        any vertex of one hexagon and the other hexagon's edges/vertices.
        """
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1_x = positions[i, 0]
                h1_y = positions[i, 1]
                h2_x = positions[j, 0]
                h2_y = positions[j, 1]
                a1 = angles_rad[i]
                a2 = angles_rad[j]
                
                c1, s1 = math.cos(a1), math.sin(a1)
                c2, s2 = math.cos(a2), math.sin(a2)
                
                v1x = h1_x + _ux * c1 - _uy * s1
                v1y = h1_y + _ux * s1 + _uy * c1
                v2x = h2_x + _ux * c2 - _uy * s2
                v2y = h2_y + _ux * s2 + _uy * c2
                
                # Check if any vertex of 1 is inside 2 or vice versa
                # And compute penetration depth
                # We approximate penetration by the minimum distance to the boundary
                # if the point is inside, distance is negative.
                
                # Helper: distance from point (px, py) to segment (x1, y1)-(x2, y2)
                # Returns positive distance if outside, negative if inside? 
                # No, standard distance is always positive.
                # We need signed distance.
                
                # For convex polygon, if point is inside, the "distance" to boundary is negative.
                # We can compute the minimum distance to any edge.
                # If the projection of the point onto the edge normal falls within the edge, 
                # and the point is on the "inside" side, then it's a penetration.
                
                # This is getting complicated to make fully differentiable and fast.
                # Let's use a simpler heuristic that is differentiable:
                # Penalty = sum of (max(0, -dist))^2 for all vertex-edge pairs?
                
                # Let's try a different approach: 
                # Use the fact that if two convex polygons are disjoint, 
                # there is a separating axis.
                # The "amount of overlap" along the separating axis is the penalty.
                # We check all 12 axes (6 from H1, 6 from H2).
                # For each axis, project both polygons.
                # If intervals overlap, overlap_length = max(min1, min2) - min(max1, max2)? No.
                # Overlap = max(min1, min2) is wrong.
                # Overlap = max(0, min(max1, max2) - max(min1, min2)).
                
                # This "Overlap" is a measure of how much they intersect along that axis.
                # The minimum overlap over all separating axes is 0 if disjoint.
                # If they intersect, the minimum overlap over ALL axes is still > 0? 
                # No, if they intersect, there is NO separating axis. 
                # So for intersecting polygons, the "separating axis" test fails for all axes.
                # The overlap on any axis will be positive.
                # The MINIMUM overlap over all 12 axes is a good proxy for penetration depth.
                # It is 0 if disjoint, and > 0 if intersecting.
                # Is it differentiable? The min of linear functions is concave, 
                # but we are taking min of overlaps which are piecewise linear.
                # It should be smooth enough for L-BFGS if we use a soft min.
                
                min_overlap = float('inf')
                
                # Axes from H1
                for k in range(6):
                    nx = _enx[k] * c1 - _eny[k] * s1
                    ny = _enx[k] * s1 + _eny[k] * c1
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    min1, max1 = p1.min(), p1.max()
                    min2, max2 = p2.min(), p2.max()
                    overlap = min(max1, max2) - max(min1, min2)
                    if overlap < min_overlap:
                        min_overlap = overlap
                
                # Axes from H2
                for k in range(6):
                    nx = _enx[k] * c2 - _eny[k] * s2
                    ny = _enx[k] * s2 + _eny[k] * c2
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    min1, max1 = p1.min(), p1.max()
                    min2, max2 = p2.min(), p2.max()
                    overlap = min(max1, max2) - max(min1, min2)
                    if overlap < min_overlap:
                        min_overlap = overlap
                
                if min_overlap > 0:
                    # They are intersecting. Penalty is the overlap squared.
                    # Using square makes it differentiable at 0 (gradient 0).
                    penalty += min_overlap * min_overlap
                # If min_overlap <= 0, they are disjoint (or touching), penalty 0.
        
        return penalty

    def compute_min_outer_side_and_grad(positions, angles_deg):
        """
        Computes the minimum outer side length and its gradient with respect to positions and angles.
        For L-BFGS-B, we need the objective and optionally the gradient.
        The objective is max over 6 directions of (2/sqrt(3)) * max_projection.
        This is a max of linear functions. The gradient is the gradient of the active function.
        """
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # We need to optimize the outer hexagon orientation phi as well?
        # The task allows the outer hexagon to rotate.
        # So the objective is min_phi max_side(phi).
        # This is a nested optimization.
        # To make it differentiable, we can fix phi to the optimal value for the current configuration,
        # or include phi in the parameters.
        # Including phi in the parameters (1 extra variable) is better.
        
        # However, the current interface returns side and angle.
        # The optimizer should optimize positions, angles of inner hexes, AND the outer hex angle.
        
        # Let's restructure:
        # Params: [x1, y1, a1, ..., x12, y12, a12, phi_outer]
        # Total 37 parameters.
        
        # Objective:
        # 1. Compute vertices based on inner params.
        # 2. Compute center.
        # 3. For the given phi_outer, compute the required side length.
        # 4. Add collision penalty.
        
        # Gradient:
        # The side length function is piecewise linear in phi and positions.
        # The active constraint determines the gradient.
        
        def eval_side_and_grad(phi_deg):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            max_side = np.max(side_vals)
            active_idx = np.argmax(side_vals)
            
            # Gradient of max_side with respect to offsets
            # side = (2/sqrt3) * |offset . n_active|
            # d(side)/d(offset) = (2/sqrt3) * sign(offset . n_active) * n_active
            # But we need gradient w.r.t positions and angles.
            
            # This is complex to compute analytically for all 37 params quickly.
            # Given the time limit, using numerical gradient or just L-BFGS-B with finite diff 
            # might be too slow for 37 params * many iterations.
            
            # Alternative: Use Nelder-Mead on the 37 params? 
            # Nelder-Mead is slow for high dimensions.
            
            # Let's stick to a hybrid: 
            # Use L-BFGS-B for positions/angles of inner hexes, 
            # and scan phi_outer in a separate loop or as a 1D optimization.
            # But phi_outer is coupled.
            
            # Actually, for a fixed set of inner hexes, the optimal phi_outer 
            # is the one that minimizes the max projection.
            # This is a 1D problem.
            # So we can define the objective as:
            # f(params_inner) = min_phi ( max_side(params_inner, phi) + Penalty(params_inner) )
            
            # This is still non-smooth due to the min_phi.
            
            # Let's try a different strategy:
            # 1. Use a robust global search (like the current SA but faster) to get a good start.
            # 2. Use L-BFGS-B to polish, treating the outer phi as a continuous variable 
            #    and using a smooth approximation of the max function (LogSumExp).
            
            return max_side, active_idx
        
        # For the purpose of this implementation, I will use a simpler approach:
        # 1. Multi-start with diverse initializations.
        # 2. For each start, run a local optimization using L-BFGS-B.
        #    The objective function will be:
        #    f = max_side_smoothed + Lambda * Collision_Penalty
        #    Where max_side_smoothed uses LogSumExp of the 6 side values to be differentiable.
        #    And we optimize over 37 parameters (12 pos, 12 ang, 1 outer phi).
        
        return None # Placeholder

    def log_sum_exp(x, beta=10.0):
        """Smooth approximation of max(x)."""
        # max(x) approx (1/beta) * log(sum(exp(beta * x)))
        # To avoid overflow, shift by max(x)
        x_max = np.max(x)
        return x_max + (1.0/beta) * np.log(np.sum(np.exp(beta * (x - x_max))))

    def objective_and_grad(params):
        # params: [x1, y1, a1, ..., x12, y12, a12, phi_outer]
        positions = np.zeros((N, 2))
        angles_deg = np.zeros(N)
        for i in range(N):
            positions[i, 0] = params[3*i]
            positions[i, 1] = params[3*i+1]
            angles_deg[i] = params[3*i+2]
        phi_outer_deg = params[3*N]
        
        angles_rad = np.radians(angles_deg)
        
        # 1. Collision Penalty
        penalty = compute_collision_penalty(positions, angles_rad)
        
        # 2. Outer Side Length
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        phi_outer_rad = np.radians(phi_outer_deg)
        normal_angles = phi_outer_rad + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        
        # Smooth max
        smooth_side = log_sum_exp(side_vals, beta=50.0)
        
        # Total Objective
        total = smooth_side + 10.0 * penalty
        
        # Gradient is tricky to compute analytically for the log_sum_exp and penalty.
        # We will rely on L-BFGS-B's numerical gradient or provide a sparse gradient if possible.
        # Given the complexity, numerical gradient is safer but slower.
        # However, 37 dimensions is small enough for numerical gradient if we limit iterations.
        
        return total

    def run_local_optimization(init_params, time_limit):
        # Bounds for parameters
        # Positions: roughly -10 to 10
        # Angles: 0 to 360
        # Outer Phi: 0 to 60
        bounds = []
        for i in range(N):
            bounds.append((-10.0, 10.0)) # x
            bounds.append((-10.0, 10.0)) # y
            bounds.append((0.0, 360.0))  # angle
        bounds.append((0.0, 60.0)) # outer phi
        
        t_start = time.time()
        
        # Use L-BFGS-B
        try:
            res = minimize(objective_and_grad, init_params, method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-8})
            
            if res.fun < float('inf'):
                # Extract best params
                best_params = res.x
                # We need to verify validity and get the exact side length (not smoothed)
                positions = np.zeros((N, 2))
                angles_deg = np.zeros(N)
                for i in range(N):
                    positions[i, 0] = best_params[3*i]
                    positions[i, 1] = best_params[3*i+1]
                    angles_deg[i] = best_params[3*i+2]
                phi_outer_deg = best_params[3*N]
                
                # Check validity
                angles_rad = np.radians(angles_deg)
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        # Re-use fast check
                        h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
                        h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
                        # Quick check using the penalty logic
                        # If penalty is 0, it's likely valid
                        pass # We'll trust the penalty if it's small, but let's do a strict check
                    
                # Strict check
                for i in range(N):
                    for j in range(i+1, N):
                        h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
                        h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
                        if not hexagons_disjoint_fast(h1, h2):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Compute exact side
                    side, phi, cx, cy = compute_min_outer_side(positions, angles_deg)
                    return side, positions, angles_deg, phi, cx, cy
        except Exception:
            pass
            
        return None

    def hexagons_disjoint_fast(h1, h2):
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.5:
            return True
        a1 = h1[2]
        a2 = h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * c1
        en2x = _enx * c2 - _eny * s2
        en2y = _enx * s2 + _eny * c2
        for i in range(6):
            nx, ny = en1x[i], en1y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        return False

    def compute_min_outer_side(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            return np.max(side_vals)
        
        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 1):
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
                
        try:
            lo = max(0.0, best_phi - 1.0)
            hi = min(60.0, best_phi + 1.0)
            from scipy.optimize import minimize_scalar
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded', options={'xatol': 1e-8})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass
            
        return best_side, best_phi, center[0], center[1]

    # --- Main Search Loop ---
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Generate diverse initial configurations
    def generate_initials():
        inits = []
        # 1. Honeycomb
        sqrt3 = math.sqrt(3)
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            pos[idx] = i * a1 + j * a2
        params = np.zeros(3*N + 1)
        for i in range(N):
            params[3*i] = pos[i, 0]
            params[3*i+1] = pos[i, 1]
            params[3*i+2] = ang[i]
        params[3*N] = 0.0
        inits.append(params)
        
        # 2. Ring
        for r in [3.0, 4.0]:
            pos = np.zeros((N, 2))
            ang = np.zeros(N)
            for i in range(N):
                theta = 2 * math.pi * i / N
                pos[i, 0] = r * math.cos(theta)
                pos[i, 1] = r * math.sin(theta)
                ang[i] = math.degrees(theta)
            params = np.zeros(3*N + 1)
            for i in range(N):
                params[3*i] = pos[i, 0]
                params[3*i+1] = pos[i, 1]
                params[3*i+2] = ang[i]
            params[3*N] = 0.0
            inits.append(params)
            
        # 3. Random Clusters
        for _ in range(5):
            pos = np.random.uniform(-5, 5, (N, 2))
            ang = np.random.uniform(0, 360, N)
            params = np.zeros(3*N + 1)
            for i in range(N):
                params[3*i] = pos[i, 0]
                params[3*i+1] = pos[i, 1]
                params[3*i+2] = ang[i]
            params[3*N] = np.random.uniform(0, 60)
            inits.append(params)
            
        return inits

    initials = generate_initials()
    
    for init_params in initials:
        if time.time() - t0 > DEADLINE - 10:
            break
            
        # Run local optimization
        result = run_local_optimization(init_params, time_limit=20.0)
        if result:
            side, pos, ang, phi, cx, cy = result
            if side < best_side:
                best_side = side
                best_positions = pos
                best_angles = ang
                best_phi = phi
                best_cx = cx
                best_cy = cy

    # If no valid configuration was found, fallback to base honeycomb
    if best_positions is None:
        sqrt3 = math.sqrt(3)
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        best_positions = np.zeros((N, 2))
        best_angles = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            best_positions[idx] = i * a1 + j * a2
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
