import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def required_outer_side(positions):
        """
        Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
        that contains all unit hexagons at given positions.
        """
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0)
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
        return np.max(all_L)

    def check_validity(positions):
        """
        Check if all hexagons are pairwise disjoint using SAT.
        """
        vertices_list = []
        for i in range(N):
            x, y = positions[i]
            verts = hexagon_vertices(x, y, 1.0, 0.0)
            vertices_list.append(verts)
        
        for i in range(N):
            for j in range(i + 1, N):
                p1 = vertices_list[i]
                p2 = vertices_list[j]
                
                # SAT check
                disjoint = False
                for i_sat in range(6):
                    j_sat = (i_sat + 1) % 6
                    edge = p1[j_sat] - p1[i_sat]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10:
                        continue
                    normal /= norm
                    proj1 = p1.dot(normal)
                    proj2 = p2.dot(normal)
                    if np.max(proj1) < np.min(proj2) - 1e-5 or np.max(proj2) < np.min(proj1) - 1e-5:
                        disjoint = True
                        break
                
                if not disjoint:
                    for i_sat in range(6):
                        j_sat = (i_sat + 1) % 6
                        edge = p2[j_sat] - p2[i_sat]
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm < 1e-10:
                            continue
                        normal /= norm
                        proj1 = p1.dot(normal)
                        proj2 = p2.dot(normal)
                        if np.max(proj1) < np.min(proj2) - 1e-5 or np.max(proj2) < np.min(proj1) - 1e-5:
                            disjoint = True
                            break
                    if not disjoint:
                        return False
        return True

    def objective_and_grad(params):
        """
        Objective function for gradient-based optimization.
        Params: [x1, y1, ..., xN, yN]
        Returns: (value, gradient)
        Value = Outer_Side + Penalty * Sum(max(0, min_dist - separation)^2)
        """
        positions = params.reshape(N, 2)
        
        # 1. Calculate Outer Side
        side = required_outer_side(positions)
        
        # Gradient of Outer Side
        # The side length is max of 6 linear functions. 
        # The gradient is the normal of the active constraint(s).
        # For a regular hexagon containing points, the "active" constraint is the one
        # that determines the max extent.
        
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0)
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
        
        # Find max value and which axis it corresponds to
        max_val = np.max(all_L)
        max_indices = np.where(all_L == max_val)[0]
        
        grad_side = np.zeros((N, 2))
        # Define normals for the 6 sides of the outer hexagon (unit normals pointing inwards? No, outwards for support function)
        # The support function h(u) = max p . u. 
        # For a regular hexagon centered at origin, the side length S is related to the support number h.
        # Specifically, for a hexagon of side S, the distance from center to side is S * sqrt(3)/2.
        # The support function in direction u (unit vector) for the hexagon is S * sqrt(3)/2 * (u . n_k) where n_k is the normal?
        # Actually, it's easier to just compute the gradient numerically or analytically for the active constraint.
        
        # The constraint is that all points must satisfy:
        # p . n_1 <= R_1
        # p . n_2 <= R_2
        # ...
        # where R_k is the support distance of the outer hexagon in direction n_k.
        # For a regular hexagon of side S, the support distance in the direction of the normal to a side is S * sqrt(3)/2.
        # The normals to the sides of a regular hexagon (angle 0) are at angles 0, 60, 120, 180, 240, 300.
        
        normals = []
        for k in range(6):
            angle = np.radians(k * 60)
            normals.append([np.cos(angle), np.sin(angle)])
        normals = np.array(normals)
        
        # The value calculated in required_outer_side is actually the max of (p . n_k) / (sqrt(3)/2) ?
        # Let's check:
        # L1 = x + y/sqrt3. Normal is (1, 1/sqrt3). Norm is sqrt(1 + 1/3) = 2/sqrt3.
        # Unit normal is (sqrt3/2, 1/2).
        # p . (sqrt3/2, 1/2) = x*sqrt3/2 + y/2.
        # My L1 is x + y/sqrt3 = (2/sqrt3) * (x*sqrt3/2 + y/2) = (2/sqrt3) * (p . n_unit).
        # So L1 = (2/sqrt3) * h_1.
        # The side length S is such that h_1 = S * sqrt(3)/2.
        # So S = h_1 / (sqrt(3)/2) = 2 * h_1 / sqrt(3).
        # And L1 = (2/sqrt3) * h_1 = S.
        # So required_outer_side returns S directly.
        
        # Gradient of S with respect to positions:
        # S = max_k (p_i . n_k_unit) * (2/sqrt3)
        # dS/dp_i = (2/sqrt3) * n_k_unit for the active k.
        # If multiple are active, we can take the average or just pick one.
        
        # Map indices 0..5 to normals
        # L1 corresponds to normal (1, 1/sqrt3) -> angle 30 deg? 
        # Wait, standard hexagon vertices are at 0, 60, 120...
        # Side 1 is between vertex 0 and 1. Midpoint angle 30. Normal angle 30.
        # My L1: x + y/sqrt3. 
        # Let's just use the subgradient of the max function.
        
        active_normals = []
        for idx in max_indices:
            # Determine which normal corresponds to this linear form
            # L0: x + y/sqrt3  -> n=(1, 1/sqrt3)
            # L1: 2y/sqrt3    -> n=(0, 2/sqrt3)
            # L2: -x + y/sqrt3 -> n=(-1, 1/sqrt3)
            # L3: -x - y/sqrt3 -> n=(-1, -1/sqrt3)
            # L4: -2y/sqrt3   -> n=(0, -2/sqrt3)
            # L5: x - y/sqrt3 -> n=(1, -1/sqrt3)
            
            if idx == 0: n = np.array([1.0, 1.0/np.sqrt(3)])
            elif idx == 1: n = np.array([0.0, 2.0/np.sqrt(3)])
            elif idx == 2: n = np.array([-1.0, 1.0/np.sqrt(3)])
            elif idx == 3: n = np.array([-1.0, -1.0/np.sqrt(3)])
            elif idx == 4: n = np.array([0.0, -2.0/np.sqrt(3)])
            else: n = np.array([1.0, -1.0/np.sqrt(3)])
            
            active_normals.append(n)
        
        if active_normals:
            # Average the gradients for active constraints
            avg_n = np.mean(active_normals, axis=0)
            # Scale factor: S = (2/sqrt3) * (p . n_unscaled)? 
            # Let's verify: p . n_unscaled = L_idx. S = L_idx.
            # So dS/dp = n_unscaled.
            # Wait, S = L_idx. So gradient is just n_unscaled.
            # But n_unscaled is not unit length.
            # L1 = x + y/sqrt3. dL1/dx = 1, dL1/dy = 1/sqrt3.
            # So gradient is indeed the vector in the formula.
            
            for i in range(N):
                grad_side[i] = avg_n
        
        # 2. Calculate Collision Penalty
        # We want to minimize the maximum overlap.
        # Approximate overlap by distance between centers minus minimum separation.
        # For two unit hexagons, min separation is 2 * (sqrt(3)/2) = sqrt(3) if aligned?
        # Actually, the "radius" of a hexagon is 1 (vertex to center).
        # The inradius is sqrt(3)/2.
        # Two hexagons touch if distance between centers >= 2 * inradius = sqrt(3) (for flat-to-flat) or 2 (for vertex-to-vertex)?
        # The minimum distance between two disjoint unit hexagons is the width of the hexagon, which is sqrt(3).
        # So we penalize if dist < sqrt(3).
        
        penalty = 0.0
        grad_penalty = np.zeros((N, 2))
        
        min_dist = np.sqrt(3.0)
        penalty_weight = 10.0
        
        for i in range(N):
            for j in range(i + 1, N):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                
                if dist < min_dist:
                    overlap = min_dist - dist
                    # Penalty: overlap^2
                    penalty += overlap * overlap
                    # Gradient of overlap^2 w.r.t pos_i
                    # d(overlap^2)/dpos_i = 2 * overlap * d(overlap)/dpos_i
                    # d(overlap)/dpos_i = - d(dist)/dpos_i = - diff / dist
                    if dist > 1e-6:
                        grad_vec = -2.0 * overlap * (diff / dist)
                    else:
                        grad_vec = np.array([1.0, 0.0]) # Avoid division by zero
                    
                    grad_penalty[i] += penalty_weight * grad_vec
                    grad_penalty[j] -= penalty_weight * grad_vec
        
        value = side + penalty_weight * penalty
        grad = grad_side + grad_penalty
        
        return value, grad.flatten()

    # Initial configurations
    configs = []
    
    # 1. Hexagonal lattice (compact)
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    init1 = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        init1.append([x, y])
    configs.append(np.array(init1))
    
    # 2. Circular arrangement
    init2 = np.zeros((N, 2))
    for i in range(N):
        angle = 2 * np.pi * i / N
        r = 2.5
        init2[i, 0] = r * np.cos(angle)
        init2[i, 1] = r * np.sin(angle)
    configs.append(init2)
    
    # 3. Random
    np.random.seed(42)
    init3 = np.random.uniform(-5, 5, (N, 2))
    configs.append(init3)
    
    best_positions = None
    best_side = float('inf')
    
    for init in configs:
        if time.time() > deadline:
            break
        
        # Ensure initial config is valid; if not, spread out
        if not check_validity(init):
            for scale in [1.2, 1.5, 2.0, 2.5, 3.0]:
                scaled = init * scale
                if check_validity(scaled):
                    init = scaled
                    break
            else:
                continue
        
        # Run L-BFGS-B
        try:
            res = minimize(
                objective_and_grad,
                init.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-5, 'maxfun': 10000}
            )
            
            opt_positions = res.x.reshape(N, 2)
            
            # Check validity of the optimized result
            if check_validity(opt_positions):
                side = required_outer_side(opt_positions)
                if side < best_side:
                    best_side = side
                    best_positions = opt_positions
            else:
                # If invalid, we might need to project back or try a different init
                # For now, skip if invalid. The penalty should have prevented this mostly.
                pass
                
        except Exception as e:
            pass # Continue to next config

    # Fallback
    if best_positions is None:
        best_positions = configs[0]
        best_side = required_outer_side(best_positions)
    
    # Final validation
    if not check_validity(best_positions):
        # Try to find a valid config
        for init in configs:
            pos = init.copy()
            for scale in [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
                scaled = pos * scale
                if check_validity(scaled):
                    best_positions = scaled
                    best_side = required_outer_side(best_positions)
                    break
            if check_validity(best_positions):
                break
        else:
            # Last resort: spread out linearly
            best_positions = np.array([[i * 3.0, 0.0] for i in range(N)])
            best_side = required_outer_side(best_positions)
    
    # Calculate final exact side
    final_side = required_outer_side(best_positions)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), 0.0])
        
    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END
