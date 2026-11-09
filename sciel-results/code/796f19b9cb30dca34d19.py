import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin (evaluator limit 150s)
    
    SQRT3 = math.sqrt(3)
    
    # Helper: Get vertices of a unit hexagon centered at (cx, cy) with rotation angle
    def get_hex_vertices(cx, cy, angle_deg):
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    # Helper: Get axes for SAT (normals to edges)
    def get_hex_axes(angle_deg):
        axes = []
        for i in range(6):
            normal_angle = math.radians(angle_deg + i * 60 + 30)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            axes.append((nx, ny))
        return axes

    def project_poly(vertices, axis):
        nx, ny = axis
        min_val = float('inf')
        max_val = float('-inf')
        for vx, vy in vertices:
            proj = vx * nx + vy * ny
            if proj < min_val:
                min_val = proj
            if proj > max_val:
                max_val = proj
        return min_val, max_val

    def is_disjoint(hex1_verts, hex2_verts, axes1, axes2, d_sq):
        """
        Fast disjoint check.
        1. If d_sq > 4.0 (d > 2.0), definitely disjoint.
        2. If d_sq < 3.0 (d < sqrt(3)), definitely overlap (min distance for disjoint is sqrt(3)).
        3. Otherwise, use SAT.
        """
        if d_sq > 4.0:
            return True
        if d_sq < 3.0:
            return False
            
        # SAT check
        all_axes = axes1 + axes2
        for axis in all_axes:
            min1, max1 = project_poly(hex1_verts, axis)
            min2, max2 = project_poly(hex2_verts, axis)
            if max1 < min2 or max2 < min1:
                return True
        return False

    def calc_outer_side(hexagons, outer_angle_deg):
        """
        Calculate the minimum side length S of a regular hexagon centered at (0,0)
        with rotation outer_angle_deg that contains all given hexagons.
        """
        base_angles = [30, 90, 150, 210, 270, 330]
        axis_dirs = []
        for a in base_angles:
            rad = math.radians(a + outer_angle_deg)
            axis_dirs.append((math.cos(rad), math.sin(rad)))
        
        max_proj = 0.0
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for nx, ny in axis_dirs:
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        
        # Apothem of outer hex is S * SQRT3 / 2
        # S >= 2 * max_proj / SQRT3
        return 2.0 * max_proj / SQRT3

    def calc_outer_side_smooth(hexagons, outer_angle_deg, temp=0.1):
        """
        Smooth version using log-sum-exp for the max projection.
        """
        base_angles = [30, 90, 150, 210, 270, 330]
        axis_dirs = []
        for a in base_angles:
            rad = math.radians(a + outer_angle_deg)
            axis_dirs.append((math.cos(rad), math.sin(rad)))
        
        max_proj_sq = 0.0
        max_val = 0.0
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for nx, ny in axis_dirs:
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_val:
                        max_val = proj
        
        # Use a simple smooth max approximation: max + temp * sum(exp((p - max)/temp))
        # But for just one value, standard max is fine. 
        # The kink is in the switching of which vertex/axis is max.
        # To make it differentiable, we can use the sum of squares approach or log-sum-exp.
        # Since we have discrete vertices, the objective is piecewise linear.
        # Nelder-Mead handles piecewise linear well, but the log-sum-exp trick
        # mentioned in inspirations suggests using a smooth surrogate.
        # Let's try a soft-min/soft-max approach.
        
        # Actually, the "kink" is at the boundary where the active constraint changes.
        # A common trick is to use:
        # max(x_i) ≈ (1/K) * log(sum(exp(K*x_i)))
        # But here x_i are the projections.
        
        # Let's stick to the hard max for the final result, but use smooth for optimization?
        # The inspiration said: "Smooth log-sum-exp outer side objective during Nelder-Mead polish 
        # removes the kink at the max projection vertex"
        
        # Implementation:
        # Projections P.
        # Smooth Max ≈ (1/beta) * log( sum(exp(beta * p_i)) )
        
        beta = 10.0
        projs = []
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for nx, ny in axis_dirs:
                    projs.append(abs(vx * nx + vy * ny))
        
        if not projs:
            return 0.0
            
        # Find max to prevent overflow
        p_max = max(projs)
        sum_exp = sum(math.exp(beta * (p - p_max)) for p in projs)
        smooth_max = p_max + (1.0 / beta) * math.log(sum_exp)
        
        return 2.0 * smooth_max / SQRT3

    def evaluate(p, outer_angle_deg=0.0, smooth=False):
        centers = []
        angles = []
        for i in range(N):
            centers.append((p[i*3], p[i*3+1]))
            angles.append(p[i*3+2])
        
        cx_avg = sum(c[0] for c in centers) / N
        cy_avg = sum(c[1] for c in centers) / N
        shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
        
        hexagons = []
        hex_axes = []
        for i in range(N):
            cx, cy = shifted_centers[i]
            a = angles[i]
            hex_verts = get_hex_vertices(cx, cy, a)
            hexagons.append(hex_verts)
            hex_axes.append(get_hex_axes(a))
        
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = shifted_centers[i][0] - shifted_centers[j][0]
                dy = shifted_centers[i][1] - shifted_centers[j][1]
                d_sq = dx*dx + dy*dy
                if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j], d_sq):
                    overlap_penalty += 1000.0
        
        if smooth:
            side = calc_outer_side_smooth(hexagons, outer_angle_deg)
        else:
            side = calc_outer_side(hexagons, outer_angle_deg)
            
        return overlap_penalty + side

    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Initial configurations
    # 1. Flower + extensions
    coords1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
    # 2. Compact 3-5-3
    coords2 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
    # 3. Linear
    coords3 = [(0,0), (1,0), (2,0), (3,0), (4,0), (-1,0), (-2,0), (0,1), (1,1), (2,1), (3,1)]
    # 4. Hexagonal lattice 3-3-3-2 (offset)
    coords4 = [(-1,-1),(0,-1),(1,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1),(2,0),(2,1)]
    # 5. Dense cluster
    coords5 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,1), (-2,1), (-2,0)]
    # 6. Another variant
    coords6 = [(-1,-1),(0,-1),(1,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1),(0,2),(1,2)]
    
    configs = [coords1, coords2, coords3, coords4, coords5, coords6]
    
    best_params = None
    best_score = float('inf')
    
    def evaluate_rigid(base_centers, base_angles, tx, ty, theta_deg, scale):
        """
        Evaluate a rigid transformation of a base configuration.
        """
        theta_rad = math.radians(theta_deg)
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)
        
        centers = []
        
        for i in range(N):
            bx, by = base_centers[i]
            # Scale and rotate
            rx = bx * scale
            ry = by * scale
            x = rx * cos_t - ry * sin_t + tx
            y = rx * sin_t + ry * cos_t + ty
            centers.append((x, y))
        
        # Center the configuration
        cx_avg = sum(c[0] for c in centers) / N
        cy_avg = sum(c[1] for c in centers) / N
        shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
        
        hexagons_final = []
        hex_axes_final = []
        for i in range(N):
            cx, cy = shifted_centers[i]
            a = base_angles[i] + theta_deg
            hex_verts = get_hex_vertices(cx, cy, a)
            hexagons_final.append(hex_verts)
            hex_axes_final.append(get_hex_axes(a))
        
        # Check overlaps
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = shifted_centers[i][0] - shifted_centers[j][0]
                dy = shifted_centers[i][1] - shifted_centers[j][1]
                d_sq = dx*dx + dy*dy
                if not is_disjoint(hexagons_final[i], hexagons_final[j], hex_axes_final[i], hex_axes_final[j], d_sq):
                    overlap_penalty += 1000.0
        
        side = calc_outer_side(hexagons_final, 0.0)
        return overlap_penalty + side

    # Run SA on rigid parameters for each base config
    for config in configs:
        if time.time() > deadline - 30.0:
            break
            
        base_centers = []
        base_angles = []
        for q, r in config:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            base_centers.append((x, y))
            base_angles.append(0.0)
        
        # Center the base config
        cx_avg = sum(c[0] for c in base_centers) / N
        cy_avg = sum(c[1] for c in base_centers) / N
        base_centers = [(x - cx_avg, y - cy_avg) for x, y in base_centers]
        
        rigid_params = np.array([0.0, 0.0, 0.0, 1.0])
        
        current_score = evaluate_rigid(base_centers, base_angles, *rigid_params)
        local_best_params = rigid_params.copy()
        local_best_score = current_score
        
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.995
        
        while T > T_min and time.time() < deadline - 15.0:
            new_params = rigid_params.copy()
            
            # Perturb scale, rotation, translation
            new_params[0] += random.gauss(0, 0.1) # tx
            new_params[1] += random.gauss(0, 0.1) # ty
            new_params[2] += random.gauss(0, 5.0) # theta
            new_params[3] += random.gauss(0, 0.05) # scale
            
            # Keep scale reasonable
            new_params[3] = max(0.8, min(1.2, new_params[3]))
            
            new_score = evaluate_rigid(base_centers, base_angles, *new_params)
            
            delta_e = new_score - current_score
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                rigid_params = new_params
                current_score = new_score
                if current_score < local_best_score:
                    local_best_score = current_score
                    local_best_params = rigid_params.copy()
            
            T *= cooling_rate
        
        # Convert best rigid result to full parameters
        if local_best_score < best_score:
            best_score = local_best_score
            tx, ty, theta_deg, scale = local_best_params
            theta_rad = math.radians(theta_deg)
            cos_t = math.cos(theta_rad)
            sin_t = math.sin(theta_rad)
            
            full_params = []
            for i in range(N):
                bx, by = base_centers[i]
                rx = bx * scale
                ry = by * scale
                x = rx * cos_t - ry * sin_t + tx
                y = rx * sin_t + ry * cos_t + ty
                a = base_angles[i] + theta_deg
                full_params.extend([x, y, a])
            
            best_params = np.array(full_params)

    # If no rigid SA was run or failed, fallback to standard SA
    if best_params is None:
        # Run standard SA on one config
        config = configs[0]
        params = []
        for q, r in config:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            params.extend([x, y, 0.0])
        best_params = np.array(params)
        
        current_score = evaluate(params)
        local_best_params = params.copy()
        local_best_score = current_score
        
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.995
        
        while T > T_min and time.time() < deadline - 5.0:
            new_params = params.copy()
            idx = random.randint(0, N-1)
            dx = random.gauss(0, 0.3)
            dy = random.gauss(0, 0.3)
            da = random.gauss(0, 15.0)
            
            new_params[idx*3] += dx
            new_params[idx*3+1] += dy
            new_params[idx*3+2] += da
            
            new_score = evaluate(new_params)
            
            delta_e = new_score - current_score
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                params = new_params
                current_score = new_score
                if current_score < local_best_score:
                    local_best_score = current_score
                    local_best_params = params.copy()
            
            T *= cooling_rate
        
        best_params = local_best_params
        best_score = local_best_score

    # Polish with Nelder-Mead on the full parameter space
    # Stage 1: Smooth objective
    def objective_nm_smooth(p):
        return evaluate(p, outer_angle_deg=0.0, smooth=True)
        
    try:
        res1 = minimize(objective_nm_smooth, best_params, method='Nelder-Mead', 
                       options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-4})
        if res1.fun < best_score:
            best_score = res1.fun
            best_params = res1.x
    except Exception:
        pass

    # Stage 2: Hard objective (exact)
    def objective_nm_hard(p):
        return evaluate(p, outer_angle_deg=0.0, smooth=False)
        
    try:
        res2 = minimize(objective_nm_hard, best_params, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
        if res2.fun < best_score:
            best_score = res2.fun
            best_params = res2.x
    except Exception:
        pass

    # Final Validation and Outer Angle Optimization
    final_params = best_params.copy()
    
    centers = []
    angles = []
    for i in range(N):
        centers.append((final_params[i*3], final_params[i*3+1]))
        angles.append(final_params[i*3+2])
        
    cx_avg = sum(c[0] for c in centers) / N
    cy_avg = sum(c[1] for c in centers) / N
    shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
    
    hexagons = []
    hex_axes = []
    for i in range(N):
        cx, cy = shifted_centers[i]
        a = angles[i]
        hex_verts = get_hex_vertices(cx, cy, a)
        hexagons.append(hex_verts)
        hex_axes.append(get_hex_axes(a))
        
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            dx = shifted_centers[i][0] - shifted_centers[j][0]
            dy = shifted_centers[i][1] - shifted_centers[j][1]
            d_sq = dx*dx + dy*dy
            if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j], d_sq):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback
        inner_list = []
        for q, r in coords1:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            inner_list.append([x, y, 0.0])
        cx_avg_fb = sum(p[0] for p in inner_list) / N
        cy_avg_fb = sum(p[1] for p in inner_list) / N
        inner_list = [[x - cx_avg_fb, y - cy_avg_fb, a] for x, y, a in inner_list]
        hexagons_fb = [get_hex_vertices(p[0], p[1], p[2]) for p in inner_list]
        best_side = calc_outer_side(hexagons_fb, 0.0)
        return (inner_list, (0.0, 0.0), best_side, 0.0)

    # Optimize outer angle
    best_side = float('inf')
    best_oa = 0.0
    for oa_deg in range(0, 600):
        oa = oa_deg * 0.1
        side = calc_outer_side(hexagons, oa)
        if side < best_side:
            best_side = side
            best_oa = oa
            
    inner_list = []
    for i in range(N):
        inner_list.append([shifted_centers[i][0], shifted_centers[i][1], angles[i]])
        
    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
