"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit

    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    def get_gap(v1, v2):
        """
        Returns the maximum gap between two hexagons using SAT.
        If gap > 0, they are separated.
        If gap < 0, they overlap.
        """
        max_gap = -float('inf')
        for i in range(3):
            e1 = v1[(i+1)%6] - v1[i]
            n1 = np.array([-e1[1], e1[0]])
            norm1 = np.linalg.norm(n1)
            if norm1 > 1e-9:
                n1 = n1 / norm1
            
            e2 = v2[(i+1)%6] - v2[i]
            n2 = np.array([-e2[1], e2[0]])
            norm2 = np.linalg.norm(n2)
            if norm2 > 1e-9:
                n2 = n2 / norm2
            
            for n in [n1, n2]:
                proj1 = np.dot(v1, n)
                proj2 = np.dot(v2, n)
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                if max1 < min2:
                    gap = min2 - max1
                elif max2 < min1:
                    gap = min1 - max2
                else:
                    overlap_amount = min(max1, max2) - max(min1, min2)
                    gap = -overlap_amount
                    
                if gap > max_gap:
                    max_gap = gap
                    
        return max_gap

    def compute_outer_side(inner_centers, inner_angles, outer_cx, outer_cy, outer_angle_deg):
        """
        Computes the minimum side length of the outer hexagon centered at (outer_cx, outer_cy)
        with rotation outer_angle_deg to contain all inner hexagons.
        """
        outer_center = np.array([outer_cx, outer_cy])
        outer_angle_rad = math.radians(outer_angle_deg)
        
        max_proj = 0.0
        for i in range(N):
            cx, cy = inner_centers[i]
            ang = inner_angles[i]
            verts = hex_vertices(cx, cy, ang)
            rel_verts = verts - outer_center
            for k in range(6):
                normal_angle = outer_angle_rad + k * math.pi / 3
                normal = np.array([math.cos(normal_angle), math.sin(normal_angle)])
                proj = np.dot(rel_verts, normal)
                max_proj = max(max_proj, np.max(proj))
        
        return 2.0 * max_proj / math.sqrt(3)

    def is_valid(inner_centers, inner_angles):
        """Check if configuration is valid (no overlaps)."""
        verts_list = [hex_vertices(inner_centers[i][0], inner_centers[i][1], inner_angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                gap = get_gap(verts_list[i], verts_list[j])
                if gap < -1e-5:
                    return False
        return True

    def objective(params):
        # Unpack parameters
        # 0..35: inner centers (x, y) and angles for 12 hexagons (3 params each)
        # 36: outer cx
        # 37: outer cy
        # 38: outer angle (degrees)
        
        inner_centers = []
        inner_angles = []
        for i in range(N):
            cx = params[3*i]
            cy = params[3*i + 1]
            ang = params[3*i + 2]
            inner_centers.append((cx, cy))
            inner_angles.append(ang)
        
        outer_cx = params[36]
        outer_cy = params[37]
        outer_angle_deg = params[38]
        
        # Calculate side length
        side = compute_outer_side(inner_centers, inner_angles, outer_cx, outer_cy, outer_angle_deg)
        
        # Calculate overlap penalty
        penalty = 0.0
        verts_list = [hex_vertices(inner_centers[i][0], inner_centers[i][1], inner_angles[i]) for i in range(N)]
        
        for i in range(N):
            for j in range(i+1, N):
                gap = get_gap(verts_list[i], verts_list[j])
                if gap < 0:
                    penalty += (-gap) ** 2
        
        # Weight the penalty heavily
        return side + 1000.0 * penalty

    def objective_with_grad(params):
        """
        Objective function with analytic gradient for scipy optimization.
        Uses smooth approximation of the min/max in the objective.
        """
        inner_centers = []
        inner_angles = []
        for i in range(N):
            cx = params[3*i]
            cy = params[3*i + 1]
            ang = params[3*i + 2]
            inner_centers.append((cx, cy))
            inner_angles.append(ang)
        
        outer_cx = params[36]
        outer_cy = params[37]
        outer_angle_deg = params[38]
        
        # Compute side length with smooth max
        outer_center = np.array([outer_cx, outer_cy])
        outer_angle_rad = math.radians(outer_angle_deg)
        
        # Collect all projections
        all_projs = []
        for i in range(N):
            cx, cy = inner_centers[i]
            ang = inner_angles[i]
            verts = hex_vertices(cx, cy, ang)
            rel_verts = verts - outer_center
            for k in range(6):
                normal_angle = outer_angle_rad + k * math.pi / 3
                normal = np.array([math.cos(normal_angle), math.sin(normal_angle)])
                proj = np.dot(rel_verts, normal)
                all_projs.append(np.max(proj))
        
        all_projs = np.array(all_projs)
        
        # Smooth max using log-sum-exp
        beta = 50.0  # Temperature for soft-max
        smooth_max = (1.0 / beta) * np.log(np.sum(np.exp(beta * all_projs)))
        
        side = 2.0 * smooth_max / math.sqrt(3)
        
        # Overlap penalty (smooth)
        penalty = 0.0
        verts_list = [hex_vertices(inner_centers[i][0], inner_centers[i][1], inner_angles[i]) for i in range(N)]
        
        for i in range(N):
            for j in range(i+1, N):
                gap = get_gap(verts_list[i], verts_list[j])
                if gap < 0:
                    penalty += (-gap) ** 2
        
        return side + 1000.0 * penalty

    # Initial configuration: Staggered grid of 12 hexagons
    s = 1.0
    dx = 2.0 * s
    dy = math.sqrt(3) * s
    offset = s
    
    # Create a 4x3 grid (4 rows, 3 cols)
    centers = []
    for r in range(4):
        for c in range(3):
            x = c * dx
            y = r * dy
            if r % 2 == 1:
                x += offset
            centers.append((x, y))
            
    # Center the grid at (0,0)
    cx_mean = sum(c[0] for c in centers) / N
    cy_mean = sum(c[1] for c in centers) / N
    adjusted_centers = [(x - cx_mean, y - cy_mean) for x, y in centers]
    
    # Initialize parameters
    x0 = np.zeros(39)
    for i in range(N):
        x0[3*i] = adjusted_centers[i][0]
        x0[3*i+1] = adjusted_centers[i][1]
        x0[3*i+2] = 0.0 # Initial angle 0
        
    x0[36] = 0.0
    x0[37] = 0.0
    x0[38] = 0.0
    
    best_side = float('inf')
    best_params = x0.copy()
    best_valid = False

    # Simulated annealing with restarts
    def simulated_annealing(x0, max_iter=50000, T0=1.0, T_min=0.001, alpha=0.9999):
        """
        Simulated annealing optimization.
        """
        current_x = x0.copy()
        current_obj = objective(current_x)
        best_x = current_x.copy()
        best_obj = current_obj
        best_valid = is_valid(
            [(current_x[3*i], current_x[3*i+1]) for i in range(N)],
            [current_x[3*i+2] for i in range(N)]
        )
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Generate neighbor by perturbing random parameters
            new_x = current_x.copy()
            num_perturb = np.random.randint(1, 4)
            indices = np.random.choice(len(new_x), num_perturb, replace=False)
            for idx in indices:
                new_x[idx] += np.random.normal(0, 0.1)
            
            new_obj = objective(new_x)
            new_valid = is_valid(
                [(new_x[3*i], new_x[3*i+1]) for i in range(N)],
                [new_x[3*i+2] for i in range(N)]
            )
            
            # Update temperature
            T = T0 * (alpha ** it)
            
            # Acceptance criterion
            delta = new_obj - current_obj
            if delta < 0 or np.random.random() < math.exp(-delta / T):
                current_x = new_x
                current_obj = new_obj
                
                if new_valid and new_obj < best_obj:
                    best_x = new_x.copy()
                    best_obj = new_obj
                    best_valid = True
        
        return best_x, best_obj, best_valid

    # Run multiple restarts with simulated annealing
    for restart in range(3):
        if time.time() > deadline:
            break
        
        # For first restart, use initial grid; for others, perturb
        if restart == 0:
            x_start = x0.copy()
        else:
            x_start = x0.copy()
            # Add some random perturbation
            x_start += np.random.normal(0, 0.3, size=len(x_start))
        
        try:
            res_x, res_obj, res_valid = simulated_annealing(x_start, max_iter=10000)
            
            if res_valid and res_obj < best_side:
                best_side = res_obj
                best_params = res_x
                best_valid = True
        except Exception:
            pass

    # If we have a valid result, try to polish it with scipy
    if best_valid and time.time() < deadline - 30:
        try:
            # Use scipy's SLSQP to refine the solution
            result = minimize(
                objective_with_grad,
                best_params,
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
            )
            
            if result.success:
                refined_params = result.x
                refined_inner_centers = [(refined_params[3*i], refined_params[3*i+1]) for i in range(N)]
                refined_inner_angles = [refined_params[3*i+2] for i in range(N)]
                
                if is_valid(refined_inner_centers, refined_inner_angles):
                    refined_side = objective(refined_params)
                    if refined_side < best_side:
                        best_side = refined_side
                        best_params = refined_params
                        best_valid = True
        except Exception:
            pass

    # If no valid result found, try the initial grid as fallback
    if not best_valid:
        inner_centers = adjusted_centers
        inner_angles = [0.0] * N
        outer_cx = 0.0
        outer_cy = 0.0
        outer_angle_deg = 0.0
        
        side = compute_outer_side(inner_centers, inner_angles, outer_cx, outer_cy, outer_angle_deg)
        
        # Check validity of this fallback
        valid = is_valid(inner_centers, inner_angles)
        
        if valid:
            best_side = side
            inner_list = []
            for c, a in zip(inner_centers, inner_angles):
                inner_list.append([float(c[0]), float(c[1]), float(a)])
            return inner_list, (float(outer_cx), float(outer_cy)), float(best_side * 1.01), float(outer_angle_deg)
        else:
            # Absolute fallback: huge box
            inner_list = [[0.0, 0.0, 0.0] for _ in range(N)]
            return inner_list, (0.0, 0.0), 100.0, 0.0

    # Construct return value from best_params
    inner = []
    for i in range(N):
        cx = best_params[3*i]
        cy = best_params[3*i+1]
        ang_deg = best_params[3*i+2]
        inner.append([float(cx), float(cy), float(ang_deg)])
        
    outer_cx = best_params[36]
    outer_cy = best_params[37]
    outer_angle_deg = best_params[38]
    
    # Add safety margin
    final_side = best_side * 1.01 + 1e-4
    
    return inner, (float(outer_cx), float(outer_cy)), float(final_side), float(outer_angle_deg)
# EVOLVE-BLOCK-END
