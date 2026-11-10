import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (evaluator limit 150s)
    
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

    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Initial configurations (lattice-based)
    coords1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
    coords2 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
    coords3 = [(0,0), (1,0), (2,0), (3,0), (4,0), (-1,0), (-2,0), (0,1), (1,1), (2,1), (3,1)]
    coords4 = [(-1,-1),(0,-1),(1,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1),(2,0),(2,1)]
    coords5 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,1), (-2,1), (-2,0)]
    
    configs = [coords1, coords2, coords3, coords4, coords5]
    
    best_params = None
    best_score = float('inf')
    
    def evaluate_full(p, outer_angle_deg):
        """
        Evaluate a full parameter set. Returns (score, valid, side_length).
        p: [x1, y1, a1, x2, y2, a2, ...]
        """
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
        
        side = calc_outer_side(hexagons, outer_angle_deg)
        return overlap_penalty + side, overlap_penalty == 0.0, side

    def relax_configuration(base_centers, base_angles, outer_angle_deg, iterations=2000, dt=0.05, k_repel=10.0, k_attract=0.5):
        """
        Force-directed relaxation.
        Hexagons repel if overlapping (based on center distance approximation for speed, refined by SAT check).
        Hexagons are attracted to center to minimize bounding box.
        """
        # State: centers (N, 2), angles (N,)
        centers = np.array(base_centers, dtype=float)
        angles = np.array(base_angles, dtype=float)
        
        # Re-center
        cx_avg = np.mean(centers[:, 0])
        cy_avg = np.mean(centers[:, 1])
        centers -= np.array([cx_avg, cy_avg])
        
        current_score, is_valid, side = evaluate_full(
            np.concatenate([centers[:, 0], centers[:, 1], angles]).tolist(), 
            outer_angle_deg
        )
        
        for it in range(iterations):
            if time.time() > deadline - 10.0:
                break
                
            forces = np.zeros_like(centers)
            torques = np.zeros_like(angles)
            
            # 1. Repulsion between overlapping hexagons
            # Approximate overlap by center distance < 2*apothem? No, min distance is sqrt(3).
            # Use d < 2.0 as a trigger for repulsion calculation to save time.
            for i in range(N):
                for j in range(i+1, N):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    d_sq = dx*dx + dy*dy
                    
                    # If definitely disjoint, skip
                    if d_sq > 4.0: 
                        continue
                    
                    # Check actual disjointness
                    hex_i_verts = get_hex_vertices(centers[i, 0], centers[i, 1], angles[i])
                    hex_j_verts = get_hex_vertices(centers[j, 0], centers[j, 1], angles[j])
                    axes_i = get_hex_axes(angles[i])
                    axes_j = get_hex_axes(angles[j])
                    
                    if is_disjoint(hex_i_verts, hex_j_verts, axes_i, axes_j, d_sq):
                        continue
                    
                    # Overlapping: apply repulsive force
                    d = math.sqrt(d_sq)
                    if d < 1e-6:
                        # Random nudge
                        fx = random.gauss(0, 1)
                        fy = random.gauss(0, 1)
                    else:
                        fx = dx / d
                        fy = dy / d
                    
                    # Force magnitude proportional to overlap depth (approximated by 2.0 - d)
                    # Actually, just a constant strong force is often stable enough for relaxation
                    f_mag = k_repel * (2.0 - d) 
                    if f_mag < 0: f_mag = k_repel # Safety
                    
                    forces[i, 0] += fx * f_mag
                    forces[i, 1] += fy * f_mag
                    forces[j, 0] -= fx * f_mag
                    forces[j, 1] -= fy * f_mag
                    
                    # Torque: try to rotate to minimize overlap? 
                    # Simple heuristic: rotate towards alignment with vector between centers
                    # This is complex. Instead, let's just apply a small random torque or 
                    # torque based on "friction" to dampen rotation.
                    # For now, keep angles fixed during relaxation unless we add a specific torque model.
                    # Let's add a small torque to align edges:
                    # The optimal orientation for two hexes to be disjoint often involves aligning parallel edges.
                    # We can approximate the "best" relative angle.
                    # For simplicity, we will NOT change angles in this relaxation phase, 
                    # but rely on the initial angles being 0 or optimized later.
                    # However, if angles are fixed, we might get stuck. 
                    # Let's add a small random torque to escape local minima in rotation space.
                    torques[i] += random.gauss(0, 0.1)
                    torques[j] += random.gauss(0, 0.1)

            # 2. Attraction to center (to minimize bounding box)
            # The "pressure" from the container walls.
            # Approximate by pulling centers towards (0,0)
            for i in range(N):
                forces[i, 0] -= centers[i, 0] * k_attract
                forces[i, 1] -= centers[i, 1] * k_attract
            
            # Update positions
            centers += forces * dt
            angles += torques * dt
            
            # Re-center after update to prevent drift
            cx_avg = np.mean(centers[:, 0])
            cy_avg = np.mean(centers[:, 1])
            centers -= np.array([cx_avg, cy_avg])
            
            # Evaluate
            p_flat = np.concatenate([centers[:, 0], centers[:, 1], angles])
            new_score, new_valid, new_side = evaluate_full(p_flat.tolist(), outer_angle_deg)
            
            if new_score < current_score:
                current_score = new_score
                is_valid = new_valid
                side = new_side
                # If we found a valid solution, we can stop or continue to polish
                if is_valid and current_score < 2.5: # Heuristic threshold for "good"
                    # Optional: break early if we have a very good valid solution
                    pass
            
            # Adaptive step size? Keep it simple.
            
        return centers.tolist(), angles.tolist(), current_score, is_valid, side

    # --- Main Loop: Grid Search over Outer Angle + Relaxation ---
    
    # We will test a range of outer angles. 
    # The optimal outer angle is likely aligned with the symmetry of the packing.
    # Let's test 0 to 60 degrees in steps of 5 degrees, and refine the best one.
    
    for oa_deg in range(0, 61, 5):
        if time.time() > deadline - 30.0:
            break
            
        for config in configs:
            if time.time() > deadline - 15.0:
                break
                
            base_centers = []
            base_angles = []
            for q, r in config:
                x, y = axial_to_cartesian(q, r, spacing=1.0)
                base_centers.append((x, y))
                base_angles.append(0.0)
            
            # Relax
            res_centers, res_angles, score, valid, side = relax_configuration(
                base_centers, base_angles, oa_deg, iterations=1500
            )
            
            if valid and score < best_score:
                best_score = score
                best_params = np.concatenate([
                    np.array(res_centers)[:, 0], 
                    np.array(res_centers)[:, 1], 
                    np.array(res_angles)
                ])
                best_oa = oa_deg

    # If no valid solution found via relaxation, fallback to a known valid lattice
    if best_params is None:
        config = configs[0]
        params = []
        for q, r in config:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            params.extend([x, y, 0.0])
        best_params = np.array(params)
        best_oa = 0.0
        # Calculate score for fallback
        centers = [(best_params[i*3], best_params[i*3+1]) for i in range(N)]
        angles = [best_params[i*3+2] for i in range(N)]
        cx_avg = sum(c[0] for c in centers) / N
        cy_avg = sum(c[1] for c in centers) / N
        shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
        hexagons = [get_hex_vertices(sc[0], sc[1], a) for sc, a in zip(shifted_centers, angles)]
        best_score = calc_outer_side(hexagons, best_oa)

    # --- Polish Phase: Nelder-Mead on the best found configuration ---
    # We optimize the inner positions/angles AND the outer angle simultaneously?
    # The interface asks for separate return values, but we can optimize the inner
    # relative to a fixed outer angle found above, or jointly.
    # Let's fix the outer angle to the best found, and polish the inner.
    
    final_oa = best_oa
    
    def objective_nm(p):
        # p: [x1, y1, a1, ...]
        # We want to minimize the side length, penalizing overlaps heavily.
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
        
        side = calc_outer_side(hexagons, final_oa)
        return overlap_penalty + side

    try:
        res = minimize(objective_nm, best_params, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass

    # Final Validation and Outer Angle Refinement
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
        # Fallback to the initial lattice which is always valid
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

    # Refine outer angle with fine grid search
    best_side = float('inf')
    best_oa_final = final_oa
    # Search around the best found angle
    search_range = range(-10, 11) # +/- 10 degrees
    for delta in search_range:
        oa = final_oa + delta
        side = calc_outer_side(hexagons, oa)
        if side < best_side:
            best_side = side
            best_oa_final = oa
            
    # Even finer search around best_oa_final
    for delta in np.linspace(-2, 2, 21):
        oa = best_oa_final + delta
        side = calc_outer_side(hexagons, oa)
        if side < best_side:
            best_side = side
            best_oa_final = oa

    inner_list = []
    for i in range(N):
        inner_list.append([shifted_centers[i][0], shifted_centers[i][1], angles[i]])
        
    return (inner_list, (0.0, 0.0), best_side, best_oa_final)

# EVOLVE-BLOCK-END
