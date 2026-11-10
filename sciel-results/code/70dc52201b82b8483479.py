import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (evaluator limit 150s)
    
    def get_hex_vertices(cx, cy, angle_deg):
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

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
        if d_sq > 4.0:
            return True
        if d_sq < 3.0:
            return False
        all_axes = axes1 + axes2
        for axis in all_axes:
            min1, max1 = project_poly(hex1_verts, axis)
            min2, max2 = project_poly(hex2_verts, axis)
            if max1 < min2 or max2 < min1:
                return True
        return False

    def calc_outer_side(hexagons, outer_angle_deg):
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
        
        return 2.0 * max_proj / SQRT3

    def evaluate_full(centers, angles, outer_angle_deg=0.0):
        """
        Full evaluation of a configuration.
        centers: list of (x, y)
        angles: list of angle_degrees
        """
        # Center the configuration
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
        return overlap_penalty + side

    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # --- STRATEGY: Template-Based Search ---
    # We define specific topological structures (templates) for the 11 hexagons.
    # Each template defines relative positions and contact constraints.
    # We then solve for the exact positions that satisfy these constraints
    # while minimizing the bounding hexagon size.

    # Template 1: 3-4-3-1 (Compact cluster)
    # Rows of 3, 4, 3, and 1 hexagon on top.
    # This is a very dense packing.
    # Coordinates in axial system (q, r)
    # Row 0 (bottom): 3 hexes
    # Row 1: 4 hexes
    # Row 2: 3 hexes
    # Row 3: 1 hex
    # Note: Standard hex grid axial coords.
    
    templates = []
    
    # T1: "Pyramid" 3-4-3-1
    # Bottom row y=0: q=-1,0,1
    # Next row y=1: q=-1.5, -0.5, 0.5, 1.5 (approx in cartesian, but let's use axial logic)
    # Actually, let's use Cartesian coordinates for templates to be explicit.
    # Unit hexagon side 1. Distance between centers of touching hexes is SQRT3.
    
    def make_template_3431():
        # Center the structure at (0,0)
        # Row 0 (y=0): 3 hexes at x = -SQRT3, 0, SQRT3
        # Row 1 (y=1.5): 4 hexes at x = -1.5*SQRT3, -0.5*SQRT3, 0.5*SQRT3, 1.5*SQRT3
        # Row 2 (y=3.0): 3 hexes at x = -SQRT3, 0, SQRT3
        # Row 3 (y=4.5): 1 hex at x = 0
        
        pts = []
        # Row 0
        for i in range(3):
            x = (i - 1) * SQRT3
            pts.append((x, 0.0))
        # Row 1
        for i in range(4):
            x = (i - 1.5) * SQRT3
            pts.append((x, 1.5))
        # Row 2
        for i in range(3):
            x = (i - 1) * SQRT3
            pts.append((x, 3.0))
        # Row 3
        pts.append((0.0, 4.5))
        
        # Center it
        cx = sum(p[0] for p in pts) / N
        cy = sum(p[1] for p in pts) / N
        pts = [(x-cx, y-cy) for x, y in pts]
        return pts

    def make_template_23321():
        # Row 0: 2
        # Row 1: 3
        # Row 2: 3
        # Row 3: 2
        # Row 4: 1
        pts = []
        # Row 0
        for i in range(2):
            x = (i - 0.5) * SQRT3
            pts.append((x, 0.0))
        # Row 1
        for i in range(3):
            x = (i - 1) * SQRT3
            pts.append((x, 1.5))
        # Row 2
        for i in range(3):
            x = (i - 1) * SQRT3
            pts.append((x, 3.0))
        # Row 3
        for i in range(2):
            x = (i - 0.5) * SQRT3
            pts.append((x, 4.5))
        # Row 4
        pts.append((0.0, 6.0))
        
        cx = sum(p[0] for p in pts) / N
        cy = sum(p[1] for p in pts) / N
        pts = [(x-cx, y-cy) for x, y in pts]
        return pts

    def make_template_flower_ext():
        # Central flower (7) + 4 extensions
        # Center (0,0)
        # Ring 1: 6 hexes
        # Extensions: 4 hexes attached to the "corners" of the ring?
        # Let's try a different shape: 1 in center, 6 around, 4 more attached to the "sides"
        # This is harder to define statically. Let's use a "brick wall" pattern.
        # Row 0: 3
        # Row 1: 4
        # Row 2: 4
        pts = []
        for i in range(3):
            x = (i - 1) * SQRT3
            pts.append((x, 0.0))
        for i in range(4):
            x = (i - 1.5) * SQRT3
            pts.append((x, 1.5))
        for i in range(4):
            x = (i - 1.5) * SQRT3
            pts.append((x, 3.0))
            
        cx = sum(p[0] for p in pts) / N
        cy = sum(p[1] for p in pts) / N
        pts = [(x-cx, y-cy) for x, y in pts]
        return pts

    templates = [make_template_3431(), make_template_23321(), make_template_flower_ext()]
    
    best_score = float('inf')
    best_centers = None
    best_angles = None
    best_outer_angle = 0.0

    # For each template, we perform a local optimization.
    # Variables:
    # - Global rotation of the whole cluster (1 var)
    # - Individual rotations of each hexagon (N vars)
    # - Small perturbations to the template positions (2*N vars) to allow "breathing"
    # Total: 1 + N + 2N = 3N + 1 = 34 vars.
    
    # However, a simpler and more robust approach:
    # 1. Take the template.
    # 2. Optimize individual angles and global rotation.
    # 3. Use Nelder-Mead on the full set of positions (2N) + angles (N) + outer rotation (1).
    #    To prevent drifting away from the template (which might lead to a worse local min),
    #    we add a small penalty term that pulls centers back to the template positions.
    
    def objective_with_anchor(p, template_pts):
        # p: [x0, y0, a0, x1, y1, a1, ..., x10, y10, a10, outer_angle]
        centers = []
        angles = []
        for i in range(N):
            centers.append((p[i*3], p[i*3+1]))
            angles.append(p[i*3+2])
        outer_angle = p[N*3]
        
        score = evaluate_full(centers, angles, outer_angle)
        
        # Anchor penalty: keep centers close to template
        anchor_penalty = 0.0
        for i in range(N):
            dx = centers[i][0] - template_pts[i][0]
            dy = centers[i][1] - template_pts[i][1]
            anchor_penalty += (dx*dx + dy*dy) * 0.1 # Small weight
            
        return score + anchor_penalty

    for template_pts in templates:
        if time.time() > deadline - 10.0:
            break
            
        # Initialize parameters from template
        init_params = []
        for x, y in template_pts:
            init_params.extend([x, y, 0.0])
        init_params.append(0.0) # outer angle
        
        # Run Nelder-Mead
        try:
            res = minimize(objective_with_anchor, init_params, args=(template_pts,), 
                           method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False})
            
            if res.fun < best_score:
                # Remove anchor penalty for final score
                centers = []
                angles = []
                for i in range(N):
                    centers.append((res.x[i*3], res.x[i*3+1]))
                    angles.append(res.x[i*3+2])
                outer_angle = res.x[N*3]
                
                final_score = evaluate_full(centers, angles, outer_angle)
                
                # Check validity (no overlap)
                valid = True
                hexagons = [get_hex_vertices(c[0], c[1], a) for c, a in zip(centers, angles)]
                hex_axes = [get_hex_axes(a) for a in angles]
                for i in range(N):
                    for j in range(i+1, N):
                        dx = centers[i][0] - centers[j][0]
                        dy = centers[i][1] - centers[j][1]
                        d_sq = dx*dx + dy*dy
                        if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j], d_sq):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    best_score = final_score
                    best_centers = centers
                    best_angles = angles
                    best_outer_angle = outer_angle
                    
        except Exception:
            pass

    # Fallback if templates failed or didn't find a valid solution
    if best_centers is None:
        # Use the first template as a valid fallback (it's geometrically valid if not optimized)
        template_pts = templates[0]
        best_centers = [(x, y) for x, y in template_pts]
        best_angles = [0.0] * N
        best_outer_angle = 0.0
        best_score = evaluate_full(best_centers, best_angles, best_outer_angle)

    # Final Polish: Optimize outer angle precisely for the best configuration
    # and do a very tight Nelder-Mead on just the outer angle and small global rotation
    
    if time.time() < deadline - 2.0:
        # Re-center best_centers for final output
        cx_avg = sum(c[0] for c in best_centers) / N
        cy_avg = sum(c[1] for c in best_centers) / N
        best_centers = [(x - cx_avg, y - cy_avg) for x, y in best_centers]
        
        # Optimize outer angle
        best_side = float('inf')
        best_oa = best_outer_angle
        for oa_deg in range(-300, 301, 2):
            oa = oa_deg * 0.1
            side = calc_outer_side([get_hex_vertices(c[0], c[1], a) for c, a in zip(best_centers, best_angles)], oa)
            if side < best_side:
                best_side = side
                best_oa = oa
                
        # Fine tune outer angle
        for oa_deg in range(-100, 101, 1):
            oa = best_oa + oa_deg * 0.01
            side = calc_outer_side([get_hex_vertices(c[0], c[1], a) for c, a in zip(best_centers, best_angles)], oa)
            if side < best_side:
                best_side = side
                best_oa = oa

        # Final check
        hexagons = [get_hex_vertices(c[0], c[1], a) for c, a in zip(best_centers, best_angles)]
        hex_axes = [get_hex_axes(a) for a in best_angles]
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                dx = best_centers[i][0] - best_centers[j][0]
                dy = best_centers[i][1] - best_centers[j][1]
                d_sq = dx*dx + dy*dy
                if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j], d_sq):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to known valid config (template 0)
            template_pts = templates[0]
            cx_avg = sum(p[0] for p in template_pts) / N
            cy_avg = sum(p[1] for p in template_pts) / N
            best_centers = [(x - cx_avg, y - cy_avg) for x, y in template_pts]
            best_angles = [0.0] * N
            best_side = calc_outer_side([get_hex_vertices(c[0], c[1], a) for c, a in zip(best_centers, best_angles)], 0.0)
            best_oa = 0.0

    inner_list = []
    for i in range(N):
        inner_list.append([best_centers[i][0], best_centers[i][1], best_angles[i]])
        
    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
