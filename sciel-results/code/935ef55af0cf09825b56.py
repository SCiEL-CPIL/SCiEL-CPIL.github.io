"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (evaluator limit 150s, keep 10s buffer)
    
    SQRT3 = math.sqrt(3)
    
    # Rigid cluster: a compact 3-5-3 block of unit hexagons
    # Axial coordinates (q, r) for the 11 hexagons
    cluster_axial = [
        (-1, -1), (0, -1), (1, -1),
        (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1)
    ]
    
    # Convert to local Cartesian coordinates (centered at origin)
    def axial_to_cartesian(q, r):
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        return x, y
    
    local_coords = [axial_to_cartesian(q, r) for q, r in cluster_axial]
    # Center the cluster
    cx_avg = sum(c[0] for c in local_coords) / N
    cy_avg = sum(c[1] for c in local_coords) / N
    local_coords = [(x - cx_avg, y - cy_avg) for x, y in local_coords]
    
    # Precompute the 6 vertices of a unit hexagon at origin (angle 0)
    unit_hex_verts = []
    for i in range(6):
        theta = math.radians(i * 60)
        unit_hex_verts.append((math.cos(theta), math.sin(theta)))
    unit_hex_verts = np.array(unit_hex_verts)
    
    # Helper: Rotate and translate a point
    def rotate_point(x, y, angle_rad):
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        return x * c - y * s, x * s + y * c
    
    # Helper: Get vertices of a hexagon centered at (cx, cy) with rotation angle
    def get_hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for vx, vy in unit_hex_verts:
            rx, ry = rotate_point(vx, vy, angle_rad)
            verts.append((cx + rx, cy + ry))
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
    
    def is_disjoint(hex1_verts, hex2_verts, axes1, axes2):
        all_axes = axes1 + axes2
        for axis in all_axes:
            min1, max1 = project_poly(hex1_verts, axis)
            min2, max2 = project_poly(hex2_verts, axis)
            if max1 < min2 or max2 < min1:
                return True
        return False
    
    # Objective function: minimize outer side length
    # Parameters: [cx, cy, inner_angle_deg, outer_angle_deg]
    def objective(p):
        cx, cy, inner_angle, outer_angle = p
        
        # Compute all hexagon vertices and axes
        hexagons = []
        hex_axes = []
        for lx, ly in local_coords:
            # Rotate local coords by inner_angle, then translate by (cx, cy)
            rx, ry = rotate_point(lx, ly, math.radians(inner_angle))
            hx, hy = cx + rx, cy + ry
            hex_verts = get_hex_vertices(hx, hy, inner_angle)
            hexagons.append(hex_verts)
            hex_axes.append(get_hex_axes(inner_angle))
        
        # Check for overlaps (should be none for rigid cluster with unit spacing)
        # But we'll add a small penalty just in case
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                    overlap_penalty += 1000.0
        
        # Calculate minimum outer side length
        # Outer hexagon is centered at (0,0) with rotation outer_angle
        base_angles = [30, 90, 150, 210, 270, 330]
        axis_dirs = []
        for a in base_angles:
            rad = math.radians(a + outer_angle)
            axis_dirs.append((math.cos(rad), math.sin(rad)))
        
        max_proj = 0.0
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for nx, ny in axis_dirs:
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        
        side = 2.0 * max_proj / SQRT3
        return overlap_penalty + side
    
    # Global search: coarse grid over inner_angle and outer_angle
    best_params = None
    best_score = float('inf')
    
    # Coarse grid: inner_angle 0-60 deg step 5, outer_angle 0-60 deg step 5
    # cx, cy fixed at 0 for now (we'll optimize later)
    for ia_deg in range(0, 60, 5):
        if time.time() > deadline - 30.0:
            break
        for oa_deg in range(0, 60, 5):
            p = np.array([0.0, 0.0, float(ia_deg), float(oa_deg)])
            score = objective(p)
            if score < best_score:
                best_score = score
                best_params = p.copy()
    
    # Local refinement: Nelder-Mead on [cx, cy, inner_angle, outer_angle]
    # Multiple restarts with increasing tightness
    for beta in [100, 400, 1000]:
        if time.time() > deadline - 10.0:
            break
        
        # Small perturbations around best_params
        for restart in range(3):
            p0 = best_params.copy()
            if restart > 0:
                p0[0] += random.gauss(0, 0.1)
                p0[1] += random.gauss(0, 0.1)
                p0[2] += random.gauss(0, 2.0)
                p0[3] += random.gauss(0, 2.0)
            
            try:
                res = minimize(objective, p0, method='Nelder-Mead', 
                               options={'maxiter': beta, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x
            except Exception:
                pass
    
    # Extract final parameters
    cx, cy, inner_angle, outer_angle = best_params
    
    # Build final inner list
    inner_list = []
    for lx, ly in local_coords:
        rx, ry = rotate_point(lx, ly, math.radians(inner_angle))
        hx, hy = cx + rx, cy + ry
        inner_list.append([hx, hy, inner_angle])
    
    # Final validation
    hexagons = []
    hex_axes = []
    for hx, hy, a in inner_list:
        hex_verts = get_hex_vertices(hx, hy, a)
        hexagons.append(hex_verts)
        hex_axes.append(get_hex_axes(a))
    
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: use initial cluster at origin
        inner_list = []
        for lx, ly in local_coords:
            inner_list.append([lx, ly, 0.0])
        # Recalculate side
        hexagons_fb = [get_hex_vertices(x, y, 0) for x, y, a in inner_list]
        max_proj = 0.0
        for hex_verts in hexagons_fb:
            for vx, vy in hex_verts:
                for nx, ny in [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in [30, 90, 150, 210, 270, 330]]:
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        best_side = 2.0 * max_proj / SQRT3
        return (inner_list, (0.0, 0.0), best_side, 0.0)
    
    # Calculate final side length
    max_proj = 0.0
    for hex_verts in hexagons:
        for vx, vy in hex_verts:
            for a in [30, 90, 150, 210, 270, 330]:
                rad = math.radians(a + outer_angle)
                nx, ny = math.cos(rad), math.sin(rad)
                proj = abs(vx * nx + vy * ny)
                if proj > max_proj:
                    max_proj = proj
    best_side = 2.0 * max_proj / SQRT3
    
    return (inner_list, (0.0, 0.0), best_side, outer_angle)

# EVOLVE-BLOCK-END
