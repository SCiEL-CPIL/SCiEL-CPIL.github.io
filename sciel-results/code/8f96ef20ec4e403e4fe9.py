"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    angles = angle_rad + np.pi / 3 * np.arange(6)
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def compute_side_length_smooth(inner, outer_center, outer_angle_rad, beta=50.0):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    Uses log-sum-exp for a smooth approximation of the max projection.
    
    inner: list of (x, y, angle_rad)
    outer_center: (x, y)
    outer_angle_rad: rotation of the outer hexagon
    """
    ox, oy = outer_center
    # Normals for a standard hexagon (vertices at 0, 60, ...):
    # Outward normals are at 30, 90, 150, 210, 270, 330 degrees.
    normal_angles = np.radians(30) + np.pi / 3 * np.arange(6)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Rotation matrix for outer frame
    cos_a = np.cos(outer_angle_rad)
    sin_a = np.sin(outer_angle_rad)
    
    # We want to project points onto the rotated normals.
    # Equivalent to rotating points by -outer_angle and projecting onto standard normals.
    
    max_proj = 0.0
    # To use log-sum-exp efficiently, we gather all projections
    # But for N=11, 66 points, 6 normals -> 396 projections. 
    # We can compute the max for each inner hexagon first? No, the max is over ALL points.
    
    # Let's compute the vector of all projections
    projs = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, iangle)
        # Translate to outer center
        dx = verts[:, 0] - ox
        dy = verts[:, 1] - oy
        
        # Rotate points by -outer_angle
        # x' = x cos a + y sin a
        # y' = -x sin a + y cos a
        lx = dx * cos_a + dy * sin_a
        ly = -dx * sin_a + dy * cos_a
        
        # Project onto standard normals
        # proj = lx * nx + ly * ny
        # normals is 6x2
        # We want (6x2) @ (2x66) -> 6x66 matrix of projections
        # Actually, let's just iterate for clarity and speed in Python loop is okay for 11 hexes.
        for k in range(6):
            nx, ny = normals[k]
            p = lx * nx + ly * ny
            projs.extend(p)
            
    projs = np.array(projs)
    
    # Smooth max: (1/beta) * log(sum(exp(beta * p)))
    # Shift by max for numerical stability
    p_max = np.max(projs)
    if p_max < -1000:
        return 0.0
    
    # Clip to avoid overflow in exp
    scaled = beta * (projs - p_max)
    # Sum exp
    sum_exp = np.sum(np.exp(scaled))
    smooth_max = p_max + np.log(sum_exp) / beta
    
    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * smooth_max / SQRT3

def compute_side_length_exact(inner, outer_center, outer_angle_rad):
    """Compute exact minimum outer side length."""
    ox, oy = outer_center
    normal_angles = np.radians(30) + np.pi / 3 * np.arange(6)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    cos_a = np.cos(outer_angle_rad)
    sin_a = np.sin(outer_angle_rad)
    
    max_proj = 0.0
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, iangle)
        dx = verts[:, 0] - ox
        dy = verts[:, 1] - oy
        
        lx = dx * cos_a + dy * sin_a
        ly = -dx * sin_a + dy * cos_a
        
        for k in range(6):
            nx, ny = normals[k]
            p = np.dot(lx, nx) + np.dot(ly, ny)
            if p > max_proj:
                max_proj = p
                
    return 2.0 * max_proj / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons overlap using SAT.
    v1, v2: Nx2 arrays.
    Returns True if they overlap (including touching).
    """
    n1 = len(v1)
    n2 = len(v2)
    
    # Get axes from edges of v1
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-9:
            normal = normal / norm
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            if proj1.max() < proj2.min() + eps or proj2.max() < proj1.min() + eps:
                return False
                
    # Get axes from edges of v2
    for i in range(n2):
        p1 = v2[i]
        p2 = v2[(i + 1) % n2]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-9:
            normal = normal / norm
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            if proj1.max() < proj2.min() + eps or proj2.max() < proj1.min() + eps:
                return False
                
    return True

def check_validity(inner, outer_center, outer_side, outer_angle_rad):
    """Check if configuration is valid: no overlaps, all inside outer hexagon."""
    ox, oy = outer_center
    
    # 1. Check overlaps
    verts_list = []
    for (ix, iy, iangle) in inner:
        v = get_hex_vertices(ix, iy, 1.0, iangle)
        verts_list.append(v)
        
    for i in range(len(inner)):
        for j in range(i + 1, len(inner)):
            if sat_overlap_check(verts_list[i], verts_list[j], eps=1e-8):
                return False
                
    # 2. Check containment
    # A point is inside a convex polygon if it is on the same side of all edges.
    # For a regular hexagon centered at (ox,oy) with side S and angle theta:
    # The distance from center to any side is S * sqrt(3) / 2.
    # The inward normal for each side points towards the center.
    # Let's use the projection method:
    # For each side of the outer hexagon, the projection of any inner point onto the 
    # outward normal must be <= S * sqrt(3) / 2.
    
    apothem = outer_side * SQRT3 / 2.0
    normal_angles = np.radians(30) + np.pi / 3 * np.arange(6)
    # Rotate normals by outer angle
    normal_angles += outer_angle_rad
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    for v in verts_list:
        # Translate to outer center
        dx = v[:, 0] - ox
        dy = v[:, 1] - oy
        for k in range(6):
            n = normals[k]
            proj = np.dot(dx, n[0]) + np.dot(dy, n[1])
            if proj > apothem + 1e-8:
                return False
                
    return True

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    # 1. Define initial hexagonal lattice positions for N=11
    # Axial coordinates (q, r)
    # Center: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2: We need 4 more. The "corners" of the hexagon are at (2,-1), (1,1), (-1,2), (-2,1), (-1,-1), (1,-2).
    # Let's pick 4 that maintain symmetry or compactness.
    # Standard "11-hexagon" cluster is often center + ring 1 + 4 from ring 2.
    # To be symmetric, we might pick 3? No, 11 is not 3k+1. 
    # 1, 7, 19... 
    # 11 = 1 + 6 + 4.
    # Let's use a specific set that is known to be compact.
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-1, 2), (-2, 1) # 4 from ring 2
    ]
    
    # Convert axial to Cartesian
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    centers = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        centers.append((x, y))
        
    # Initial inner: all angles 0
    initial_inner = [[x, y, 0.0] for x, y in centers]
    
    best_side = float('inf')
    best_inner = initial_inner
    best_outer_center = (0.0, 0.0)
    best_outer_angle_rad = 0.0

    # Centroid of initial inner
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)

    # Phase 1: Optimize Outer Hexagon (Center and Angle) with fixed Inner
    # This reduces dimensionality to 3 and avoids the complex interaction of rotating inner hexes
    # which often leads to local minima in full joint optimization.
    
    def objective_outer(params):
        ox, oy, ang_rad = params
        # Use smooth side length for gradient
        s = compute_side_length_smooth(initial_inner, (ox, oy), ang_rad, beta=50.0)
        return s
        
    # Try multiple initial angles for the outer hexagon
    # The outer hexagon can be rotated relative to the inner cluster.
    # Symmetries suggest trying 0, 30, 60... but due to the specific shape of the 11-cluster,
    # the optimal angle might be slightly off or exactly 0/30.
    
    init_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
    
    for ang_deg in init_angles:
        if time.time() > deadline - 30:
            break
            
        init_params = [cx_avg, cy_avg, np.radians(ang_deg)]
        
        try:
            # Nelder-Mead is robust for low-dimensional problems without strict gradient requirements
            res = minimize(objective_outer, init_params, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if res.fun < best_side:
                # Verify with exact calculation
                ox, oy, ang_rad = res.x
                s_exact = compute_side_length_exact(initial_inner, (ox, oy), ang_rad)
                if s_exact < best_side:
                    best_side = s_exact
                    best_outer_center = (ox, oy)
                    best_outer_angle_rad = ang_rad
                    best_inner = initial_inner
                    
        except Exception:
            pass
            
    # Phase 2: Joint Refinement
    # Now that we have a good outer container, we can move the inner hexagons slightly
    # to further reduce the bounding box.
    # We will optimize all 3*N + 3 variables.
    
    # Initial params for joint optimization
    p0 = np.zeros(3 * N + 3)
    for i, (x, y, a) in enumerate(best_inner):
        p0[3*i] = x
        p0[3*i+1] = y
        p0[3*i+2] = a
    p0[3*N] = best_outer_center[0]
    p0[3*N+1] = best_outer_center[1]
    p0[3*N+2] = best_outer_angle_rad
    
    def objective_joint(params):
        inner = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner.append((x, y, a))
        ox = params[3*N]
        oy = params[3*N+1]
        ang = params[3*N+2]
        
        # Smooth side length
        s = compute_side_length_smooth(inner, (ox, oy), ang, beta=20.0)
        
        # Penalty for overlaps
        penalty = 0.0
        # Pre-compute vertices for overlap check
        verts = []
        for (ix, iy, iangle) in inner:
            v = get_hex_vertices(ix, iy, 1.0, iangle)
            verts.append(v)
            
        for i in range(N):
            for j in range(i + 1, N):
                # Quick distance check
                dx = inner[i][0] - inner[j][0]
                dy = inner[i][1] - inner[j][1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < 4.0: # (2*1)^2, min distance between centers
                    if sat_overlap_check(verts[i], verts[j], eps=1e-9):
                        # Penalty proportional to overlap depth? 
                        # For simplicity, large penalty if overlapping
                        penalty += 100.0 * (2.0 - np.sqrt(dist_sq))**2
                        
        return s + penalty

    try:
        # Use L-BFGS-B for joint optimization
        # It might be slow, so limit iterations
        res_joint = minimize(objective_joint, p0, method='L-BFGS-B',
                             options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if res_joint.fun < best_side + 1e-5:
            # Extract results
            inner_res = []
            for i in range(N):
                x = res_joint.x[3*i]
                y = res_joint.x[3*i+1]
                a = res_joint.x[3*i+2]
                inner_res.append((x, y, a))
            ox = res_joint.x[3*N]
            oy = res_joint.x[3*N+1]
            ang = res_joint.x[3*N+2]
            
            s_exact = compute_side_length_exact(inner_res, (ox, oy), ang)
            
            # Check validity
            if check_validity(inner_res, (ox, oy), s_exact, ang):
                if s_exact < best_side:
                    best_side = s_exact
                    best_inner = inner_res
                    best_outer_center = (ox, oy)
                    best_outer_angle_rad = ang
                    
    except Exception:
        pass
        
    # Phase 3: Final Polish with Exact Objective (Nelder-Mead on small neighborhood)
    # Sometimes L-BFGS-B with smooth surrogate stops early.
    # We can do a final local search using the exact side length.
    
    def objective_exact(params):
        inner = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner.append((x, y, a))
        ox = params[3*N]
        oy = params[3*N+1]
        ang = params[3*N+2]
        
        s = compute_side_length_exact(inner, (ox, oy), ang)
        
        # Penalty for overlaps (hard constraint)
        penalty = 0.0
        verts = []
        for (ix, iy, iangle) in inner:
            v = get_hex_vertices(ix, iy, 1.0, iangle)
            verts.append(v)
            
        for i in range(N):
            for j in range(i + 1, N):
                if sat_overlap_check(verts[i], verts[j], eps=1e-9):
                    penalty += 1000.0
                    
        return s + penalty

    p_best = np.zeros(3 * N + 3)
    for i, (x, y, a) in enumerate(best_inner):
        p_best[3*i] = x
        p_best[3*i+1] = y
        p_best[3*i+2] = a
    p_best[3*N] = best_outer_center[0]
    p_best[3*N+1] = best_outer_center[1]
    p_best[3*N+2] = best_outer_angle_rad
    
    if time.time() < deadline - 10:
        try:
            res_final = minimize(objective_exact, p_best, method='Nelder-Mead',
                                 options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res_final.fun < best_side + 1e-6:
                inner_res = []
                for i in range(N):
                    x = res_final.x[3*i]
                    y = res_final.x[3*i+1]
                    a = res_final.x[3*i+2]
                    inner_res.append((x, y, a))
                ox = res_final.x[3*N]
                oy = res_final.x[3*N+1]
                ang = res_final.x[3*N+2]
                
                s_exact = compute_side_length_exact(inner_res, (ox, oy), ang)
                
                if check_validity(inner_res, (ox, oy), s_exact, ang):
                    if s_exact < best_side:
                        best_side = s_exact
                        best_inner = inner_res
                        best_outer_center = (ox, oy)
                        best_outer_angle_rad = ang
        except Exception:
            pass

    # Construct return value
    # inner: list of (x, y, angle_degrees)
    # center: (x, y)
    # side: float
    # angle: float (degrees)
    
    inner_out = []
    for (x, y, a) in best_inner:
        inner_out.append([x, y, np.degrees(a) % 360.0])
        
    outer_center_out = (best_outer_center[0], best_outer_center[1])
    outer_angle_out = np.degrees(best_outer_angle_rad) % 360.0
    
    return (inner_out, outer_center_out, best_side, outer_angle_out)

# EVOLVE-BLOCK-END
