"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    return 2.0 * max_projection / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """
    Smooth approximation using log-sum-exp for the max projection.
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    projections = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                projections.append(proj)
    
    if not projections:
        return 0.0
    
    max_proj = max(projections)
    if max_proj < -1000:
        return 0.0
    
    scaled = [beta * (p - max_proj) for p in projections]
    sum_exp = sum(math.exp(s) for s in scaled)
    smooth_max = max_proj + math.log(sum_exp) / beta
    
    return 2.0 * smooth_max / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons overlap using SAT.
    Returns True if they overlap (including touching).
    """
    axes = []
    
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Axial coordinate helpers
    def hex_dist(q1, r1, q2, r2):
        return max(abs(q1-q2), abs(r1-r2), abs((q1+r1)-(q2+r2)))

    # Generate all connected clusters of size N=11 within a reasonable bounding box
    # We'll use a systematic enumeration of all hexagonal clusters of size 11
    # that fit within a hex of radius 3 (which gives us enough room)
    
    # Pool of axial coordinates within hex distance <= 3 from origin
    pool = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if max(abs(q), abs(r), abs(q+r)) <= 3:
                pool.append((q, r))
    
    # Build adjacency
    adj = {}
    for i, (q1, r1) in enumerate(pool):
        adj[i] = []
        for j, (q2, r2) in enumerate(pool):
            if i != j and hex_dist(q1, r1, q2, r2) == 1:
                adj[i].append(j)
    
    # Enumerate connected subsets of size N using DFS
    # This can be expensive, so we'll use a smarter approach:
    # Generate all "hexagonal polyominoes" of size 11 that are connected
    # and have a bounding hex radius <= 3
    
    # Actually, let's use a different approach: enumerate all subsets of the pool
    # of size 11 that are connected, but prune aggressively.
    # For N=11 and pool size ~37, C(37,11) is too large.
    # Instead, let's use the fact that we want compact shapes.
    
    # Better approach: use known compact hexagonal clusters and also do local search
    # on the positions of the hexagons.
    
    # Let's define several candidate base arrangements
    base_arrangements = []
    
    # Arrangement 1: center + 6 neighbors + 4 more (compact)
    # (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (1,1), (0,2), (-1,2)
    base_arrangements.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ])
    
    # Arrangement 2: different 4 outer positions
    base_arrangements.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (2, -1), (1, 1), (0, 2)
    ])
    
    # Arrangement 3
    base_arrangements.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-1, 2), (1, 1), (0, 2)
    ])
    
    # Arrangement 4
    base_arrangements.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (2, -1), (-1, 2), (0, 2)
    ])
    
    # Arrangement 5: more symmetric
    base_arrangements.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ])
    
    # Arrangement 6
    base_arrangements.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (2, -1), (-2, 0)
    ])
    
    # Arrangement 7
    base_arrangements.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 0)
    ])
    
    # Now for each arrangement, we'll do joint optimization of:
    # 1. The positions of the inner hexagons (allowing small perturbations from lattice)
    # 2. The rotations of each inner hexagon
    # 3. The outer hexagon center and rotation
    # 
    # But this is a high-dimensional problem. Let's be more strategic.
    # 
    # Strategy:
    # Phase 1: For each base arrangement, optimize the outer hexagon (center + angle)
    #          assuming inner hexagons are on the lattice with angle 0.
    # Phase 2: From the best result, do a joint local optimization allowing
    #          small perturbations of inner positions and rotations.
    # Phase 3: Final polish of the outer container.
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    epsilon = 1e-5
    scale = 1.0 + epsilon
    
    for arr_idx, axial_coords in enumerate(base_arrangements):
        if time.time() > deadline - 40:
            break
        
        if len(axial_coords) != N:
            continue
        
        # Convert to Cartesian
        centers = []
        for q, r in axial_coords:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            centers.append((x, y))
        
        initial_inner = [[x, y, 0.0] for x, y in centers]
        
        # Centroid
        cx_avg = sum(c[0] for c in centers) / N
        cy_avg = sum(c[1] for c in centers) / N
        
        # Phase 1: Optimize outer hexagon
        def obj_outer_smooth(params):
            cx, cy, angle_deg = params
            return compute_required_side_smooth(initial_inner, (cx, cy), angle_deg, beta=50.0)
        
        def obj_outer_exact(params):
            cx, cy, angle_deg = params
            return compute_required_side(initial_inner, (cx, cy), angle_deg)
        
        best_arr_val = float('inf')
        best_arr_result = None
        
        init_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        
        for init_angle in init_angles:
            if time.time() > deadline - 35:
                break
            
            init_params = [cx_avg, cy_avg, float(init_angle)]
            
            try:
                res_smooth = minimize(obj_outer_smooth, init_params, method='Nelder-Mead',
                                      options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 500})
                
                if res_smooth.fun < best_arr_val:
                    res_exact = minimize(obj_outer_exact, res_smooth.x, method='Nelder-Mead',
                                        options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 2000})
                    if res_exact.fun < best_arr_val:
                        best_arr_val = res_exact.fun
                        best_arr_result = res_exact.x
                    else:
                        best_arr_val = res_smooth.fun
                        best_arr_result = res_smooth.x
            except Exception:
                pass
        
        if best_arr_result is None:
            continue
        
        cur_side = best_arr_val
        cur_inner = initial_inner
        cur_center = (best_arr_result[0], best_arr_result[1])
        cur_angle = best_arr_result[2]
        
        # Phase 2: Joint optimization with small perturbations
        # Parameters: for each inner hex, (dx, dy, dangle) + outer (cx, cy, oangle)
        # Total: 3*N + 3 = 36 parameters
        
        def joint_objective(params):
            if time.time() > deadline - 5:
                return 1e6
            
            inner_params = params[:3*N]
            outer_params = params[3*N:]
            ocx, ocy, oangle = outer_params
            
            # Build perturbed inner
            pert_inner = []
            for i in range(N):
                dx = inner_params[3*i]
                dy = inner_params[3*i+1]
                dangle = inner_params[3*i+2]
                px = centers[i][0] + dx
                py = centers[i][1] + dy
                pangle = dangle
                pert_inner.append([px, py, pangle])
            
            # Check disjointness with a penalty
            penalty = 0.0
            for i in range(N):
                vi = get_hex_vertices(pert_inner[i][0], pert_inner[i][1], 1.0, math.radians(pert_inner[i][2]))
                for j in range(i+1, N):
                    vj = get_hex_vertices(pert_inner[j][0], pert_inner[j][1], 1.0, math.radians(pert_inner[j][2]))
                    # Quick check: if centers are too close, there's likely overlap
                    cx_diff = vi[0][0] - vj[0][0]
                    cy_diff = vi[0][1] - vj[0][1]
                    dist = math.sqrt(cx_diff**2 + cy_diff**2)
                    if dist < 3.5:  # Potential overlap region
                        if sat_overlap_check(vi, vj, 1e-8):
                            penalty += 10.0
            
            side = compute_required_side(pert_inner, (ocx, ocy), oangle)
            return side + penalty
        
        # Only do joint optimization if we have time
        if time.time() < deadline - 25 and cur_side < best_side:
            init_joint = []
            for i in range(N):
                init_joint.extend([0.0, 0.0, 0.0])
            init_joint.extend([cur_center[0], cur_center[1], cur_angle])
            
            try:
                res_joint = minimize(joint_objective, init_joint, method='Nelder-Mead',
                                     options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 3000})
                
                if res_joint.fun < cur_side - 1e-8:
                    # Extract results
                    inner_params = res_joint.x[:3*N]
                    outer_params = res_joint.x[3*N:]
                    new_inner = []
                    for i in range(N):
                        new_inner.append([
                            centers[i][0] + inner_params[3*i],
                            centers[i][1] + inner_params[3*i+1],
                            inner_params[3*i+2]
                        ])
                    new_center = (outer_params[0], outer_params[1])
                    new_angle = outer_params[2]
                    new_side = compute_required_side(new_inner, new_center, new_angle)
                    
                    # Verify disjointness
                    if check_all_disjoint(new_inner, 1e-6):
                        cur_side = new_side
                        cur_inner = new_inner
                        cur_center = new_center
                        cur_angle = new_angle
            except Exception:
                pass
        
        if cur_side < best_side:
            best_side = cur_side
            best_inner = cur_inner
            best_outer_center = cur_center
            best_outer_angle = cur_angle
    
    # Fallback
    if best_inner is None:
        axial_coords = base_arrangements[0]
        centers = []
        for q, r in axial_coords:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            centers.append((x, y))
        best_inner = [[x, y, 0.0] for x, y in centers]
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    
    # Phase 3: Final high-precision polish of the outer container
    def obj_polish(params):
        cx, cy, angle_deg = params
        return compute_required_side(best_inner, (cx, cy), angle_deg)
    
    best_polish_val = best_side
    best_polish_params = [best_outer_center[0], best_outer_center[1], best_outer_angle]
    
    # Try multiple starting angles for polish
    polish_starts = [[best_outer_center[0], best_outer_center[1], best_outer_angle]]
    for a in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
        polish_starts.append([best_outer_center[0], best_outer_center[1], float(a)])
    
    for ps in polish_starts:
        if time.time() > deadline - 3:
            break
        try:
            res = minimize(obj_polish, ps, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 10000})
            if res.fun < best_polish_val:
                best_polish_val = res.fun
                best_polish_params = res.x
        except Exception:
            pass
    
    if best_polish_params is not None:
        best_outer_center = (best_polish_params[0], best_polish_params[1])
        best_outer_angle = best_polish_params[2]
        best_side = best_polish_val
    
    # Final verification and cleanup
    # Ensure the result is valid
    final_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    if not check_all_disjoint(best_inner, 1e-6):
        # Fallback to a known valid arrangement
        axial_coords = base_arrangements[0]
        centers = []
        for q, r in axial_coords:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            centers.append((x, y))
        best_inner = [[x, y, 0.0] for x, y in centers]
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    else:
        best_side = final_side
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
