"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = SQRT3 / 2.0

# EVOLVE-BLOCK-START

def get_hex_support(angle_rad, normal_angle_rad):
    """
    Compute the support function of a unit regular hexagon (side 1) 
    centered at origin, rotated by angle_rad, 
    projected onto a unit normal at normal_angle_rad.
    
    Vertices are at angle_rad + k*pi/3.
    h = max(cos(normal_angle - (angle_rad + k*pi/3)))
    """
    # Relative angle
    delta = normal_angle_rad - angle_rad
    # Normalize delta to [0, pi/3)
    # The max cos is 1 when delta is multiple of pi/3.
    # Actually, h(theta) for regular hexagon is periodic with pi/3.
    # h(delta) = cos(delta mod pi/3 - pi/6) * (1 / cos(pi/6))? No.
    # Let's just calculate the 6 values. It's fast enough.
    # Or use the formula:
    # The support function of a regular polygon with apothem 'a' is a / cos(delta) where delta is distance to nearest normal.
    # For unit hexagon, apothem a = sqrt(3)/2.
    # The normals of the inner hexagon are at angle_rad + pi/6 + k*pi/3.
    # We want projection on normal_angle_rad.
    # Let alpha = normal_angle_rad - (angle_rad + pi/6).
    # The nearest inner normal is at alpha mod pi/3.
    # Distance to nearest normal d = alpha mod pi/3.
    # If d > pi/6, d = pi/3 - d.
    # h = a / cos(d).
    
    alpha = normal_angle_rad - (angle_rad + math.pi / 6.0)
    # Normalize alpha to [0, pi/3)
    alpha = alpha % (math.pi / 3.0)
    # Distance to closest multiple of pi/3 (which corresponds to a normal)
    # The function is symmetric around pi/6.
    if alpha > math.pi / 6.0:
        d = math.pi / 3.0 - alpha
    else:
        d = alpha
    
    # h = apothem / cos(d)
    return APOTHEM / math.cos(d)

def compute_outer_side_fast(centers, angles_rad, outer_cx, outer_cy, outer_angle_rad):
    """
    Compute the minimum outer side length required to contain the inner hexagons.
    centers: list of (x, y)
    angles_rad: list of rotation angles in radians
    outer_cx, outer_cy: center of outer hexagon
    outer_angle_rad: rotation of outer hexagon
    """
    if not centers:
        return 0.0
    
    # Normals of the outer hexagon
    # Vertices of outer hex are at outer_angle_rad + k*pi/3
    # Normals are at outer_angle_rad + pi/6 + k*pi/3
    max_proj = 0.0
    
    for k in range(6):
        n_angle = outer_angle_rad + math.pi / 6.0 + k * math.pi / 3.0
        nx = math.cos(n_angle)
        ny = math.sin(n_angle)
        
        # For each inner hexagon, compute max projection in this direction
        # proj = n . (c_j - C) + h_j(n)
        # We want max over j of (n . c_j + h_j(n)) - n . C
        # Let M = max_j (n . c_j + h_j(n))
        # Then constraint is M - n . C <= A
        # So A >= M - n . C
        
        max_inner_proj = -1e18
        for i in range(len(centers)):
            cx, cy = centers[i]
            ang = angles_rad[i]
            
            # Projection of center
            p_c = cx * nx + cy * ny
            
            # Support function
            h = get_hex_support(ang, n_angle)
            
            val = p_c + h
            if val > max_inner_proj:
                max_inner_proj = val
        
        # Subtract projection of outer center
        p_C = outer_cx * nx + outer_cy * ny
        proj = max_inner_proj - p_C
        
        if proj > max_proj:
            max_proj = proj
            
    # A = max_proj
    # S = 2 * A / sqrt(3)
    return 2.0 * max_proj / SQRT3

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
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
    verts_list = []
    for i in range(n):
        x, y, ang_deg = inner[i]
        verts = get_hex_vertices(x, y, 1.0, math.radians(ang_deg))
        verts_list.append(verts)
        
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap_check(verts_list[i], verts_list[j], eps):
                return False
    return True

def check_inside(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """Check if all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    outer_verts = get_hex_vertices(ox, oy, outer_side, math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for (vx, vy) in verts:
            inside = True
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                ex = p2[0] - p1[0]
                ey = p2[1] - p1[1]
                px = vx - p1[0]
                py = vy - p1[1]
                cross = ex * py - ey * px
                if cross < -eps:
                    inside = False
                    break
            if not inside:
                return False
    return True

def check_validity(inner, outer_center, outer_side, outer_angle_deg):
    """Check if the configuration is valid."""
    if not check_all_disjoint(inner):
        return False
    if not check_inside(inner, outer_center, outer_side, outer_angle_deg):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Candidate topologies (axial coordinates)
    # (q, r) -> x = sqrt(3) * (q + r/2), y = 1.5 * r
    # Note: The inspiration used x = sqrt(3)*(q+r/2), y=1.5r.
    # My current program used x=1.5*q, y=sqrt(3)*(r+q/2).
    # These are just rotations/reflections. I will stick to one convention.
    # Let's use: x = 1.5 * q, y = sqrt(3) * (r + q/2)
    
    candidates = [
        # Cluster: Center + 6 neighbors + 4 more
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, -1), (1, 1), (-2, 1), (-1, -1)],
        # Linear-ish
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (3, 0), (3, 1)],
        # Diamond
        [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (2, 0), (-2, 0), (0, 2), (0, -2)],
        # Compact 3x4 block
        [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2)],
        # Another variant
        [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (1, -1), (2, -1)],
        # Hexagonal ring + center + 4 extra
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0), (-2, 0), (0, 2), (1, 1)],
    ]

    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    for ax_coords in candidates:
        if time.time() > deadline - 20:
            break
            
        if len(ax_coords) != N:
            continue
            
        # Convert to Cartesian
        centers = []
        for q, r in ax_coords:
            x = 1.5 * q
            y = SQRT3 * (r + q / 2.0)
            centers.append((x, y))
            
        # Initial angles: 0
        angles_rad = [0.0] * N
        
        # Initial outer center: centroid
        cx_avg = sum(c[0] for c in centers) / N
        cy_avg = sum(c[1] for c in centers) / N
        oa_avg = 0.0
        
        # Objective function for joint optimization
        def objective(params):
            # Unpack
            centers_opt = []
            angles_opt = []
            for i in range(N):
                x = params[3*i]
                y = params[3*i+1]
                a = params[3*i+2]
                centers_opt.append((x, y))
                angles_opt.append(a)
                
            ocx = params[3*N]
            ocy = params[3*N+1]
            oang = params[3*N+2]
            
            # Compute side length
            S = compute_outer_side_fast(centers_opt, angles_opt, ocx, ocy, oang)
            
            # Overlap penalty
            # Approximate overlap by distance between centers
            # If dist < 2, they might overlap.
            penalty = 0.0
            for i in range(N):
                xi, yi = centers_opt[i]
                for j in range(i+1, N):
                    xj, yj = centers_opt[j]
                    dx = xi - xj
                    dy = yi - yj
                    d = math.sqrt(dx*dx + dy*dy)
                    if d < 2.0:
                        penalty += (2.0 - d)**2 * 10.0
            
            return S + penalty

        # Initial params
        init_params = []
        for i in range(N):
            init_params.append(centers[i][0])
            init_params.append(centers[i][1])
            init_params.append(angles_rad[i])
        init_params.append(cx_avg)
        init_params.append(cy_avg)
        init_params.append(oa_avg)
        
        try:
            # Optimize
            res = minimize(objective, init_params, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.success or res.fun < best_side:
                # Extract result
                inner_res = []
                for i in range(N):
                    x = res.x[3*i]
                    y = res.x[3*i+1]
                    a_deg = math.degrees(res.x[3*i+2])
                    inner_res.append([x, y, a_deg])
                
                ocx = res.x[3*N]
                ocy = res.x[3*N+1]
                oang_deg = math.degrees(res.x[3*N+2])
                
                # Compute exact side length
                S_res = compute_outer_side_fast(
                    [(p[0], p[1]) for p in inner_res],
                    [math.radians(p[2]) for p in inner_res],
                    ocx, ocy, math.radians(oang_deg)
                )
                
                # Check validity
                if check_validity(inner_res, (ocx, ocy), S_res, oang_deg):
                    if S_res < best_side:
                        best_side = S_res
                        best_inner = inner_res
                        best_outer_center = (ocx, ocy)
                        best_outer_angle = oang_deg
                        
        except Exception as e:
            pass

    # Fallback if no valid solution found (should not happen)
    if best_inner is None:
        # Use the first candidate with a large enough outer hex
        ax_coords = candidates[0]
        centers = []
        for q, r in ax_coords:
            x = 1.5 * q
            y = SQRT3 * (r + q / 2.0)
            centers.append((x, y))
            
        best_inner = [[x, y, 0.0] for x, y in centers]
        cx_avg = sum(c[0] for c in centers) / N
        cy_avg = sum(c[1] for c in centers) / N
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_outer_side_fast(
            [(p[0], p[1]) for p in best_inner],
            [0.0] * N,
            cx_avg, cy_avg, 0.0
        )
        # Ensure validity by expanding if needed (though compute_outer_side_fast should be tight)
        # Add a small buffer
        best_side += 0.01

    return (best_inner, best_outer_center, best_side, best_outer_angle)

# EVOLVE-BLOCK-END
