import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

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
    inner: list of [x, y, angle_deg]
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    # Normals for a hexagon with angle 0 (vertices at 0, 60, ...):
    # Outward normals are at 30, 90, 150, 210, 270, 330 degrees.
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    # Rotate outer frame to match outer_angle
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            # Translate to outer center
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate points by -outer_angle to align with standard hex axes
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * max_projection / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap (including touching).
    """
    axes = []
    
    # Get axes from edges of v1
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    # Get axes from edges of v2
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
        
        # Check for gap
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

def check_inside(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """Check if all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    # Outer hexagon vertices
    outer_verts = get_hex_vertices(ox, oy, outer_side, math.radians(outer_angle_deg))
    
    # For each inner hexagon, check if all vertices are inside the outer hexagon
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for (vx, vy) in verts:
            # Check if point is inside convex polygon using cross products
            inside = True
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                # Vector from p1 to p2
                ex = p2[0] - p1[0]
                ey = p2[1] - p1[1]
                # Vector from p1 to point
                px = vx - p1[0]
                py = vy - p1[1]
                # Cross product
                cross = ex * py - ey * px
                # For a convex polygon with vertices in counter-clockwise order,
                # the point is inside if all cross products have the same sign
                # We assume the hexagon is oriented such that interior is to the left
                if cross < -eps:
                    inside = False
                    break
            if not inside:
                return False
    return True

def check_validity(inner, outer_center, outer_side, outer_angle_deg):
    """Check if the configuration is valid: no overlaps, all inside outer hexagon."""
    if not check_all_disjoint(inner):
        return False
    if not check_inside(inner, outer_center, outer_side, outer_angle_deg):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Define 11 hexagons on a hexagonal lattice
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-1, -1)
    ]
    
    scale = math.sqrt(3)
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
        
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Center the configuration
    cx = sum(p[0] for p in inner) / N
    cy = sum(p[1] for p in inner) / N
    for p in inner:
        p[0] -= cx
        p[1] -= cy

    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False
    
    def update_best(inner_cfg, S, ocx, ocy, oangle):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        if S < best_S:
            best_S = S
            best_inner = [list(x) for x in inner_cfg]
            best_center = (ocx, ocy)
            best_angle = oangle
            best_valid = True

    # 2. Optimization using Scipy
    # We optimize the positions and angles of the 11 hexagons.
    # Variables: x1, y1, a1, x2, y2, a2, ..., x11, y11, a11
    # Total 33 variables.
    
    # Initial guess
    x0 = []
    for p in inner:
        x0.extend([p[0], p[1], p[2]])
    
    # Objective function
    def objective(params):
        if time.time() > deadline:
            return float('inf')
            
        # Unpack
        cur_inner = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            cur_inner.append([x, y, a])
            
        # Compute S
        S = compute_required_side(cur_inner, (0,0), best_angle)
        
        # Penalty for overlaps
        # Use a simple check: if overlap, add large penalty
        # To make it fast, we can just check validity and return inf if invalid?
        # No, that makes the landscape discontinuous.
        # Let's use a penalty based on min distance.
        # For now, let's just check validity and return S + 1000 * overlap_count
        # This is not differentiable, but Nelder-Mead doesn't need gradients.
        
        # Fast overlap check
        overlap_penalty = 0.0
        # Pre-compute vertices
        verts_list = []
        for p in cur_inner:
            verts_list.append(get_hex_vertices(p[0], p[1], 1.0, math.radians(p[2])))
        
        for i in range(N):
            for j in range(i+1, N):
                if sat_overlap_check(verts_list[i], verts_list[j], eps=1e-9):
                    overlap_penalty += 100.0
                    
        return S + overlap_penalty

    # We will try a few outer angles
    outer_angles = [0.0, 30.0]
    
    for oangle in outer_angles:
        if time.time() > deadline:
            break
            
        best_angle = oangle
        # Reset best for this angle? No, we want global best.
        
        # Run optimization
        # Use Nelder-Mead for derivative-free optimization
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if time.time() > deadline:
                break
                
            # Check result
            cur_inner = []
            for i in range(N):
                x = res.x[i*3]
                y = res.x[i*3+1]
                a = res.x[i*3+2]
                cur_inner.append([x, y, a])
                
            # Verify validity
            if check_all_disjoint(cur_inner):
                S = compute_required_side(cur_inner, (0,0), oangle)
                if S < best_S:
                    update_best(cur_inner, S, 0.0, 0.0, oangle)
        except Exception:
            pass

    # Fallback
    if not best_valid or best_inner is None:
        S = compute_required_side(inner, (0,0), 0.0)
        return inner, (0.0, 0.0), S, 0.0

    return best_inner, best_center, best_S, best_angle
