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
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """
    Compute smooth approximation of the minimum outer side length.
    Uses Log-Sum-Exp on the support function values.
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    # Normals for a hexagon with angle 0: 30, 90, 150, 210, 270, 330
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    # Collect all projections
    projections = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate to align outer hexagon with standard orientation
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                projections.append(proj)
    
    if not projections:
        return 0.0
    
    # Log-Sum-Exp for smooth max
    max_proj = max(projections)
    # Scale to prevent overflow
    scaled = [beta * (p - max_proj) for p in projections]
    sum_exp = sum(math.exp(s) for s in scaled)
    smooth_max = max_proj + math.log(sum_exp) / beta
    
    return 2.0 * smooth_max / SQRT3

def overlap_penalty(inner, strength=100.0):
    """
    Penalty for overlapping inner hexagons.
    Uses center distance. Minimum distance for non-overlap is sqrt(3) ~ 1.732.
    """
    penalty = 0.0
    n = len(inner)
    min_d = SQRT3 - 0.01 # Slightly less to allow touching
    for i in range(n):
        cx1, cy1, _ = inner[i]
        for j in range(i + 1, n):
            cx2, cy2, _ = inner[j]
            d = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            if d < min_d:
                excess = min_d - d
                penalty += strength * excess**2
    return penalty

def objective(params, deadline):
    """
    Objective function: Smooth outer side length + overlap penalty.
    """
    if time.time() > deadline:
        return float('inf')
    
    n = N
    inner = []
    for i in range(n):
        cx = params[3 * i]
        cy = params[3 * i + 1]
        ang = np.degrees(params[3 * i + 2]) % 360
        inner.append((cx, cy, ang))
    
    ox = params[3 * n]
    oy = params[3 * n + 1]
    outer_angle = np.degrees(params[3 * n + 2]) % 360
    
    S = compute_required_side_smooth(inner, (ox, oy), outer_angle, beta=50.0)
    penalty = overlap_penalty(inner, strength=100.0)
    
    return S + penalty

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            # SAT check
            axes = []
            for v in (vi, vj):
                for k in range(6):
                    p1 = v[k]
                    p2 = v[(k+1)%6]
                    edge = (p2[0]-p1[0], p2[1]-p1[1])
                    normal = (-edge[1], edge[0])
                    norm = math.hypot(normal[0], normal[1])
                    if norm > 1e-9:
                        axes.append((normal[0]/norm, normal[1]/norm))
            
            overlap = True
            for axis in axes:
                min1 = min(p[0]*axis[0] + p[1]*axis[1] for p in vi)
                max1 = max(p[0]*axis[0] + p[1]*axis[1] for p in vi)
                min2 = min(p[0]*axis[0] + p[1]*axis[1] for p in vj)
                max2 = max(p[0]*axis[0] + p[1]*axis[1] for p in vj)
                
                if max1 < min2 + eps or max2 < min1 + eps:
                    overlap = False
                    break
            if overlap:
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
    if not check_all_disjoint(inner):
        return False
    if not check_inside(inner, outer_center, outer_side, outer_angle_deg):
        return False
    return True

def _build_initial_conditions():
    """Generate diverse initial conditions based on hexagonal lattice."""
    conditions = []
    
    # Axial coordinates for a compact cluster of 11
    # Center (0,0), Ring 1 (6), Ring 2 (4 selected)
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-1, -1)
    ]
    
    # Scale factor to ensure no overlap initially (slightly > 1)
    # Distance between centers in lattice is sqrt(3) * scale
    # We want distance > sqrt(3), so scale > 1
    scale = 1.0 + 1e-3
    
    for rot_offset in [0, 30, 60]:
        rot = math.radians(rot_offset)
        ic = np.zeros(3 * N + 3)
        for i, (q, r) in enumerate(axial_coords):
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            # Apply rotation
            xr = x * math.cos(rot) - y * math.sin(rot)
            yr = x * math.sin(rot) + y * math.cos(rot)
            
            ic[3 * i] = xr
            ic[3 * i + 1] = yr
            ic[3 * i + 2] = rot # Align inner hexagons with lattice rotation
            
        ic[3 * N] = 0.0
        ic[3 * N + 1] = 0.0
        ic[3 * N + 2] = rot
        conditions.append(ic)
        
        # Perturbed version
        np.random.seed(42 + rot_offset)
        ic_pert = ic + np.random.randn(3 * N + 3) * 0.05
        conditions.append(ic_pert)

    return conditions

def construct():
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + 135.0
    
    best_result = None
    best_score = float('inf')
    
    initial_conditions = _build_initial_conditions()
    
    for ic in initial_conditions:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(objective, ic, args=(deadline,), method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < float('inf'):
                params = res.x
                inner = []
                for i in range(N):
                    cx = params[3 * i]
                    cy = params[3 * i + 1]
                    ang = np.degrees(params[3 * i + 2]) % 360
                    inner.append([cx, cy, ang])
                ox = params[3 * N]
                oy = params[3 * N + 1]
                outer_angle = np.degrees(params[3 * N + 2]) % 360
                
                # Calculate exact side length for validity check
                # We need to find the exact S that fits this configuration
                # Re-use the smooth function but with high beta or just use the max projection
                # For validity, we compute the exact required side
                ox, oy = ox, oy
                oa = outer_angle
                normals = []
                for i in range(6):
                    theta = math.radians(30 + i * 60)
                    normals.append((math.cos(theta), math.sin(theta)))
                
                cos_a = math.cos(math.radians(oa))
                sin_a = math.sin(math.radians(oa))
                
                max_proj = 0.0
                for (ix, iy, iangle) in inner:
                    verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
                    for v in verts:
                        dx = v[0] - ox
                        dy = v[1] - oy
                        lx = dx * cos_a + dy * sin_a
                        ly = -dx * sin_a + dy * cos_a
                        for nx, ny in normals:
                            proj = lx * nx + ly * ny
                            if proj > max_proj:
                                max_proj = proj
                S = 2.0 * max_proj / SQRT3
                
                if check_validity(inner, (ox, oy), S, outer_angle):
                    if S < best_score:
                        best_score = S
                        best_result = (inner, (ox, oy), S, outer_angle)
        except Exception:
            pass

    # Fallback: Simple lattice
    if best_result is None:
        axial_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, -1), (1, 1), (-2, 1), (-1, -1)
        ]
        inner = []
        for q, r in axial_coords:
            x = 1.5 * q
            y = SQRT3 * (r + q / 2.0)
            inner.append([x, y, 0.0])
        
        # Compute S
        max_proj = 0.0
        normals = []
        for i in range(6):
            theta = math.radians(30 + i * 60)
            normals.append((math.cos(theta), math.sin(theta)))
            
        for (ix, iy, iangle) in inner:
            verts = get_hex_vertices(ix, iy, 1.0, 0.0)
            for v in verts:
                for nx, ny in normals:
                    proj = v[0] * nx + v[1] * ny
                    if proj > max_proj:
                        max_proj = proj
        S = 2.0 * max_proj / SQRT3
        best_result = (inner, (0.0, 0.0), S, 0.0)

    return best_result

# EVOLVE-BLOCK-END
