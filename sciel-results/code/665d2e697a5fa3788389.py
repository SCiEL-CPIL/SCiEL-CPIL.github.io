"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as (x, y) tuples."""
    angles = angle_rad + np.pi / 3.0 * np.arange(6)
    return [(cx + side * math.cos(a), cy + side * math.sin(a)) for a in angles]

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
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Define 11 hexagons on a hexagonal lattice - use tight, symmetric arrangement
    # Center hexagon + 6 around it + 4 more in a compact cluster
    # Axial coordinates for hexagonal lattice (q, r)
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-1, -1)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    # Convert axial to Cartesian with tight packing
    # For unit hexagons, the distance between centers in adjacent positions is sqrt(3)
    # Axial to Cartesian: x = 1.5 * q, y = sqrt(3) * (r + q/2)
    scale = 1.0
    
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
        
    # Initial layout: All hexagons aligned (angle 0)
    # This is a valid tight packing for the standard hexagonal lattice
    
    def objective(params):
        """Objective function: outer side length + penalty for overlaps."""
        n = N
        inner = []
        for i in range(n):
            cx = params[3*i]
            cy = params[3*i+1]
            ang = np.degrees(params[3*i+2]) % 360
            inner.append((cx, cy, ang))
        ox = params[3*n]
        oy = params[3*n+1]
        outer_angle = np.degrees(params[3*n+2]) % 360
        
        S = compute_required_side(inner, (ox, oy), outer_angle)
        
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                cx1, cy1, a1 = inner[i]
                cx2, cy2, a2 = inner[j]
                d = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                if d < 2.0:
                    v1 = get_hex_vertices(cx1, cy1, 1.0, math.radians(a1))
                    v2 = get_hex_vertices(cx2, cy2, 1.0, math.radians(a2))
                    if sat_overlap_check(v1, v2, 1e-8):
                        penalty += (2.0 - d) ** 2 * 100
        
        return S + penalty
    
    # Generate diverse initial conditions
    initial_conditions = []
    
    # IC 1: Standard hexagonal lattice, outer angle 0
    ic1 = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(centers):
        ic1[3*i] = px
        ic1[3*i+1] = py
        ic1[3*i+2] = 0.0
    ic1[3*N] = 0.0
    ic1[3*N+1] = 0.0
    ic1[3*N+2] = 0.0
    initial_conditions.append(ic1)
    
    # IC 2: Standard hexagonal lattice, outer angle 30
    ic2 = ic1.copy()
    ic2[3*N+2] = np.radians(30)
    initial_conditions.append(ic2)
    
    # IC 3: Slightly rotated inner hexagons
    ic3 = ic1.copy()
    for i in range(N):
        ic3[3*i+2] = np.radians(15)
    initial_conditions.append(ic3)
    
    # IC 4: Perturbed standard lattice
    np.random.seed(42)
    ic4 = ic1 + np.random.randn(3*N+3) * 0.05
    initial_conditions.append(ic4)
    
    # IC 5: Tighter lattice (scale 0.98) - may cause overlaps, optimizer should fix
    centers_tight = []
    scale_tight = 0.98
    for q, r in axial_coords:
        x = 1.5 * scale_tight * q
        y = SQRT3 * scale_tight * (r + q / 2.0)
        centers_tight.append((x, y))
    
    ic5 = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(centers_tight):
        ic5[3*i] = px
        ic5[3*i+1] = py
        ic5[3*i+2] = 0.0
    ic5[3*N] = 0.0
    ic5[3*N+1] = 0.0
    ic5[3*N+2] = 0.0
    initial_conditions.append(ic5)
    
    # IC 6: Clustered arrangement (center + tight ring)
    positions6 = [(0.0, 0.0)]
    for k in range(6):
        angle = np.radians(60 * k)
        x = 1.7 * np.cos(angle)
        y = 1.7 * np.sin(angle)
        positions6.append((x, y))
    for k in range(4):
        angle = np.radians(30 + 60 * k)
        x = 3.0 * np.cos(angle)
        y = 3.0 * np.sin(angle)
        positions6.append((x, y))
    ic6 = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(positions6[:N]):
        ic6[3*i] = px
        ic6[3*i+1] = py
        ic6[3*i+2] = 0.0
    ic6[3*N] = 0.0
    ic6[3*N+1] = 0.0
    ic6[3*N+2] = 0.0
    initial_conditions.append(ic6)
    
    best_result = None
    best_score = float('inf')
    
    # Optimize each initial condition
    for idx, ic in enumerate(initial_conditions):
        if time.time() > deadline - 20:
            break
        
        try:
            res = minimize(objective, ic, method='L-BFGS-B', 
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.success or res.fun < best_score:
                params = res.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append([cx, cy, ang])
                ox = params[3*N]
                oy = params[3*N+1]
                outer_angle = np.degrees(params[3*N+2]) % 360
                
                S = compute_required_side(inner, (ox, oy), outer_angle)
                
                if check_validity(inner, (ox, oy), S, outer_angle):
                    if S < best_score:
                        best_score = S
                        best_result = (inner, (ox, oy), S, outer_angle)
        except Exception:
            pass
    
    # Fallback: Use the initial lattice if optimization failed
    if best_result is None:
        inner = [[px, py, 0.0] for px, py in centers]
        S = compute_required_side(inner, (0.0, 0.0), 0.0) + 0.1
        best_result = (inner, (0.0, 0.0), S, 0.0)
    
    # Refine if time permits
    if time.time() < deadline - 50:
        try:
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = np.radians(ang)
            params_best[3*N] = best_result[1][0]
            params_best[3*N+1] = best_result[1][1]
            params_best[3*N+2] = np.radians(best_result[3])
            
            res = minimize(objective, params_best, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res.fun < best_score:
                params = res.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append([cx, cy, ang])
                ox = params[3*N]
                oy = params[3*N+1]
                outer_angle = np.degrees(params[3*N+2]) % 360
                S = compute_required_side(inner, (ox, oy), outer_angle)
                
                if check_validity(inner, (ox, oy), S, outer_angle):
                    best_result = (inner, (ox, oy), S, outer_angle)
        except Exception:
            pass
    
    return best_result

# EVOLVE-BLOCK-END
