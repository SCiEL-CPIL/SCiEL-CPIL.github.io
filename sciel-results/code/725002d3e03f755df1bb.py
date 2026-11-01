"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon (flat-top aligned with angle_rad)."""
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

def generate_hex_lattice(n):
    """
    Generate a compact hexagonal lattice of n unit hexagons.
    Returns list of [x, y, angle_deg].
    """
    # Hexagonal coordinates (q, r)
    # Center is (0,0)
    # Ring 0: 1 hex
    # Ring 1: 6 hex
    # Ring 2: 12 hex
    # Total: 1, 7, 19
    
    coords = [(0, 0)]
    
    # Generate rings
    # Directions: E, NE, NW, W, SW, SE
    # In axial coordinates:
    # E: (1, 0)
    # NE: (0, 1)
    # NW: (-1, 1)
    # W: (-1, 0)
    # SW: (0, -1)
    # SE: (1, -1)
    
    def hex_to_pix(q, r):
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        return x, y
    
    # We need n hexagons.
    # If n <= 1, return center.
    # If n <= 7, use ring 1.
    # If n <= 19, use ring 2.
    
    # Let's just generate a spiral or ring by ring and take the first n.
    # To ensure compactness, we fill rings completely.
    
    current_count = 1
    if n == 1:
        return [[0.0, 0.0, 0.0]]
    
    ring = 1
    while current_count < n:
        # Generate hexes in current ring
        # Start at (ring, 0) and go around
        # The 6 corners of the ring are:
        # (ring, 0), (0, ring), (-ring, ring), (-ring, 0), (0, -ring), (ring, -ring)
        
        # We can iterate by "side"
        # Side 1: from (ring, 0) to (0, ring) -> q decreases by 1, r increases by 1? No.
        # Let's use a standard ring generation algorithm.
        
        # Corner 1: (ring, 0)
        # Corner 2: (0, ring)
        # Corner 3: (-ring, ring)
        # Corner 4: (-ring, 0)
        # Corner 5: (0, -ring)
        # Corner 6: (ring, -ring)
        
        corners = [
            (ring, 0),
            (0, ring),
            (-ring, ring),
            (-ring, 0),
            (0, -ring),
            (ring, -ring)
        ]
        
        for i in range(6):
            start = corners[i]
            end = corners[(i + 1) % 6]
            
            # Vector from start to end
            dq = end[0] - start[0]
            dr = end[1] - start[1]
            
            # Number of steps is ring
            for step in range(ring):
                if current_count >= n:
                    break
                q = start[0] + (dq * step) / ring
                r = start[1] + (dr * step) / ring
                # Ensure integers
                q = round(q)
                r = round(r)
                
                # Avoid duplicates (corners are shared)
                if (q, r) in coords:
                    continue
                    
                coords.append((q, r))
                current_count += 1
            if current_count >= n:
                break
        ring += 1
        
    # Convert to pixel coords
    result = []
    for q, r in coords[:n]:
        x, y = hex_to_pix(q, r)
        result.append([x, y, 0.0])
        
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Generate initial layout using hexagonal lattice
    initial_inner = generate_hex_lattice(N)
    
    # Verify disjointness (should be true for lattice with spacing 1.0 * sqrt(3) approx)
    # Note: The distance between centers in lattice is sqrt(3) for flat-top?
    # Let's check:
    # Center (0,0) to (1,0): x=1.5, y=sqrt(3)/2. Dist = sqrt(2.25 + 0.75) = sqrt(3) ~ 1.732.
    # Hexagon width (flat-to-flat) is sqrt(3). So they touch.
    # This is a valid packing (touching is allowed? "disjoint" usually means interiors disjoint.
    # SAT with eps=1e-6 allows touching.
    
    if not check_all_disjoint(initial_inner, eps=1e-5):
        # If lattice fails (shouldn't), fallback to simple grid
        print("Lattice check failed, using grid")
        initial_inner = []
        for i in range(N):
            # Simple grid
            row = i // 4
            col = i % 4
            x = col * 2.0
            y = row * SQRT3
            initial_inner.append([x, y, 0.0])
    
    best_side = float('inf')
    best_inner = initial_inner
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # 2. Optimize Outer Hexagon
    # Calculate centroid
    cx_avg = sum(h[0] for h in initial_inner) / len(initial_inner)
    cy_avg = sum(h[1] for h in initial_inner) / len(initial_inner)
    
    def objective_outer(params):
        cx, cy, angle_deg = params
        return compute_required_side(initial_inner, (cx, cy), angle_deg)
    
    best_val = float('inf')
    best_result = None
    
    # Try a few angles
    init_angles = [0, 30, 60, 90, 120, 150]
    
    for init_angle in init_angles:
        if time.time() > deadline - 20:
            break
        init_params = [cx_avg, cy_avg, float(init_angle)]
        try:
            res = minimize(objective_outer, init_params, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 1000})
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x
        except Exception:
            continue
    
    if best_result is not None:
        best_outer_center = (best_result[0], best_result[1])
        best_outer_angle = best_result[2]
        best_side = best_val
    else:
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side(initial_inner, best_outer_center, best_outer_angle)
    
    best_inner = [list(h) for h in initial_inner]
    
    # 3. Attempt Joint Optimization with Strict Penalty
    # We will try to move inner hexagons to reduce container size.
    # Penalty for overlap is very high.
    
    def joint_objective(params):
        inner_params = params[:3*N]
        outer_params = params[3*N:]
        
        inner = []
        for i in range(N):
            x = inner_params[3*i]
            y = inner_params[3*i+1]
            angle = inner_params[3*i+2]
            inner.append([x, y, angle])
        
        cx, cy, angle_deg = outer_params
        
        # Overlap Penalty
        penalty = 0.0
        for i in range(N):
            vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
            for j in range(i + 1, N):
                vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
                if sat_overlap_check(vi, vj, eps=1e-7):
                    penalty += 100.0
        
        side = compute_required_side(inner, (cx, cy), angle_deg)
        return side + penalty
    
    # Initialize from best
    x0 = []
    for i in range(N):
        x0.extend([best_inner[i][0], best_inner[i][1], best_inner[i][2]])
    x0.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    if time.time() < deadline - 10:
        try:
            res_joint = minimize(joint_objective, x0, method='Nelder-Mead',
                                 options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6, 'maxfev': 600})
            
            # Validate result
            inner_params = res_joint.x[:3*N]
            outer_params = res_joint.x[3*N:]
            
            candidate_inner = []
            for i in range(N):
                candidate_inner.append([inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]])
            
            candidate_center = (outer_params[0], outer_params[1])
            candidate_angle = outer_params[2]
            
            # Strict Check
            if check_all_disjoint(candidate_inner, eps=1e-6):
                candidate_side = compute_required_side(candidate_inner, candidate_center, candidate_angle)
                if candidate_side < best_side:
                    best_side = candidate_side
                    best_inner = candidate_inner
                    best_outer_center = candidate_center
                    best_outer_angle = candidate_angle
        except Exception:
            pass
    
    # Final Polish on Outer
    if time.time() < deadline - 2:
        def final_objective(params):
            cx, cy, angle_deg = params
            return compute_required_side(best_inner, (cx, cy), angle_deg)
        
        try:
            res_final = minimize(final_objective, [best_outer_center[0], best_outer_center[1], best_outer_angle],
                                 method='Nelder-Mead',
                                 options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 500})
            if res_final.fun < best_side:
                best_side = res_final.fun
                best_outer_center = (res_final.x[0], res_final.x[1])
                best_outer_angle = res_final.x[2]
        except Exception:
            pass
    
    # Ensure strict containment
    best_side *= (1.0 + 1e-5)
    
    inner = [[best_inner[i][0], best_inner[i][1], best_inner[i][2]] for i in range(N)]
    return inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
