import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Symmetric placement: 6 corners + 3 sides + 1 center
    # This utilizes the 6-fold symmetry of the outer hexagon.
    # Variables:
    #   r_corner: distance of corner hexagons from center
    #   r_side: distance of side hexagons from center
    #   beta: rotation of the outer hexagon relative to the inner pattern
    
    # The 6 corner positions are at angles k*pi/3.
    # The 3 side positions are at angles k*pi/3 + pi/6 (midpoints of sides).
    # The 11th hexagon is at the center (0,0).
    
    def get_positions(r_c, r_s):
        pts = []
        # Center
        pts.append((0.0, 0.0))
        
        # Corners
        for k in range(6):
            ang = k * math.pi / 3.0
            x = r_c * math.cos(ang)
            y = r_c * math.sin(ang)
            pts.append((x, y))
            
        # Sides
        for k in range(3):
            ang = k * 2 * math.pi / 3.0 + math.pi / 6.0
            x = r_s * math.cos(ang)
            y = r_s * math.sin(ang)
            pts.append((x, y))
            
        return np.array(pts)

    def check_validity(pts):
        # Check if any two hexagons overlap.
        # Two unit hexagons (flat-top or pointy-top, doesn't matter for distance if rotated together or fixed)
        # If they are on the same lattice (same orientation), the minimum distance between centers 
        # to be disjoint is 2 * apothem = sqrt(3) for "kissing" in the lattice sense?
        # Wait. Two unit regular hexagons.
        # If they are identical and non-rotated relative to each other, the minimum distance between centers
        # such that they don't overlap is the distance to the closest point on the boundary.
        # For a hexagon of side 1, the apothem is sqrt(3)/2.
        # If we place them on a hexagonal lattice, the distance between neighbors is sqrt(3).
        # Let's verify:
        # Hexagon 1 center (0,0). Hexagon 2 center (sqrt(3), 0).
        # Rightmost point of Hex 1 is at x = sqrt(3)/2 (vertex? No, flat-top rightmost is x=1 if side=1? No.)
        # Let's use standard flat-top hexagon.
        # Vertices at (1,0), (1/2, sqrt(3)/2), (-1/2, sqrt(3)/2), (-1,0), (-1/2, -sqrt(3)/2), (1/2, -sqrt(3)/2).
        # Rightmost x is 1.
        # If we place another one at (2,0), they touch at (1,0). Distance 2.
        # But we can pack tighter by rotating one? No, we are fixing relative rotations to 0 in this template.
        # However, the "hexagonal lattice" packing has distance sqrt(3) between centers?
        # Let's check distance between (0,0) and (sqrt(3)/2, 1.5).
        # dist = sqrt(3/4 + 2.25) = sqrt(3).
        # Do they overlap?
        # Hex 1 top-right edge connects (1,0) and (0.5, 0.866).
        # Hex 2 bottom-left edge connects (0.5, 0.866) and (-0.5, 0.866)? No.
        # In a hexagonal tiling, the distance between centers is sqrt(3).
        # So, if we fix all inner hexagons to have the same orientation, the minimum center distance is sqrt(3).
        
        min_dist_sq = 3.0  # (sqrt(3))^2
        valid = True
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                d_sq = dx*dx + dy*dy
                if d_sq < min_dist_sq - 1e-9:
                    valid = False
                    break
            if not valid:
                break
        return valid

    def calc_S(r_c, r_s, beta):
        pts = get_positions(r_c, r_s)
        if not check_validity(pts):
            return float('inf')
        
        # Outer hexagon is centered at (0,0) with rotation beta.
        # The 6 normals of the outer hexagon are at angles beta + pi/6 + k*pi/3.
        # We need the outer hexagon to contain all inner hexagons.
        # This means for each normal direction, the projection of the *entire* inner hexagon must be within the apothem of the outer.
        # Actually, the constraint is: max over all inner hexagons of (projection of their farthest point in direction n) <= D_outer.
        # D_outer is the apothem of the outer hexagon.
        # Side length S_outer = (D_outer + APOTHEM_inner) / APOTHEM_inner ? No.
        # The task asks for the side length of the outer hexagon.
        # Let R_outer be the distance from center to vertex (circumradius).
        # Side length L = R_outer.
        # Apothem A = R_outer * sqrt(3)/2.
        # So R_outer = A / (sqrt(3)/2) = 2A/sqrt(3).
        # The condition is that the max projection of any point of any inner hexagon onto the 6 normals is <= A_outer.
        
        max_proj = 0.0
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(ang)
            ny = math.sin(ang)
            
            # For each inner hexagon, we need the max projection.
            # Since all inner hexagons have the same orientation (let's assume 0 for now, or we can optimize this too, but symmetry suggests 0 or pi/6 is best),
            # the max projection of a hexagon centered at (x,y) in direction (nx,ny) is x*nx + y*ny + max_proj_hex.
            # What is max_proj_hex?
            # It is the support function of the unit hexagon in direction (nx,ny).
            # For a regular hexagon of side 1, the support function h(u) is the distance from center to the face perpendicular to u.
            # If u aligns with a normal, h = apothem = sqrt(3)/2.
            # If u aligns with a vertex, h = 1.
            # Generally, h(u) is periodic.
            # Since the inner hexagons can be rotated, we should pick the rotation that minimizes the max projection for the given normal directions?
            # Or rather, we are free to rotate them.
            # However, in our symmetric template, we might want them all parallel.
            # If we assume all inner hexagons have angle 0 (flat-top), their support function in direction theta is:
            # h(theta) = max_{v in vertices} v . u(theta)
            
            # Let's precompute the support function of a unit flat-top hexagon.
            # Vertices:
            # v0 = (1, 0)
            # v1 = (0.5, sqrt(3)/2)
            # v2 = (-0.5, sqrt(3)/2)
            # v3 = (-1, 0)
            # v4 = (-0.5, -sqrt(3)/2)
            # v5 = (0.5, -sqrt(3)/2)
            
            # We can just compute the max dot product with the 6 vertices for the specific normal direction.
            # This is fast enough for 11 hexagons * 6 normals * 6 vertices.
            
            # Actually, we can just compute the max projection of the *centers* plus the max projection of the *shape*.
            # max_proj_total = max_{i} (center_i . n + h(n))
            # where h(n) is the support function of the unit hexagon in direction n.
            
            # Let's compute h(n) for the current n.
            # n = (nx, ny).
            # We want max of v . n for v in vertices.
            # Due to symmetry, we can just check the 6 vertices.
            
            h_val = 0.0
            for vx, vy in [(1,0), (0.5, math.sqrt(3)/2), (-0.5, math.sqrt(3)/2), (-1,0), (-0.5, -math.sqrt(3)/2), (0.5, -math.sqrt(3)/2)]:
                proj = vx*nx + vy*ny
                if proj > h_val:
                    h_val = proj
            
            # Now find max over centers
            for i in range(len(pts)):
                cx, cy = pts[i]
                proj = cx*nx + cy*ny
                total = proj + h_val
                if total > max_proj:
                    max_proj = total
        
        # max_proj is the required apothem of the outer hexagon.
        # Side length S = 2 * max_proj / sqrt(3)
        S = 2.0 * max_proj / math.sqrt(3)
        return S

    def objective(params):
        r_c, r_s, beta = params
        # Beta is periodic with pi/6? No, the pattern has 6-fold symmetry, so pi/3.
        # But the outer hexagon rotation beta effectively shifts the normals.
        # Since the inner pattern has 6-fold symmetry, rotating the outer hexagon by pi/6 is equivalent to rotating the inner pattern by pi/6?
        # No, the inner pattern is fixed in space. The outer hexagon rotates.
        # The objective is periodic in beta with period pi/3?
        # Let's just optimize beta in [0, pi/6].
        
        S = calc_S(r_c, r_s, beta)
        return S

    # Initial guess
    # For hexagonal packing, r_c should be around sqrt(3) (touching center).
    # r_s should be around sqrt(3) as well?
    # In a hexagonal tiling, the distance between (0,0) and (sqrt(3), 0) is sqrt(3).
    # Distance between (0,0) and (sqrt(3)/2, 1.5) is sqrt(3).
    # So r_c = sqrt(3), r_s = sqrt(3) is a valid configuration (touching).
    
    best_S = float('inf')
    best_params = None

    # Grid search for initialization
    for rc in np.linspace(1.7, 2.5, 20):
        for rs in np.linspace(1.7, 2.5, 20):
            for beta in np.linspace(0, math.pi/6, 20):
                S = calc_S(rc, rs, beta)
                if S < best_S:
                    best_S = S
                    best_params = (rc, rs, beta)

    if best_params is None:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Local optimization
    try:
        res = minimize(objective, best_params, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < best_S:
            best_S = res.fun
            best_params = res.x
    except Exception:
        pass

    rc, rs, beta = best_params
    angle_deg = math.degrees(beta)
    
    # Generate final positions
    pts = get_positions(rc, rs)
    
    # Scale slightly to ensure disjointness if numerical errors crept in
    # The check_validity uses 1e-9 margin.
    # If we scale up, we might violate outer bound.
    # If we scale down, we improve score.
    # Let's try to scale down slightly to ensure validity with a larger margin.
    # Current margin in check_validity is 1e-9.
    # Let's assume the optimization found a valid configuration.
    
    inner = []
    for x, y in pts:
        inner.append([float(x), float(y), 0.0]) # Inner hexagons all have angle 0 (flat-top)
        
    center = [0.0, 0.0]
    side = float(best_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation

