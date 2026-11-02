import math
import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
SQRT3 = math.sqrt(3)

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback to a simple grid
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 10.0, 0.0

    # Define a "corner-filling" template in axial coordinates (q, r).
    # We want to fill the 6 corners of a hexagon of radius R=3 (side length 3 in grid units? No, side length in hexagons).
    # A hexagon of side length L (in units of hex side) contains 3L^2 - 3L + 1 hexagons.
    # For L=3, count = 27 - 9 + 1 = 19. We only have 11.
    # So we take a subset of the L=3 hexagon that fills the corners.
    # Corners of L=3 are at distance 3 from center.
    # Let's generate all hexagons in a hexagon of radius 3 (side length 3).
    # Axial coords (q, r) such that max(|q|, |r|, |q+r|) <= 2? No, distance from center (0,0) is max(|q|, |r|, |q+r|).
    # For side length 3 (radius 2 in some conventions? Let's use standard: side length S has vertices at distance S from center? No.)
    # Standard hex grid: center (0,0). Neighbors at distance 1.
    # A hexagon of side length L (number of hexagons along edge) has radius L-1?
    # Let's just use the set of hexagons that fit in a hexagon of side length 3.
    # The vertices of the outer hexagon (side length 3) are at distance 3*sqrt(3) from center?
    # Let's stick to axial coordinates.
    # A hexagon of side length L contains hexagons with max(|q|, |r|, |q+r|) <= L-1.
    # For L=3, max <= 2. Count = 19.
    # We need 11.
    # Strategy: Pick the 6 corners (distance 2) and 5 others to fill in.
    # Corners: (2,0), (2,-2), (0,-2), (-2,0), (-2,2), (0,2).
    # Let's pick these 6.
    # Remaining 5: Pick (1,0), (0,1), (-1,1), (0,-1), (1,-1)?
    # Let's try to make it symmetric.
    # 6 corners + center + 4 others?
    # Or 6 corners + 5 forming a ring?
    
    # Let's use a specific template that is known to be compact.
    # Template:
    # Row 0 (y=0): q=-1, 0, 1  (3 hexes) -> (-1,0), (0,0), (1,0)
    # Row 1 (y=1.5): q=-2, -1, 0, 1, 2 (5 hexes) -> (-2,1), (-1,1), (0,1), (1,1), (2,1) ? No, axial to cartesian.
    
    # Let's use axial coordinates (q, r) where x = sqrt(3)*(q + r/2), y = 1.5*r.
    # Template points:
    # 1. (0, 0)
    # 2. (1, 0)
    # 3. (-1, 0)
    # 4. (0, 1)
    # 5. (0, -1)
    # 6. (1, -1)
    # 7. (-1, 1)
    # 8. (2, -1)
    # 9. (-2, 1)
    # 10. (1, 1)
    # 11. (-1, -1)
    
    # This is the "flower" of 7 plus 4 more.
    # Let's check if this is valid (non-overlapping). Yes, distinct grid positions.
    
    axial_template = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (2, -1), (-2, 1),
        (1, 1), (-1, -1)
    ]
    
    # Convert to Cartesian
    def axial_to_cartesian(q, r):
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        return np.array([x, y])
    
    pts_template = np.array([axial_to_cartesian(q, r) for q, r in axial_template])
    
    # Center the template
    center_template = np.mean(pts_template, axis=0)
    pts_template -= center_template
    
    # Function to calculate required outer side length S for a given beta (rotation of outer hex)
    # Inner hexes are aligned with the grid (angle 0).
    # Outer hex has rotation beta.
    # The distance from center to the farthest point of the union of inner hexes in the direction of the outer hex's normal.
    
    # Precompute vertices of a unit hexagon (side 1, apothem sqrt(3)/2)
    # Vertices are at radius 1, angles 0, 60, ...
    # Wait, standard orientation: flat top? Or pointy top?
    # The grid used above assumes pointy top (y=1.5*r).
    # For pointy top, vertices are at angles 30, 90, 150, ...
    # Let's use the property that the distance from center to any point in the hexagon in direction (nx, ny) is given by the support function.
    # Support function of a regular hexagon with side 1 (apothem a = sqrt(3)/2) in direction u:
    # h(u) = a / cos(theta) where theta is angle between u and the normal to the nearest side?
    # Actually, simpler: The hexagon is the intersection of 6 half-planes.
    # The distance from center to the boundary in direction u is a / max(|u dot n_i|) where n_i are the 6 unit normals to the sides.
    
    normals_unit = np.array([
        [math.cos(math.pi/6 + k*math.pi/3), math.sin(math.pi/6 + k*math.pi/3)]
        for k in range(6)
    ])
    
    def support_function(pts, u):
        # Max projection of pts onto u, plus the support of the hexagon itself in direction u
        proj = np.max(pts @ u)
        # Support of unit hexagon in direction u
        # The unit hexagon (side 1) has apothem APOTHEM.
        # The distance to the face is APOTHEM / max(|u dot n_i|) ? No.
        # The hexagon is defined by |p dot n_i| <= APOTHEM.
        # So the support in direction u is max_i (APOTHEM / |u dot n_i|) ? No, that's for the dual.
        # Let's just compute the max distance from origin to the hexagon boundary in direction u.
        # It's the minimum t such that t*u is outside.
        # t * |u dot n_i| <= APOTHEM for all i.
        # t <= APOTHEM / |u dot n_i| for all i.
        # So t = min_i (APOTHEM / |u dot n_i|).
        dots = np.abs(normals_unit @ u)
        # Avoid division by zero (shouldn't happen for hex normals)
        t = np.min(APOTHEM / dots)
        return proj + t

    def calc_S(beta):
        # Outer hexagon has rotation beta.
        # Its normals are rotated by beta.
        # We need to find the maximum "radius" of the union of inner hexes in the directions of the outer hex's normals.
        # Then S = (R + APOTHEM) / APOTHEM ?
        # The outer hexagon has apothem A_out = S * APOTHEM.
        # The condition is that the support of the union in direction n_out <= A_out.
        # So A_out = max_i support(union, n_out_i).
        # S = A_out / APOTHEM.
        
        u = np.array([math.cos(beta), math.sin(beta)])
        # The 6 normals of the outer hexagon are rotated by beta relative to the standard normals.
        # Standard normals are at 30, 90, ... degrees.
        # If we rotate the outer hexagon by beta, its normals are at 30+beta, 90+beta, ...
        
        max_dist = 0
        for k in range(6):
            angle = math.pi/6 + k*math.pi/3 + beta
            n = np.array([math.cos(angle), math.sin(angle)])
            dist = support_function(pts_template, n)
            if dist > max_dist:
                max_dist = dist
        
        A_out = max_dist
        S = A_out / APOTHEM
        return S

    # Optimize beta in [0, pi/6) due to 6-fold symmetry
    # Use minimize_scalar
    
    best_S = float('inf')
    best_beta = 0.0
    
    # Grid search first
    for i in range(60):
        beta = i * (math.pi/6) / 60
        S = calc_S(beta)
        if S < best_S:
            best_S = S
            best_beta = beta
    
    # Refine with minimize_scalar
    try:
        res = minimize_scalar(calc_S, bounds=(0, math.pi/6), method='bounded', options={'xatol': 1e-10})
        if res.success and res.fun < best_S:
            best_S = res.fun
            best_beta = res.x
    except Exception:
        pass

    # Construct the result
    # Inner hexagons are at pts_template + center_template (which was 0)
    # But we should center the whole thing properly.
    # The "center" of the outer hexagon should be the origin (0,0) in our coordinate system.
    # The inner hexagons are at pts_template.
    
    # Wait, the template was centered. So the centroid of the inner hexagons is at (0,0).
    # Is the optimal outer hexagon center at (0,0)?
    # Due to symmetry, yes.
    
    inner = []
    for p in pts_template:
        inner.append([float(p[0]), float(p[1]), 0.0])
    
    center = [0.0, 0.0]
    side = float(best_S)
    rotation = float(math.degrees(best_beta))
    
    return inner, center, side, rotation
