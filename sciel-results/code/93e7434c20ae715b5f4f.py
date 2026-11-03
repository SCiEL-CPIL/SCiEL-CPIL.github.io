"""Rewritten program for packing 11 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import math

N = 11

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        a = angle_rad + i * math.pi / 3
        x = cx + side * math.cos(a)
        y = cy + side * math.sin(a)
        vertices.append((x, y))
    return vertices

def point_in_polygon(px, py, polygon):
    """Ray casting algorithm to check if point is inside polygon."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def compute_objective_fast(params, N):
    """
    Fast objective function using distance constraints for non-overlap (approx) 
    and vertex containment for outer boundary.
    Assumes inner hexagons are aligned (angle 0) for the distance check validity.
    params: [x1, y1, ..., xN, yN, outer_cx, outer_cy, outer_side, outer_angle]
    """
    inner_centers = params[:2*N].reshape(N, 2)
    outer_cx = params[2*N]
    outer_cy = params[2*N+1]
    outer_side = params[2*N+2]
    outer_angle = params[2*N+3]
    
    if outer_side < 0.5:
        return 1e10
        
    # 1. Check outer containment
    # To speed up, we can check the 6 vertices of each inner hexagon
    # Since N=11, 66 points. This is fast enough.
    outer_poly = hexagon_vertices(outer_cx, outer_cy, outer_side, outer_angle)
    
    penalty = 0.0
    for i in range(N):
        cx, cy = inner_centers[i]
        # Check vertices of inner hex (side 1, angle 0)
        # Vertices of unit hex at (cx, cy) with angle 0:
        # (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
        # Actually, standard hex vertices at angle 0 (flat top or pointy top?)
        # The code above uses: a = i * pi/3. 
        # i=0: (side, 0) -> Pointy right.
        # So vertices are at angles 0, 60, 120...
        for k in range(6):
            a = k * math.pi / 3
            vx = cx + 1.0 * math.cos(a)
            vy = cy + 1.0 * math.sin(a)
            if not point_in_polygon(vx, vy, outer_poly):
                # Calculate distance to outside? Just penalize heavily.
                penalty += 100.0
    
    if penalty > 1e-6:
        return outer_side + penalty
        
    # 2. Check non-overlap using center distance
    # For two aligned unit hexagons, min distance is sqrt(3) ~ 1.732
    # If they are rotated, distance can be smaller, but we assume near-aligned optimal.
    min_dist = math.sqrt(3) - 0.01 # Small epsilon
    for i in range(N):
        for j in range(i+1, N):
            dx = inner_centers[i][0] - inner_centers[j][0]
            dy = inner_centers[i][1] - inner_centers[j][1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                penalty += (min_dist - dist) * 1000.0
                
    if penalty > 1e-6:
        return outer_side + penalty
        
    return outer_side

def construct():
    """
    Construct 11 unit hexagons packed in a regular hexagon, minimizing outer side length.
    Returns: (inner, center, side, angle)
    """
    N_local = 11
    sqrt3 = math.sqrt(3)
    
    # Initial guess: Hexagonal cluster
    # 1 center, 6 around, 4 more at corners of the "hexagon of hexes"
    # This forms a shape close to a regular hexagon.
    
    init_centers = []
    # Center
    init_centers.append((0.0, 0.0))
    # Ring 1 (6 hexes)
    for i in range(6):
        a = i * math.pi / 3
        init_centers.append((sqrt3 * math.cos(a), sqrt3 * math.sin(a)))
    # Ring 2 partial (4 hexes) - placed at 60, 120, 240, 300 degrees? 
    # A full ring 2 has 12. We need 4 more.
    # Let's place them at the "corners" of the hexagonal lattice to make it compact.
    # The corners of the hexagonal lattice of radius 2 are at distance 2*sqrt(3).
    # Angles: 0, 60, 120, 180, 240, 300.
    # We have 6 spots in ring 1.
    # Let's add 4 at: 0, 120, 240, 300? Or 30, 90...?
    # To minimize bounding box, we want symmetry.
    # Let's try placing them at 0, 120, 240, 300 degrees at distance 2*sqrt(3).
    for i in [0, 120, 240, 300]:
        a = math.radians(i)
        init_centers.append((2*sqrt3 * math.cos(a), 2*sqrt3 * math.sin(a)))
        
    # Total 1 + 6 + 4 = 11.
    
    # Outer hexagon initial guess
    # The extent is roughly 2*sqrt(3) + 1 from center to edge?
    # Max x is 2*sqrt(3) + 1.
    # Side length of outer hex approx max_extent / cos(30) ?
    # Let's start with side = 4.0
    outer_cx = 0.0
    outer_cy = 0.0
    outer_side = 4.0
    outer_angle = 0.0
    
    # Build initial parameter vector
    x0 = np.zeros(2*N_local + 4)
    for i in range(N_local):
        x0[2*i] = init_centers[i][0]
        x0[2*i+1] = init_centers[i][1]
    x0[2*N_local] = outer_cx
    x0[2*N_local+1] = outer_cy
    x0[2*N_local+2] = outer_side
    x0[2*N_local+3] = outer_angle
    
    # Optimize
    best_params = x0.copy()
    best_val = compute_objective_fast(x0, N_local)
    
    # Try a few restarts with slight perturbations
    np.random.seed(42)
    for restart in range(3):
        if restart == 0:
            x0_perturbed = x0
        else:
            x0_perturbed = x0.copy()
            # Perturb inner centers slightly
            noise = np.random.normal(0, 0.1, size=2*N_local)
            x0_perturbed[:2*N_local] += noise
            # Perturb outer side slightly
            x0_perturbed[2*N_local+2] += np.random.normal(0, 0.1)
            
        try:
            result = minimize(
                compute_objective_fast, 
                x0_perturbed, 
                args=(N_local,),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            val = result.fun
            if val < best_val:
                best_val = val
                best_params = result.x
        except Exception:
            pass
            
    # Extract best parameters
    inner = []
    for i in range(N_local):
        ix = best_params[2*i]
        iy = best_params[2*i+1]
        inner.append([ix, iy, 0.0]) # Angle 0
        
    outer_cx_best = best_params[2*N_local]
    outer_cy_best = best_params[2*N_local+1]
    outer_side_best = best_params[2*N_local+2]
    outer_angle_best = best_params[2*N_local+3]
    
    # Ensure validity
    if outer_side_best < 3.0:
         outer_side_best = 3.0
         
    # Final check with strict polygon intersection to ensure validity
    # If invalid, fallback to a known valid configuration
    outer_poly = hexagon_vertices(outer_cx_best, outer_cy_best, outer_side_best, outer_angle_best)
    inner_polys = []
    valid = True
    for i in range(N_local):
        poly = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        inner_polys.append(poly)
        if not check_inside(poly, outer_poly):
            valid = False
            break
            
    if valid:
        for i in range(N_local):
            for j in range(i+1, N_local):
                if polygons_intersect(inner_polys[i], inner_polys[j]):
                    valid = False
                    break
            if not valid: break
            
    if not valid:
        # Fallback: simple grid
        # 3x3 grid of hexes? No, 11.
        # Just return a large valid box
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N_local // 2), N_local - N_local // 2)], (0.0, 0.0), 40.0, 0.0
        
    return inner, (outer_cx_best, outer_cy_best), outer_side_best, outer_angle_best

def check_inside(inner_hex, outer_poly):
    """Check if all vertices of inner hexagon are inside outer polygon."""
    for v in inner_hex:
        if not point_in_polygon(v[0], v[1], outer_poly):
            return False
    return True

def segment_intersection(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def polygons_intersect(poly1, poly2):
    """Check if two polygons intersect (including one containing the other)."""
    # Check if any vertex of poly1 is inside poly2
    for v in poly1:
        if point_in_polygon(v[0], v[1], poly2):
            return True
    # Check if any vertex of poly2 is inside poly1
    for v in poly2:
        if point_in_polygon(v[0], v[1], poly1):
            return True
    # Check for edge intersections
    n1 = len(poly1)
    n2 = len(poly2)
    for i in range(n1):
        for j in range(n2):
            if segment_intersection(poly1[i], poly1[(i+1)%n1], poly2[j], poly2[(j+1)%n2]):
                return True
    return False

# EVOLVE-BLOCK-START
# (construct function defined above)
# EVOLVE-BLOCK-END
