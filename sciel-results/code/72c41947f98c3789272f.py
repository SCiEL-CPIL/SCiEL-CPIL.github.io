"""Improved program for packing 11 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import math

N = 11

def get_hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon."""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        a = angle_rad + i * math.pi / 3
        x = cx + side * math.cos(a)
        y = cy + side * math.sin(a)
        vertices.append((x, y))
    return vertices

def min_dist_point_to_segment(px, py, x1, y1, x2, y2):
    """Calculate minimum distance from point to segment."""
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx*dx + dy*dy
    if length_sq == 0:
        return math.hypot(px - x1, py - y1)
    
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = max(0, min(1, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)

def hexagon_distance(poly1, poly2):
    """
    Calculate the minimum distance between two polygons.
    Returns 0 if they intersect or touch.
    Returns positive distance if disjoint.
    """
    min_d = float('inf')
    n1 = len(poly1)
    n2 = len(poly2)
    
    # Check distance from vertices of poly1 to edges of poly2
    for i in range(n1):
        px, py = poly1[i]
        for j in range(n2):
            x1, y1 = poly2[j]
            x2, y2 = poly2[(j+1)%n2]
            d = min_dist_point_to_segment(px, py, x1, y1, x2, y2)
            if d < min_d:
                min_d = d
                if min_d < 1e-9:
                    return 0.0
                    
    # Check distance from vertices of poly2 to edges of poly1
    for i in range(n2):
        px, py = poly2[i]
        for j in range(n1):
            x1, y1 = poly1[j]
            x2, y2 = poly1[(j+1)%n1]
            d = min_dist_point_to_segment(px, py, x1, y1, x2, y2)
            if d < min_d:
                min_d = d
                if min_d < 1e-9:
                    return 0.0
                    
    return min_d

def compute_objective(params):
    """
    Compute objective function: outer side length + penalties.
    params: [x1, y1, a1, ..., x11, y11, a11, cx, cy, side, angle]
    """
    N_local = 11
    inner_params = params[:3*N_local]
    outer_cx = params[3*N_local]
    outer_cy = params[3*N_local+1]
    outer_side = params[3*N_local+2]
    outer_angle = params[3*N_local+3]
    
    if outer_side <= 0:
        return 1e10
        
    # Precompute outer hexagon vertices and normals for containment check
    outer_verts = get_hexagon_vertices(outer_cx, outer_cy, outer_side, outer_angle)
    
    # Calculate containment penalties
    # For each inner hexagon, check if its vertices are inside the outer hexagon.
    # Inside condition: For a regular hexagon, we can use the 6 half-planes.
    # However, using the precomputed vertices and a point-in-polygon test is also fine,
    # but for smoothness, we use the distance to the boundary.
    # If a vertex is outside, the distance to the polygon boundary is negative (in signed sense).
    # We will use the minimum distance from the vertex to the outer polygon edges.
    # If the vertex is inside, distance is positive. If outside, distance is negative?
    # Actually, min_dist_point_to_segment gives unsigned distance.
    # We need to know if it's inside.
    
    # Efficient containment check for regular hexagon:
    # The outer hexagon is the intersection of 6 half-planes.
    # Normal vectors for the outer hexagon (pointing inward):
    # The edges are perpendicular to the directions 0, 60, 120, 180, 240, 300 degrees + outer_angle.
    # Let's compute the 6 supporting lines.
    # Side length S, inradius r = S * sqrt(3) / 2.
    r_outer = outer_side * math.sqrt(3) / 2.0
    base_angles = [0, 60, 120, 180, 240, 300]
    normals = []
    for deg in base_angles:
        rad = math.radians(deg + outer_angle)
        # Normal pointing inward from the edge center
        # Edge center is at distance r from center.
        # Normal direction is rad.
        normals.append((math.cos(rad), math.sin(rad)))
        
    containment_penalty = 0.0
    inner_polys = []
    
    for i in range(N_local):
        ix = inner_params[3*i]
        iy = inner_params[3*i+1]
        iangle = inner_params[3*i+2]
        
        poly = get_hexagon_vertices(ix, iy, 1.0, iangle)
        inner_polys.append(poly)
        
        # Check containment of each vertex of the inner hex
        for vx, vy in poly:
            # Check against each of the 6 half-planes
            for nx, ny in normals:
                # Signed distance from point (vx, vy) to the line defined by normal (nx, ny) passing through (outer_cx, outer_cy) offset by r?
                # The line equation: nx * (x - outer_cx) + ny * (y - outer_cy) = r_outer
                # We want nx * (x - outer_cx) + ny * (y - outer_cy) <= r_outer
                val = nx * (vx - outer_cx) + ny * (vy - outer_cy)
                if val > r_outer:
                    violation = val - r_outer
                    containment_penalty += violation * violation
    
    # Calculate overlap penalties
    overlap_penalty = 0.0
    for i in range(N_local):
        for j in range(i+1, N_local):
            d = hexagon_distance(inner_polys[i], inner_polys[j])
            # We want d >= 0. If d < 0 (which shouldn't happen with unsigned dist, but if they intersect dist is 0), 
            # we need a penalty. 
            # The function hexagon_distance returns 0 if they intersect.
            # This creates a non-smooth cusp at d=0.
            # To make it smooth, we can use a soft penalty if they are "close" or intersecting.
            # However, standard Nelder-Mead handles non-smooth functions reasonably well if the initial guess is good.
            # A better smooth penalty:
            # If we could get signed distance, we'd use max(0, -d)^2.
            # Since we only have unsigned min distance, we can't easily distinguish "close" from "intersecting" with a single value if we just return 0.
            # Let's modify hexagon_distance to return a signed distance?
            # That's complex.
            # Alternative: If distance is 0, assume they are touching or overlapping.
            # We add a small penalty if distance is very small to encourage separation?
            # No, we want to allow touching.
            # The issue is that if they overlap, the distance is 0, same as if they touch.
            # The optimizer might get stuck in overlapping states because the penalty doesn't increase as they overlap more.
            
            # To fix this, we can check for intersection explicitly and add a large penalty if they intersect.
            # But that makes it non-smooth.
            
            # Let's use a different metric: 
            # The minimum distance between the *centers* minus the sum of their "radii"? No, hexagons are not circles.
            
            # Let's stick to the unsigned distance but add a heuristic:
            # If distance is 0, we check if they actually intersect using a quick check.
            # If they intersect, we add a penalty based on the area of intersection? Too slow.
            
            # Let's assume the optimizer will find a configuration where they are disjoint or touching.
            # To push them apart, we can penalize small distances? No, we want them close.
            
            # Actually, if they overlap, the "distance" between vertices and edges will still be 0.
            # The problem is that the objective function is flat in the overlap region.
            # We need to detect overlap.
            
            # Let's implement a simple overlap check inside the loop.
            # If they overlap, add a penalty proportional to the minimum "penetration depth".
            # Approximate penetration depth:
            # For each vertex of poly1 inside poly2, add the distance to the boundary of poly2.
            # This is essentially what the signed distance would give.
            
            # Let's refine hexagon_distance to return a "signed" distance estimate.
            # If a vertex of poly1 is inside poly2, the distance is negative (penetration).
            
    # Re-implementing the distance function to be signed for better optimization
    # This will be done in a helper function below.
    
    return outer_side + 50.0 * (containment_penalty + overlap_penalty)

def signed_hex_distance(poly1, poly2):
    """
    Returns a signed distance between two polygons.
    Positive if disjoint.
    Negative if overlapping (approximated by max penetration depth).
    """
    # First, check if they are disjoint using unsigned distance
    # If unsigned distance > 0, return it.
    # If unsigned distance == 0, check for overlap.
    
    # To be efficient, we can just compute the max penetration depth directly.
    # Penetration depth is the minimum distance one polygon must move to be disjoint.
    # For convex polygons, this is the minimum over all axes of the gap.
    
    # Simpler approach for optimization:
    # Check if any vertex of poly1 is inside poly2.
    # If yes, they overlap. The "distance" is negative.
    # We can estimate the negative distance as the minimum distance from that vertex to the boundary of poly2.
    
    min_unsigned_dist = float('inf')
    overlap_depth = 0.0
    
    n1 = len(poly1)
    n2 = len(poly2)
    
    # Check vertices of poly1 against poly2
    for i in range(n1):
        px, py = poly1[i]
        # Check if inside poly2
        inside = False
        j = n2 - 1
        for k in range(n2):
            x1, y1 = poly2[k]
            x2, y2 = poly2[j]
            if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1) + x1):
                inside = not inside
            j = k
            
        if inside:
            # Calculate distance to boundary of poly2
            min_dist_to_boundary = float('inf')
            for k in range(n2):
                x1, y1 = poly2[k]
                x2, y2 = poly2[(k+1)%n2]
                d = min_dist_point_to_segment(px, py, x1, y1, x2, y2)
                if d < min_dist_to_boundary:
                    min_dist_to_boundary = d
            overlap_depth = max(overlap_depth, min_dist_to_boundary)
        else:
            # Not inside, calculate distance to boundary
            min_dist_to_boundary = float('inf')
            for k in range(n2):
                x1, y1 = poly2[k]
                x2, y2 = poly2[(k+1)%n2]
                d = min_dist_point_to_segment(px, py, x1, y1, x2, y2)
                if d < min_dist_to_boundary:
                    min_dist_to_boundary = d
            min_unsigned_dist = min(min_unsigned_dist, min_dist_to_boundary)
            
    # Check vertices of poly2 against poly1
    for i in range(n2):
        px, py = poly2[i]
        inside = False
        j = n1 - 1
        for k in range(n1):
            x1, y1 = poly1[k]
            x2, y2 = poly1[j]
            if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1) + x1):
                inside = not inside
            j = k
            
        if inside:
            min_dist_to_boundary = float('inf')
            for k in range(n1):
                x1, y1 = poly1[k]
                x2, y2 = poly1[(k+1)%n1]
                d = min_dist_point_to_segment(px, py, x1, y1, x2, y2)
                if d < min_dist_to_boundary:
                    min_dist_to_boundary = d
            overlap_depth = max(overlap_depth, min_dist_to_boundary)
        else:
            min_dist_to_boundary = float('inf')
            for k in range(n1):
                x1, y1 = poly1[k]
                x2, y2 = poly1[(k+1)%n1]
                d = min_dist_point_to_segment(px, py, x1, y1, x2, y2)
                if d < min_dist_to_boundary:
                    min_dist_to_boundary = d
            min_unsigned_dist = min(min_unsigned_dist, min_dist_to_boundary)
            
    if overlap_depth > 0:
        return -overlap_depth
    else:
        return min_unsigned_dist

def compute_objective_v2(params):
    N_local = 11
    inner_params = params[:3*N_local]
    outer_cx = params[3*N_local]
    outer_cy = params[3*N_local+1]
    outer_side = params[3*N_local+2]
    outer_angle = params[3*N_local+3]
    
    if outer_side <= 0:
        return 1e10
        
    r_outer = outer_side * math.sqrt(3) / 2.0
    base_angles = [0, 60, 120, 180, 240, 300]
    normals = []
    for deg in base_angles:
        rad = math.radians(deg + outer_angle)
        normals.append((math.cos(rad), math.sin(rad)))
        
    containment_penalty = 0.0
    inner_polys = []
    
    for i in range(N_local):
        ix = inner_params[3*i]
        iy = inner_params[3*i+1]
        iangle = inner_params[3*i+2]
        
        poly = get_hexagon_vertices(ix, iy, 1.0, iangle)
        inner_polys.append(poly)
        
        for vx, vy in poly:
            for nx, ny in normals:
                val = nx * (vx - outer_cx) + ny * (vy - outer_cy)
                if val > r_outer:
                    violation = val - r_outer
                    containment_penalty += violation * violation
    
    overlap_penalty = 0.0
    for i in range(N_local):
        for j in range(i+1, N_local):
            d = signed_hex_distance(inner_polys[i], inner_polys[j])
            if d < 0:
                overlap_penalty += d * d
                
    return outer_side + 50.0 * (containment_penalty + overlap_penalty)

def construct():
    N_local = 11
    
    # Initialization: Asymmetric "defect" cluster
    # Central hexagon
    init = [(0.0, 0.0, 0.0)]
    
    # 6 neighbors
    for i in range(6):
        a = i * math.pi / 3
        r = math.sqrt(3)
        init.append((r * math.cos(a), r * math.sin(a), 0.0))
        
    # 4 more hexagons, placed asymmetrically
    # Try placing them at angles that break the 6-fold symmetry
    # e.g., 15, 75, 135, 195 degrees?
    # Let's try a configuration that extends the lattice slightly
    # Distance for next shell in lattice is 2*sqrt(3) ~ 3.46
    # Let's place them at r=3.2 to allow some compression
    angles = [math.radians(15), math.radians(75), math.radians(135), math.radians(195)]
    for a in angles:
        r = 3.2
        init.append((r * math.cos(a), r * math.sin(a), 0.0))
        
    outer_cx = 0.0
    outer_cy = 0.0
    outer_side = 4.5
    outer_angle = 0.0
    
    x0 = np.zeros(3*N_local + 4)
    for i in range(N_local):
        x0[3*i] = init[i][0]
        x0[3*i+1] = init[i][1]
        x0[3*i+2] = init[i][2]
    x0[3*N_local] = outer_cx
    x0[3*N_local+1] = outer_cy
    x0[3*N_local+2] = outer_side
    x0[3*N_local+3] = outer_angle
    
    best_params = x0.copy()
    best_val = compute_objective_v2(x0)
    
    # Run optimization
    options = {'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4}
    
    try:
        result = minimize(compute_objective_v2, x0, method='Nelder-Mead', options=options)
        if result.fun < best_val:
            best_val = result.fun
            best_params = result.x
    except Exception:
        pass
        
    # Second run with different initial rotation to break symmetry further
    x0_2 = x0.copy()
    for i in range(N_local):
        x0_2[3*i+2] += 15.0 # Rotate all by 15 deg
    x0_2[3*N_local+2] = 4.2
    
    try:
        result2 = minimize(compute_objective_v2, x0_2, method='Nelder-Mead', options=options)
        if result2.fun < best_val:
            best_val = result2.fun
            best_params = result2.x
    except Exception:
        pass
        
    inner = []
    for i in range(N_local):
        ix = best_params[3*i]
        iy = best_params[3*i+1]
        iangle = best_params[3*i+2]
        inner.append([ix, iy, iangle])
    
    outer_cx_best = best_params[3*N_local]
    outer_cy_best = best_params[3*N_local+1]
    outer_side_best = best_params[3*N_local+2]
    outer_angle_best = best_params[3*N_local+3]
    
    # Final validation
    outer_poly = get_hexagon_vertices(outer_cx_best, outer_cy_best, outer_side_best, outer_angle_best)
    inner_polys = []
    valid = True
    for i in range(N_local):
        poly = get_hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        inner_polys.append(poly)
        
        # Check containment
        r_outer = outer_side_best * math.sqrt(3) / 2.0
        base_angles = [0, 60, 120, 180, 240, 300]
        normals = []
        for deg in base_angles:
            rad = math.radians(deg + outer_angle_best)
            normals.append((math.cos(rad), math.sin(rad)))
            
        for vx, vy in poly:
            for nx, ny in normals:
                val = nx * (vx - outer_cx_best) + ny * (vy - outer_cy_best)
                if val > r_outer + 1e-6: # Small tolerance
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if valid:
        for i in range(N_local):
            for j in range(i+1, N_local):
                d = signed_hex_distance(inner_polys[i], inner_polys[j])
                if d < -1e-6: # Overlap
                    valid = False
                    break
            if not valid: break

    if not valid:
        # Fallback to a known valid configuration (loose packing)
        fallback_inner = []
        for i in range(N_local):
            row = i // 3
            col = i % 3
            x = col * 2.0 - 2.0
            y = row * 1.8 - 1.8
            fallback_inner.append([x, y, 0.0])
        return fallback_inner, (0.0, 0.0), 10.0, 0.0
        
    return inner, (outer_cx_best, outer_cy_best), outer_side_best, outer_angle_best

# EVOLVE-BLOCK-START
# (construct function defined above)
# EVOLVE-BLOCK-END
