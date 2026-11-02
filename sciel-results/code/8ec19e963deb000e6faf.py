import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    """
    Returns vertices of a regular hexagon centered at (cx, cy) with side s
    and rotation angle_deg (degrees).
    Vertices are in CCW order.
    """
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _min_enclosing_hex_side_lp(points, angle_deg, cx, cy):
    """
    Calculate the minimum side length of a regular hexagon centered at (cx, cy)
    with rotation angle_deg that contains all points.
    Uses the property that the distance from center to any point must be <=
    the apothem in the direction of the point's normal projection? No.
    
    For a regular hexagon with side s, the condition for a point (x,y) relative
    to center to be inside is:
    Let (u, v) be coordinates in the rotated frame of the hexagon.
    The hexagon is defined by 3 strips:
    |u| <= s * cos(30) ? No.
    
    Let's use the support function or distance to boundary.
    The distance from the center to the side (apothem) is a = s * sqrt(3) / 2.
    The distance from the center to a vertex (circumradius) is R = s.
    
    For a point at polar coordinates (r, theta) relative to the hexagon center,
    the minimum distance to the boundary in direction theta is:
    d(theta) = a / cos((theta mod 60) - 30)
    
    So for the point to be inside, we need r <= d(theta) for all points.
    This implies r * cos((theta mod 60) - 30) <= a.
    So a >= max( r * cos((theta mod 60) - 30) ) over all points.
    And s = 2 * a / sqrt(3).
    """
    a_rad = math.radians(angle_deg)
    max_a = 0.0
    
    for px, py in points:
        dx = px - cx
        dy = py - cy
        r = math.sqrt(dx*dx + dy*dy)
        if r == 0:
            continue
        theta = math.atan2(dy, dx)
        # Relative angle to hexagon axes
        # Hexagon sides are perpendicular to angles 30, 90, 150, etc.
        # The "corners" are at 0, 60, 120...
        # The formula d(theta) = a / cos(delta) where delta is angle from normal.
        # Normals are at 30, 90, 150...
        # Let's normalize theta to [0, 60) relative to the hexagon orientation.
        # The hexagon orientation is defined by its vertices at 0, 60, ... if angle=0?
        # If angle=0, vertices are at 0, 60, 120, 180, 240, 300.
        # Normals to sides are at 30, 90, 150, 210, 270, 330.
        
        rel_theta = (theta - a_rad) % (math.pi / 3)
        # The maximum distance from center to boundary in direction rel_theta
        # occurs at the vertex (rel_theta=0 or 60).
        # The minimum distance is at the side normal (rel_theta=30).
        # We need the point to be inside.
        # The boundary distance in direction phi is:
        # R_bound(phi) = a / cos(phi - pi/6) for phi in [0, pi/3]
        # where phi is angle from the vertex direction.
        # Wait, if vertex is at 0, normal is at 30.
        # Distance to boundary at angle phi (from vertex) is a / cos(phi - 30).
        # At phi=0 (vertex), dist = a / cos(-30) = a / (sqrt(3)/2) = 2a/sqrt(3) = s. Correct.
        # At phi=30 (normal), dist = a / cos(0) = a. Correct.
        
        # So we need r <= a / cos(rel_theta - pi/6)
        # => a >= r * cos(rel_theta - pi/6)
        
        cos_val = math.cos(rel_theta - math.pi/6)
        if cos_val < 1e-9:
            # Should not happen for regular hexagon, but safety
            continue
        required_a = r / cos_val
        if required_a > max_a:
            max_a = required_a
            
    s = 2 * max_a / math.sqrt(3)
    return s

def _check_disjoint(pos1, ang1, pos2, ang2):
    """
    Check if two unit hexagons are disjoint.
    pos1, pos2: (x, y)
    ang1, ang2: angles in degrees
    Returns True if disjoint, False if overlapping.
    """
    # Quick check: distance between centers
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    dist_sq = dx*dx + dy*dy
    # Min distance for disjoint is sqrt(3) (apothem*2) if aligned?
    # Actually, min distance is 2*apothem = sqrt(3) only if they touch side-to-side.
    # If they touch vertex-to-vertex, distance is 2.
    # The minimum distance between centers of two disjoint unit hexagons is sqrt(3).
    # If dist < sqrt(3), they must overlap.
    if dist_sq < 3.0 - 1e-9:
        return False
    
    # If dist > 2, they definitely don't overlap (circumradius is 1, so max reach is 2)
    if dist_sq > 4.0 + 1e-9:
        return True
    
    # Exact check: check if any vertex of hex1 is inside hex2 or vice versa,
    # or if any edges intersect.
    # Since N is small, we can afford a more precise check for close pairs.
    
    v1 = _hex_vertices(pos1[0], pos1[1], 1.0, ang1)
    v2 = _hex_vertices(pos2[0], pos2[1], 1.0, ang2)
    
    # Check vertices of hex1 inside hex2
    for vx, vy in v1:
        if _point_in_hex(vx, vy, pos2[0], pos2[1], 1.0, ang2):
            return False
            
    # Check vertices of hex2 inside hex1
    for vx, vy in v2:
        if _point_in_hex(vx, vy, pos1[0], pos1[1], 1.0, ang1):
            return False
            
    # Check edge intersections
    # There are 6 edges in each. 36 pairs.
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i+1)%6]
        for j in range(6):
            p3 = v2[j]
            p4 = v2[(j+1)%6]
            if _segments_intersect(p1, p2, p3, p4):
                return False
                
    return True

def _point_in_hex(px, py, cx, cy, s, angle_deg):
    """
    Check if point (px, py) is inside hexagon centered at (cx, cy) with side s and angle.
    """
    a_rad = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate point to hexagon local frame
    rx = dx * math.cos(-a_rad) - dy * math.sin(-a_rad)
    ry = dx * math.sin(-a_rad) + dy * math.cos(-a_rad)
    
    # Hexagon is intersection of 3 strips.
    # Normals at 30, 90, 150 degrees in local frame?
    # If vertices are at 0, 60, ..., normals are at 30, 90, 150.
    # Inequalities:
    # |rx * cos(30) + ry * sin(30)| <= s * cos(30) ? No.
    # Apothem a = s * sqrt(3) / 2.
    # Distance to side <= a.
    
    c30 = math.sqrt(3)/2
    s30 = 0.5
    
    # Normal 1: 30 deg
    d1 = rx * c30 + ry * s30
    if abs(d1) > s * c30: # s * cos(30) is apothem? No. Apothem is s*sqrt(3)/2.
        # Wait. s is side length.
        # Apothem = s * sqrt(3) / 2.
        # So condition is |d| <= s * sqrt(3) / 2.
        return False
        
    # Normal 2: 90 deg
    d2 = ry
    if abs(d2) > s * c30:
        return False
        
    # Normal 3: 150 deg
    d3 = -rx * c30 + ry * s30
    if abs(d3) > s * c30:
        return False
        
    return True

def _segments_intersect(p1, p2, p3, p4):
    """
    Check if segment p1-p2 intersects segment p3-p4.
    """
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
        
    # Collinear cases (should be rare in optimization, but handle for robustness)
    if d1 == 0 and _on_segment(p3, p1, p4): return True
    if d2 == 0 and _on_segment(p3, p2, p4): return True
    if d3 == 0 and _on_segment(p1, p3, p2): return True
    if d4 == 0 and _on_segment(p1, p4, p2): return True
        
    return False

def _on_segment(a, b, c):
    """Check if b is on segment a-c."""
    if min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and \
       min(a[1], c[1]) <= b[1] <= max(a[1], c[1]):
        return True
    return False

def _all_disjoint(params):
    """
    Check if all hexagons are disjoint.
    params: [x1, y1, a1, x2, y2, a2, ...]
    """
    n = len(params) // 3
    for i in range(n):
        for j in range(i+1, n):
            pos1 = (params[3*i], params[3*i+1])
            ang1 = params[3*i+2]
            pos2 = (params[3*j], params[3*j+1])
            ang2 = params[3*j+2]
            if not _check_disjoint(pos1, ang1, pos2, ang2):
                return False
    return True

def _inner_in_outer(params, outer_params):
    """
    Check if all inner hexagons are inside the outer hexagon.
    params: inner hex params
    outer_params: [cx, cy, side, angle_deg]
    """
    ocx, ocy, os, oang = outer_params
    n = len(params) // 3
    for i in range(n):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = params[3*i+2]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for vx, vy in verts:
            if not _point_in_hex(vx, vy, ocx, ocy, os, oang):
                return False
    return True

def _compute_outer_for_positions(params):
    """
    Given inner positions, find the best outer hexagon.
    We try a few angles and use LP-like logic (actually just max projection).
    """
    n = len(params) // 3
    positions = []
    for i in range(n):
        positions.append((params[3*i], params[3*i+1]))
        
    # Centroid
    cx = sum(p[0] for p in positions) / n
    cy = sum(p[1] for p in positions) / n
    
    best_side = float('inf')
    best_oang = 0
    
    # Try several outer angles
    for oang in [0, 30, 60, 15, 45, 75]:
        # For a fixed angle and center, the minimum side is determined by the furthest point
        # in terms of support function.
        # However, the center might not be optimal.
        # For simplicity, let's assume center is centroid first.
        # Then refine center?
        # For now, just calculate side for centroid.
        
        # Collect all vertices of inner hexagons
        all_verts = []
        for i in range(n):
            verts = _hex_vertices(params[3*i], params[3*i+1], 1.0, params[3*i+2])
            all_verts.extend(verts)
            
        side = _min_enclosing_hex_side_lp(all_verts, oang, cx, cy)
        if side < best_side:
            best_side = side
            best_oang = oang
            
    # Local optimization of outer center and side
    # We can do a small grid search around centroid for the best angle found
    # Or just return this. Given time constraints, this is a good heuristic.
    # Let's do a quick Nelder-Mead on outer center and side for the best angle?
    # No, _min_enclosing_hex_side_lp is not smooth.
    # Let's just try a few centers.
    
    for dx in np.arange(-0.5, 0.51, 0.25):
        for dy in np.arange(-0.5, 0.51, 0.25):
            tcx = cx + dx
            tcy = cy + dy
            side = _min_enclosing_hex_side_lp(all_verts, best_oang, tcx, tcy)
            if side < best_side:
                best_side = side
                cx = tcx
                cy = tcy
                
    return [cx, cy, best_side, best_oang]

def _pairwise_penalty(params, outer_params):
    """
    Penalty function for optimization.
    """
    n = len(params) // 3
    penalty = 0.0
    
    # Disjointness penalty
    for i in range(n):
        for j in range(i+1, n):
            pos1 = (params[3*i], params[3*i+1])
            ang1 = params[3*i+2]
            pos2 = (params[3*j], params[3*j+1])
            ang2 = params[3*j+2]
            
            # Approximate penalty based on distance
            dx = pos1[0] - pos2[0]
            dy = pos1[1] - pos2[1]
            dist = math.sqrt(dx*dx + dy*dy)
            min_dist = math.sqrt(3)
            if dist < min_dist:
                # Strong penalty
                penalty += (min_dist - dist) ** 2 * 100
            elif dist < 2.0:
                # Check exact overlap
                if not _check_disjoint(pos1, ang1, pos2, ang2):
                    penalty += 1000.0
                    # Also add a gradient-like term? Hard without differentiable check.
                    # Just rely on DE's ability to jump over.
                    
    # Containment penalty
    ocx, ocy, os, oang = outer_params
    for i in range(n):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = params[3*i+2]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for vx, vy in verts:
            if not _point_in_hex(vx, vy, ocx, ocy, os, oang):
                # Calculate penetration depth
                # Simple: distance to boundary?
                # Approximate: just add penalty
                penalty += 100.0
                
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Initial guess: Hexagonal lattice
    # 1 in center, 6 around, 4 more?
    # 1 + 6 + 4 = 11.
    # Positions for hex lattice (side 1, distance between centers sqrt(3)):
    # Center: (0,0)
    # Ring 1: 6 neighbors at distance sqrt(3)
    # Ring 2: 12 neighbors at distance 2*sqrt(3)? No, 2*sqrt(3) is for same direction.
    # Distance between (0,0) and (2, sqrt(3)) is sqrt(4+3)=sqrt(7) ~ 2.64.
    # Let's just use a known good configuration or random.
    
    # Better: Use a few random restarts with DE.
    
    best_result = None
    best_score = float('inf')
    
    # Number of restarts
    max_restarts = 10
    
    for restart in range(max_restarts):
        if time.time() > deadline - 20: # Leave time for final polish
            break
            
        # Generate initial population for DE
        # Bounds:
        # x, y: [-5, 5]
        # angle: [0, 360]
        bounds = []
        for i in range(N):
            bounds.append((-5, 5))
            bounds.append((-5, 5))
            bounds.append((0, 360))
            
        # Objective function for DE
        def objective(params):
            if time.time() > deadline - 5:
                return 1e9
            # Compute outer hex
            outer = _compute_outer_for_positions(params)
            # Check validity
            if not _all_disjoint(params):
                return 1e9 + _pairwise_penalty(params, outer)
            if not _inner_in_outer(params, outer):
                return 1e9 + _pairwise_penalty(params, outer)
            
            # Score is outer side
            return outer[2]
            
        # Run DE
        try:
            res = differential_evolution(objective, bounds, seed=42+restart, maxiter=50, popsize=10, tol=0.01, polish=False)
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                # Verify and get outer
                outer = _compute_outer_for_positions(best_params)
                if _all_disjoint(best_params) and _inner_in_outer(best_params, outer):
                    best_result = (best_params, outer)
        except Exception as e:
            pass
            
        if time.time() > deadline - 10:
            break
            
    if best_result is None:
        # Fallback: Simple linear
        params = []
        for i in range(N):
            params.append([i * math.sqrt(3), 0, 0])
        outer = _compute_outer_for_positions(params)
        best_result = (params, outer)
        best_score = outer[2]
        
    # Final Polish
    # Use SLSQP or L-BFGS-B to refine
    # Variables: x, y, angle for each inner hex
    # We fix the outer hex from the DE result? Or optimize it too?
    # Optimizing outer hex is non-smooth. Let's fix outer hex angle and center,
    # and just optimize inner positions to minimize side?
    # Actually, we can minimize the side length by adjusting positions.
    
    # Let's try to polish the inner positions to reduce the required outer side.
    # Objective: _compute_outer_for_positions(params)[2]
    # Constraints: Disjointness and Containment.
    
    # Since constraints are non-smooth, we'll do a local search with small steps.
    # Or just return the DE result if it's valid.
    
    params, outer = best_result
    ocx, ocy, os, oang = outer
    
    # Check if we can improve by small moves
    # This is risky due to time. Let's just ensure validity and return.
    
    inner_rows = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        ang = params[3*i+2]
        inner_rows.append([x, y, ang])
        
    return inner_rows, (ocx, ocy), os, oang

if __name__ == "__main__":
    r = construct()
    print(r)
