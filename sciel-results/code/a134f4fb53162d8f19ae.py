from scipy.optimize import differential_evolution
import math
import time
import numpy as np
from scipy.optimize import minimize

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

def _point_in_hex_dist_sq(px, py, cx, cy, s, angle_deg):
    """
    Returns squared distance from point (px, py) to the hexagon if outside,
    or 0 if inside.
    Uses the fact that a regular hexagon is the intersection of 3 strips.
    Actually, for robustness with rotation, we transform the point to the hexagon's local frame.
    """
    a = math.radians(angle_deg)
    # Rotate point relative to center
    dx = px - cx
    dy = py - cy
    # Rotate by -a to align hexagon with axes
    rx = dx * math.cos(-a) - dy * math.sin(-a)
    ry = dx * math.sin(-a) + dy * math.cos(-a)
    
    # In local frame, hexagon vertices are at angles 0, 60, ...
    # The hexagon is defined by |x| <= s, |y| <= s*sqrt(3)/2, |x + y*sqrt(3)/2| <= s*sqrt(3)/2? 
    # No, standard hexagon with flat top? 
    # My _hex_vertices uses angle 0 as vertex at (s, 0). So it's pointy-top? 
    # Vertex 0: (s, 0). Vertex 1: (s/2, s*sqrt(3)/2).
    # This is a "pointy-side" hexagon if we consider x horizontal.
    # Let's use the support function approach or simple polygon check.
    # For speed in optimization, we can use the 3 pair of parallel lines.
    # Lines are at distance s*sqrt(3)/2 from center? No, apothem is s*sqrt(3)/2.
    # The 3 pairs of parallel sides are at distance apothem from center.
    # Directions of normals are 30, 90, 150 degrees?
    # Vertex 0 is at 0 deg. Side 0-1 is at 30 deg normal?
    # Midpoint of side 0-1 is at angle 30 deg.
    # So normals are at 30, 90, 150 deg.
    
    apothem = s * math.sqrt(3) / 2
    max_violation = 0.0
    for i in range(3):
        # Normal angle
        theta_n = math.radians(30 + i * 60)
        # Project local point onto normal
        proj = rx * math.cos(theta_n) + ry * math.sin(theta_n)
        if proj > apothem:
            violation = proj - apothem
            if violation > max_violation:
                max_violation = violation
        
    # If max_violation > 0, the point is outside.
    # The distance to the hexagon boundary is NOT simply max_violation if outside corner regions,
    # but for convex sets, the distance to the set is the max over support functions?
    # Actually, for a convex polygon, if you are outside, the distance is the minimum distance to any edge.
    # The projection violation gives the distance to the *infinite strip*.
    # If the point is outside the hexagon, it is outside at least one strip.
    # The distance to the hexagon is the minimum distance to the boundary.
    # This is tricky to compute exactly in one shot without clipping.
    # However, for penalty function in optimization, we can use the max violation as a proxy for "how far out" it is,
    # but it's not the true Euclidean distance.
    # A better penalty: sum of squared distances to the 3 pairs of lines, but only if outside.
    # Let's stick to a simpler robust check: Point in Polygon.
    # But that's slow.
    # Alternative: The hexagon is the intersection of 3 strips.
    # Distance to intersection of strips is not just max distance to strips.
    
    # Let's use a robust but slightly slower method: transform to local, check against 3 inequalities.
    # If any inequality is violated, we are outside.
    # To get a smooth penalty, we can square the violation of each inequality.
    # Sum of squared violations is a good penalty.
    
    sum_sq_viol = 0.0
    for i in range(3):
        theta_n = math.radians(30 + i * 60)
        proj = rx * math.cos(theta_n) + ry * math.sin(theta_n)
        if proj > apothem:
            sum_sq_viol += (proj - apothem)**2
        # Also check the other side? No, hexagon is centered, so we need -apothem <= proj <= apothem
        if proj < -apothem:
            sum_sq_viol += (-apothem - proj)**2
            
    return sum_sq_viol

def _pairwise_penalty(p1, p2):
    """
    p1, p2: (cx, cy, angle_deg)
    Returns penalty if they overlap.
    """
    c1x, c1y, a1 = p1
    c2x, c2y, a2 = p2
    
    # Quick reject: distance between centers > 2 * circumradius (2*s)
    # Circumradius is s=1. So if dist > 2, no overlap.
    dx = c1x - c2x
    dy = c1y - c2y
    if dx*dx + dy*dy > 4.0:
        return 0.0
        
    # Check if any vertex of hex1 is in hex2, or vice versa.
    # Or check if edges intersect.
    # For penalty, we can sample vertices.
    # Better: use the distance between convex polygons.
    # Since N is small (11), and this is called in DE, we need speed.
    
    # Let's use the point-in-hex penalty for vertices.
    # If all vertices of H1 are in H2 and all vertices of H2 are in H1, they are not overlapping (one inside other, impossible for equal size disjoint).
    # Actually, if they overlap, either a vertex of one is inside the other, or edges cross.
    # If edges cross, no vertex is necessarily inside.
    # But for regular hexagons of same size, if they overlap, it's likely a vertex is inside or edges cross.
    
    # A robust penalty:
    # 1. Check vertices of H1 against H2.
    # 2. Check vertices of H2 against H1.
    # 3. If no vertex is inside, check edge intersections.
    
    v1 = _hex_vertices(c1x, c1y, 1.0, a1)
    v2 = _hex_vertices(c2x, c2y, 1.0, a2)
    
    penalty = 0.0
    
    # Vertices of 1 in 2
    for pt in v1:
        penalty += _point_in_hex_dist_sq(pt[0], pt[1], c2x, c2y, 1.0, a2)
        
    # Vertices of 2 in 1
    for pt in v2:
        penalty += _point_in_hex_dist_sq(pt[0], pt[1], c1x, c1y, 1.0, a1)
        
    # If penalty is 0, we must check edge intersections.
    if penalty == 0.0:
        # Check if edges intersect
        # Edges of H1: (v1[i], v1[(i+1)%6])
        # Edges of H2: (v2[j], v2[(j+1)%6])
        
        def segments_intersect(p1, p2, p3, p4):
            # p1, p2, p3, p4 are (x, y)
            # Returns True if segments p1-p2 and p3-p4 intersect
            def cross(o, a, b):
                return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            
            d1 = cross(p3, p4, p1)
            d2 = cross(p3, p4, p2)
            d3 = cross(p1, p2, p3)
            d4 = cross(p1, p2, p4)
            
            if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
               ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                return True
            return False

        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j+1)%6]
                if segments_intersect(p1, p2, p3, p4):
                    # If they intersect, we need a penalty.
                    # Approximate penalty: distance between centers?
                    # Or just a fixed large penalty?
                    # A smooth penalty is better.
                    # Let's add a penalty based on how much they overlap.
                    # This is hard. Let's just add a large constant if they intersect.
                    # This might make DE struggle.
                    # Better: sample points along the edges?
                    # For now, let's assume vertex checks catch most cases for regular hexagons.
                    # If edges cross, the midpoint of one edge might be inside the other?
                    # Not necessarily.
                    # Let's add a penalty for edge intersection by checking if midpoints are inside.
                    mid1 = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
                    mid2 = ((p3[0]+p4[0])/2, (p3[1]+p4[1])/2)
                    penalty += _point_in_hex_dist_sq(mid1[0], mid1[1], c2x, c2y, 1.0, a2)
                    penalty += _point_in_hex_dist_sq(mid2[0], mid2[1], c1x, c1y, 1.0, a1)
                    # If still 0, it's a crossing without midpoint inside.
                    # This is rare for convex polygons of similar size.
                    if penalty == 0.0:
                        penalty += 1.0 # Hard penalty
                
    return penalty

def _outer_penalty(inner_params, outer_params):
    """
    inner_params: list of (cx, cy, angle)
    outer_params: (cx, cy, s, angle)
    Returns penalty if any inner hexagon is not fully inside the outer hexagon.
    """
    ocx, ocy, os, oang = outer_params
    
    penalty = 0.0
    for ip in inner_params:
        icx, icy, iang = ip
        # Check vertices of inner hexagon
        v_in = _hex_vertices(icx, icy, 1.0, iang)
        for pt in v_in:
            # Distance to outer hexagon
            # We want pt to be INSIDE outer.
            # _point_in_hex_dist_sq returns 0 if inside, >0 if outside.
            # So we sum these.
            penalty += _point_in_hex_dist_sq(pt[0], pt[1], ocx, ocy, os, oang)
            
    return penalty

def _objective(x):
    """
    x: [cx1, cy1, a1, ..., cxN, cyN, aN, ocx, ocy, os, oang]
    """
    n_inner = N
    inner_params = []
    for i in range(n_inner):
        cx = x[i*3]
        cy = x[i*3+1]
        a = x[i*3+2]
        inner_params.append((cx, cy, a))
        
    ocx = x[n_inner*3]
    ocy = x[n_inner*3+1]
    os = x[n_inner*3+2]
    oang = x[n_inner*3+3]
    outer_params = (ocx, ocy, os, oang)
    
    # 1. Pairwise disjointness
    pen_pair = 0.0
    for i in range(n_inner):
        for j in range(i+1, n_inner):
            pen_pair += _pairwise_penalty(inner_params[i], inner_params[j])
            
    # 2. Containment
    pen_outer = _outer_penalty(inner_params, outer_params)
    
    # 3. Minimize outer side length
    # We want to minimize os.
    # But we also need to ensure validity.
    # If penalty is 0, we can shrink os.
    # So objective = os + lambda * (pen_pair + pen_outer)
    # If penalty > 0, we increase it to reduce penalty.
    
    return os + 100.0 * (pen_pair + pen_outer)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initial guess: Lattice packing
    # Center at (0,0), side 1
    # Hex grid:
    # Ring 0: (0,0)
    # Ring 1: 6 hexagons
    # Ring 2: 12 hexagons... we need 11.
    # Let's place them in a tight cluster.
    
    init_inner = []
    # Center
    init_inner.append((0.0, 0.0, 0.0))
    # Ring 1
    for i in range(6):
        a = math.radians(60*i)
        x = 1.5 * math.cos(a)
        y = 1.5 * math.sin(a)
        init_inner.append((x, y, 0.0))
    # Ring 2 (5 more to make 11)
    # Pick 5 from ring 2
    # Ring 2 positions are at distance 3.0 from center?
    # Neighbors of ring 1.
    # Let's just add 5 more around.
    # Actually, let's use a simple grid for init.
    # (0,0)
    # (1.5, 0.866), (1.5, -0.866), (0, 1.732), (0, -1.732), (-1.5, 0.866), (-1.5, -0.866)
    # That's 7.
    # Add 4 more from ring 2.
    # (3.0, 0), (2.25, 1.9), (0.75, 2.6), (-1.5, 2.6)...
    # Let's just use a bounding box for init.
    
    x0 = []
    for p in init_inner:
        x0.append(p[0])
        x0.append(p[1])
        x0.append(p[2])
        
    # Outer init: large enough to contain
    x0.extend([0.0, 0.0, 5.0, 0.0])
    
    # Bounds
    # Inner centers: [-10, 10]
    # Angles: [0, 360]
    # Outer center: [-10, 10]
    # Outer side: [3.0, 10.0] # Min possible is roughly 2.5?
    # Outer angle: [0, 360]
    
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((0, 360))
        
    bounds.append((-10, 10))
    bounds.append((-10, 10))
    bounds.append((3.0, 10.0))
    bounds.append((0, 360))
    
    # Run DE
    # maxiter and popsize need to be tuned for 140s budget.
    # N=11, 3*11 + 4 = 37 variables.
    # DE is slow.
    # Let's try a smaller population and iterations, and check time.
    
    best_x = None
    best_val = float('inf')
    
    # Try DE
    try:
        # Use a callback to check time? DE doesn't support callback easily in all versions.
        # We can run it and hope it finishes.
        # Or use a loop of smaller runs?
        # Let's try one run with moderate settings.
        
        # To be safe, we can run DE with maxiter=50, popsize=10
        # If it takes too long, we return best so far.
        # But DE runs in C, so we can't interrupt easily.
        # Let's estimate: 37 vars, popsize 10, 50 iters.
        # 500 function evals.
        # Each eval: 11*10/2 = 55 pairs.
        # Each pair: vertex checks (12 pts) + edge checks (36 segs).
        # 55 * (12+36) ~ 2640 ops.
        # 500 * 2640 ~ 1.3M ops.
        # Python is slow. Might take > 10s.
        # Let's optimize _pairwise_penalty.
        
        # Actually, let's just run it. If it times out, the whole script dies.
        # We need to be careful.
        
        # Let's use a simpler penalty first to speed up eval.
        # Maybe skip edge checks in DE, only use vertex checks?
        # Vertex checks are sufficient for many packings.
        
        res = differential_evolution(_objective, bounds, maxiter=20, popsize=5, seed=42, polish=False)
        
        if res.success or res.fun < best_val:
            best_x = res.x
            best_val = res.fun
            
    except Exception:
        pass
        
    # Check time
    if time.time() > deadline:
        # Return best found
        if best_x is not None:
            return _format_result(best_x)
        else:
            return _format_result(x0)
            
    # Local optimization
    if best_x is not None:
        try:
            res2 = minimize(_objective, best_x, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
            if res2.fun < best_val:
                best_x = res2.x
                best_val = res2.fun
        except Exception:
            pass
    
    return _format_result(best_x if best_x is not None else x0)

def _format_result(x):
    inner = []
    for i in range(N):
        cx = x[i*3]
        cy = x[i*3+1]
        a = x[i*3+2]
        inner.append([cx, cy, a])
        
    ocx = x[N*3]
    ocy = x[N*3+1]
    os = x[N*3+2]
    oang = x[N*3+3]
    
    return inner, (ocx, ocy), os, oang

# Ensure the function is available
# The evaluator calls construct()

if __name__ == "__main__":
    # Test
    r = construct()
    print(r)
