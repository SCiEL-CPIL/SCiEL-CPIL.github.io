"""Place 12 unit hexagons inside a minimal regular hexagon."""
import numpy as np
import math
import time

N = 12
DEADLINE = 145.0  # safety margin before 150s kill

def construct():
    t0 = time.time()
    
    # Honeycomb positions for 12 unit hexagons
    # Hex grid: axial coordinates (q, r) -> pixel
    # For flat-top hexagons with side=1, spacing is 2*sqrt(3) horizontally
    # and 3 vertically for adjacent rows
    # Actually for pointy-top (angle 0), the hex vertices are at angles 0,60,120,...
    # Side length 1: width = 2, height = sqrt(3)
    # Horizontal spacing between centers in same row: 2*sqrt(3) (for flat-top) 
    # or 3 (for pointy-top, horizontal neighbors)
    # Let's use standard hex grid:
    # For pointy-top hexagons (angle=0), vertices at 0,60,120,180,240,300 degrees
    # Center-to-center distance for adjacent hexagons: 2*sqrt(3)
    # In a hex grid with pointy-top:
    #   Same row: dx = 2*sqrt(3), dy = 0
    #   Adjacent row: dx = sqrt(3), dy = 3
    
    # Actually let me reconsider. A regular hexagon with side 1:
    # If angle=0 means one vertex points right, vertices at (1,0), (0.5, sqrt(3)/2), ...
    # The "width" (vertex to opposite vertex) = 2
    # The "height" (flat to opposite flat) = sqrt(3)
    # For tiling: adjacent centers are 2*sqrt(3) apart? No.
    # Two unit hexagons sharing an edge: center distance = 2 * (apothem) = 2 * sqrt(3)/2 = sqrt(3)
    # Wait: apothem = sqrt(3)/2 * side = sqrt(3)/2. Center to edge = apothem.
    # Two hexagons sharing an edge: centers are 2*apothem = sqrt(3) apart.
    
    # For a hex grid with pointy-top orientation:
    # Row 0: centers at (0,0), (sqrt(3)*2, 0), (sqrt(3)*4, 0), ...  -> spacing 2*sqrt(3)
    # Wait no. If two hexagons share a vertical edge (pointy-top), 
    # their centers are 2*sqrt(3) apart horizontally? Let me think again.
    
    # Pointy-top hex, side=1. Vertices: (0,1), (sqrt(3)/2, 1/2), (sqrt(3)/2,-1/2), (0,-1), (-sqrt(3)/2,-1/2), (-sqrt(3)/2,1/2)
    # Width (left-right) = sqrt(3), Height (top-bottom) = 2
    # Two hexagons side by side (sharing a vertical-ish edge): 
    # The rightmost vertex of left hex is at x=sqrt(3)/2, leftmost of right hex at x=-sqrt(3)/2+dx
    # They share an edge when the edge from (sqrt(3)/2,1/2) to (sqrt(3)/2,-1/2) of left
    # coincides with edge from (-sqrt(3)/2,1/2) to (-sqrt(3)/2,-1/2) of right.
    # So dx = sqrt(3)/2 + sqrt(3)/2 = sqrt(3). Centers are sqrt(3) apart horizontally.
    
    # For hex grid, row offset:
    # Even rows: x = col * sqrt(3)
    # Odd rows: x = (col + 0.5) * sqrt(3), y = row * 1.5
    # Wait, vertical spacing: top vertex at y=1, bottom at y=-1. 
    # Adjacent row shares a slanted edge. Vertical distance = 1.5 (from center to center)
    # because the shared edge midpoint is at y=0.5 from center, and the edge goes from 
    # (sqrt(3)/2, 0.5) to (0, 1) ... hmm this is getting complicated.
    
    # Let me just use the standard hex grid formula:
    # For pointy-top hexagons with "radius" (center to vertex) = 1 (side=1):
    # Actually for a regular hexagon with side s=1, the circumradius = 1, inradius = sqrt(3)/2.
    # 
    # Standard hex grid (pointy-top):
    #   x = size * sqrt(3) * (col + 0.5 * (row & 1))
    #   y = size * 3/2 * row
    # where size = circumradius = side length = 1
    # Center-to-center distance = sqrt(3) * size for horizontal neighbors
    # = sqrt(3) for our case.
    
    # Let's place 12 hexagons in a compact hexagonal arrangement:
    # A hexagon of hexagons with "radius" 2 (center + 6 + 9... no)
    # Hex numbers: 1, 7, 19, 37...
    # 12 is not a hex number. Let's do a 3x4 or similar rectangular grid,
    # or a hexagonal cluster.
    
    # Actually, for minimal enclosing hexagon, a compact hexagonal cluster is best.
    # Let's try: 1 center, 6 in first ring, 5 more in second ring = 12.
    # Or just do a rectangular grid: 4 columns x 3 rows.
    
    # Let me try the hex grid arrangement:
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 4 hexagons
    # That's 12.
    
    sqrt3 = math.sqrt(3)
    
    # Hex grid positions (pointy-top, size=1)
    positions = []
    # 3 rows of 4
    for row in range(3):
        for col in range(4):
            x = sqrt3 * (col + 0.5 * (row % 2))
            y = 1.5 * row
            positions.append((x, y, 0.0))
    
    # Center the positions
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    positions = [(x - cx, y - cy, 0.0) for x, y, a in positions]
    
    # Now find the minimal enclosing regular hexagon
    # A regular hexagon with center (cx, cy), side s, rotation angle theta
    # contains a point (px, py) if for all 6 face normals n_i:
    #   n_i . (p - center) <= s * sqrt(3)/2  (apothem)
    # Actually, the constraint is: for each of the 6 edges, the projection of 
    # (p - center) onto the outward normal must be <= apothem = s * sqrt(3)/2
    
    # The 6 outward normals for a hexagon rotated by theta:
    # Base normals (for theta=0, pointy-top): angles 0, 60, 120, 180, 240, 300
    # Wait, for a regular hexagon, the face normals are perpendicular to the edges.
    # For pointy-top (vertex at top), edges are at angles 30, 90, 150, 210, 270, 330
    # So normals are at angles 0, 60, 120, 180, 240, 300? Let me be careful.
    
    # A regular hexagon with a vertex pointing up (pointy-top):
    # Vertices at angles 90, 150, 210, 270, 330, 30 (i.e., 90 + k*60 for k=0..5)
    # Edges connect consecutive vertices. The outward normal of each edge is 
    # perpendicular to the edge, pointing outward.
    # Edge from vertex at 90 to vertex at 150: midpoint direction is 120, 
    # normal is at 120 - 90 = 30? No, the normal to the edge is at angle 120 (bisector).
    # Actually, for a regular polygon, the outward normal of edge k is at angle 
    # (vertex_k_angle + vertex_{k+1}_angle)/2 = 90 + 60*k + 30 = 120 + 60*k.
    # So normals at 120, 180, 240, 300, 0, 60 (mod 360).
    # Which is the same as 0, 60, 120, 180, 240, 300.
    
    # So for a hexagon with rotation angle theta (rotating the whole hexagon),
    # the outward normals are at angles theta + 0, theta + 60, ..., theta + 300.
    # Wait, I need to be consistent. Let me define:
    # The hexagon has vertices at angles theta + 30 + k*60 (for k=0..5) if we want
    # a flat-top when theta=0. Or vertices at theta + 90 + k*60 for pointy-top at theta=0.
    
    # Let me just use: the 6 supporting lines of the hexagon have normals at
    # angles alpha_k = theta + k * 60 degrees for k=0..5.
    # The hexagon is the intersection of half-planes:
    #   n_k . (p - center) <= apothem, where apothem = s * sqrt(3)/2
    # and n_k = (cos(alpha_k), sin(alpha_k)).
    
    # For a point p to be inside: for all k, 
    #   (p - center) . n_k <= s * sqrt(3)/2
    # So s >= 2/sqrt(3) * max_k (p - center) . n_k  for each p.
    # For all points: s >= 2/sqrt(3) * max_p max_k (p - center) . n_k
    
    # We want to minimize s over center and theta.
    
    # For fixed theta, the optimal center can be found, and s is determined.
    # Let's search over theta.
    
    best_s = float('inf')
    best_center = (0, 0)
    best_theta = 0.0
    
    # For each theta, the constraint for each point p is:
    #   max_k (p - c) . n_k(theta) <= s * sqrt(3)/2
    # We want to minimize s such that this holds for all p.
    # s = 2/sqrt(3) * max_p max_k (p - c) . n_k(theta)
    # For fixed theta, minimize over c: max_p max_k (p - c) . n_k(theta)
    # This is equivalent to: find c minimizing the maximum over all (p, k) of (p-c).n_k
    # = find c minimizing max over all points q in {p - c + ... } 
    # Actually, for fixed theta, define f(c) = max_p max_k (p - c) . n_k
    # = max_p max_k [p.n_k - c.n_k]
    # = max_p [max_k p.n_k] - min_k c.n_k ... no that's not right.
    # = max over (p,k) of (p.n_k - c.n_k)
    # = max over (p,k) of p.n_k - min over k of c.n_k ... no.
    # max_{p,k} (p.n_k - c.n_k) = max_{p,k} p.n_k - min_k c.n_k? No.
    # = max_{p,k} (p.n_k) - min_k(c.n_k)? No, the max is over pairs (p,k).
    # max_{p,k} (p.n_k - c.n_k) = max_{p,k} p.n_k - min_k c.n_k is WRONG.
    # It's: for each k, max_p (p.n_k - c.n_k) = max_p(p.n_k) - c.n_k
    # Then max over k of that: max_k [max_p(p.n_k) - c.n_k]
    
    # So for fixed theta: minimize over c: max_k [M_k - c.n_k]
    # where M_k = max_p (p . n_k)
    # This is a convex optimization problem in c (2D).
    # The optimal c satisfies: max_k [M_k - c.n_k] is minimized.
    # This is equivalent to finding the smallest s such that c.n_k >= M_k - s for all k,
    # i.e., c is in the intersection of half-planes, and we minimize s.
    # The minimum s is achieved when c is at a vertex of the arrangement.
    
    # Alternatively, note that max_k [M_k - c.n_k] is the support function of the 
    # convex hull of {M_k * n_k} evaluated at... hmm, this is getting complex.
    
    # Simpler: for fixed theta, use the fact that the optimal center is the 
    # center of the minimum enclosing hexagon, which can be found by noting
    # that the binding constraints are at most 3 (in 2D).
    
    # Even simpler: just do a numerical search. For each theta, find optimal center
    # by minimizing max_k [M_k - c.n_k] using a few iterations.
    
    # Actually, let me just try: for each theta, the optimal center is the average
    # of the points that achieve the maximum in each direction, or we can use
    # the fact that for a regular hexagon, the optimal enclosing hexagon center
    # is related to the center of the convex hull.
    
    # Let me try a direct approach: for each theta, compute M_k = max_p(p.n_k),
    # then the minimum enclosing hexagon has s = 2/sqrt(3) * max_k M_k if centered at origin,
    # but we can shift. The shift c should minimize max_k (M_k - c.n_k).
    # The optimal c satisfies: for the active constraints (where M_k - c.n_k = s),
    # the weighted average of n_k is zero.
    
    # For simplicity, let's just try c = 0 (center of mass) and also try to optimize.
    
    # Let me use a simple approach: for each theta, set c to the centroid of the points,
    # compute s, and also try a few iterations of gradient descent on c.
    
    points = np.array([(p[0], p[1]) for p in positions])
    
    # Search over theta
    for theta_deg in range(0, 60, 1):  # 60 degree symmetry
        if time.time() - t0 > DEADLINE:
            break
        theta = math.radians(theta_deg)
        normals = np.array([[math.cos(theta + math.radians(k*60)), 
                           math.sin(theta + math.radians(k*60))] for k in range(6)])
        
        # M_k = max_p (p . n_k)
        M = np.max(points @ normals.T, axis=0)  # shape (6,)
        
        # Minimize max_k (M_k - c . n_k) over c
        # Start with c = 0
        c = np.array([0.0, 0.0])
        for _ in range(50):
            vals = M - c @ normals.T  # shape (6,)
            s = np.max(vals)
            active = vals >= s - 1e-10
            if not np.any(active):
                break
            # Gradient of max is the average of active normals (with sign)
            # d/dc max_k(M_k - c.n_k) = -n_k for active k
            # So we want to move c in direction that reduces max
            # Gradient descent: c -= lr * (-mean of active normals) = c += lr * mean(active normals)
            grad = -normals[active].mean(axis=0)
            # Actually, the subgradient of f(c) = max_k(M_k - c.n_k) is 
            # conv{-n_k : k active}. To minimize, we move in direction -grad = mean of active n_k
            # c_new = c + lr * mean(active normals)
            # But we need to be careful. Let's use: c += lr * (s - M) projected...
            # Simpler: c += lr * normals[active].mean(axis=0)
            lr = 0.1 / (1 + _ * 0.1)
            c = c + lr * normals[active].mean(axis=0)
        
        s_val = np.max(M - c @ normals.T) * 2 / sqrt3
        if s_val < best_s:
            best_s = s_val
            best_center = (c[0], c[1])
            best_theta = theta
    
    # Try finer search around best_theta
    if time.time() - t0 < DEADLINE - 5:
        for theta_deg in np.linspace(best_theta - 5, best_theta + 5, 21):
            if time.time() - t0 > DEADLINE - 2:
                break
            theta = math.radians(theta_deg)
            normals = np.array([[math.cos(theta + math.radians(k*60)), 
                               math.sin(theta + math.radians(k*60))] for k in range(6)])
            M = np.max(points @ normals.T, axis=0)
            c = np.array([0.0, 0.0])
            for _ in range(100):
                vals = M - c @ normals.T
                s = np.max(vals)
                active = vals >= s - 1e-10
                if not np.any(active):
                    break
                lr = 0.05 / (1 + _ * 0.05)
                c = c + lr * normals[active].mean(axis=0)
            s_val = np.max(M - c @ normals.T) * 2 / sqrt3
            if s_val < best_s:
                best_s = s_val
                best_center = (c[0], c[1])
                best_theta = theta
    
    # Final polish
    if time.time() - t0 < DEADLINE - 2:
        theta = best_theta
        normals = np.array([[math.cos(theta + math.radians(k*60)), 
                           math.sin(theta + math.radians(k*60))] for k in range(6)])
        M = np.max(points @ normals.T, axis=0)
        c = np.array(best_center)
        for _ in range(200):
            vals = M - c @ normals.T
            s = np.max(vals)
            active = vals >= s - 1e-12
            if not np.any(active):
                break
            lr = 0.01 / (1 + _ * 0.01)
            c = c + lr * normals[active].mean(axis=0)
        best_s = np.max(M - c @ normals.T) * 2 / sqrt3
        best_center = (c[0], c[1])
    
    # Add small safety margin
    best_s += 0.001
    
    # Return: inner positions (x, y, angle), outer center, side, angle
    inner = [[p[0], p[1], p[2]] for p in positions]
    outer_center = (best_center[0], best_center[1])
    outer_side = best_s
    outer_angle = math.degrees(best_theta)
    
    return inner, outer_center, outer_side, outer_angle

