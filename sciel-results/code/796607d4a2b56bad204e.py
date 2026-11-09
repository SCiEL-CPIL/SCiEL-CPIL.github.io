"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    # Vertices of a regular hexagon with 'pointy' orientation initially (vertex at angle 0)
    # For a regular hexagon, the circumradius equals the side length.
    # The vertices are at angles: angle + k * 60 degrees.
    # Wait, standard hexagon orientation:
    # If we consider a hexagon with a flat top, the vertices are at 30, 90, 150, 210, 270, 330.
    # If we consider a hexagon with a pointy top, vertices are at 0, 60, 120, 180, 240, 300.
    # The problem says "regular hexagons". The rotation is a free parameter.
    # Let's define the base vertices for side=1, angle=0 as having a vertex at (1, 0).
    # v_k = (cos(angle + k*pi/3), sin(angle + k*pi/3)) * side
    # This corresponds to a 'pointy' orientation at angle=0.
    angles = angle_rad + np.arange(6) * (np.pi / 3.0)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _point_in_hexagon(px, py, cx, cy, side, angle_rad):
    """Check if point (px,py) is inside the hexagon."""
    # Transform point to local coordinates
    dx = px - cx
    dy = py - cy
    # Rotate by -angle_rad
    cos_a = math.cos(-angle_rad)
    sin_a = math.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # In local coords, hexagon has vertices at (1,0), (0.5, sqrt(3)/2), etc.
    # The hexagon is defined by |x| <= 1, |y| <= sqrt(3)/2, |x + y/sqrt(3)| <= 1? No.
    # Let's use the distance to edges method or just check against the 3 axes of symmetry.
    # For a hexagon with vertices at (1,0), (0.5, h), (-0.5, h), (-1,0), (-0.5, -h), (0.5, -h) where h=sqrt(3)/2.
    # The boundaries are:
    # 1. x <= 1
    # 2. x >= -1
    # 3. y <= h
    # 4. y >= -h
    # 5. y - sqrt(3)*x <= h  (top right edge? No, edge from (1,0) to (0.5, h))
    #    Equation of line through (1,0) and (0.5, h): slope = (h-0)/(0.5-1) = -2h = -sqrt(3).
    #    y - 0 = -sqrt(3) (x - 1) => y = -sqrt(3)x + sqrt(3) => y + sqrt(3)x = sqrt(3).
    #    So y + sqrt(3)*x <= sqrt(3).
    # 6. y - sqrt(3)*x >= -h (bottom right edge? Edge from (1,0) to (0.5, -h))
    #    Slope = (-h-0)/(0.5-1) = 2h = sqrt(3).
    #    y = sqrt(3)(x-1) => y - sqrt(3)x = -sqrt(3).
    #    So y - sqrt(3)*x >= -sqrt(3).
    
    h = math.sqrt(3) / 2.0
    # Scale local coords by 1/side
    lx /= side
    ly /= side
    
    if lx > 1.0 or lx < -1.0: return False
    if ly > h or ly < -h: return False
    
    # Check diagonal constraints
    # y + sqrt(3)*x <= sqrt(3)
    if ly + math.sqrt(3)*lx > math.sqrt(3): return False
    # y - sqrt(3)*x >= -sqrt(3)  =>  y - sqrt(3)*x + sqrt(3) >= 0
    if ly - math.sqrt(3)*lx < -math.sqrt(3): return False
    # Wait, the other side:
    # Edge from (-1,0) to (-0.5, h). Slope = h/0.5 = 2h = sqrt(3).
    # y = sqrt(3)(x+1) => y - sqrt(3)x = sqrt(3).
    # So y - sqrt(3)*x <= sqrt(3).
    if ly - math.sqrt(3)*lx > math.sqrt(3): return False
    # Edge from (-1,0) to (-0.5, -h). Slope = -h/0.5 = -sqrt(3).
    # y = -sqrt(3)(x+1) => y + sqrt(3)x = -sqrt(3).
    # So y + sqrt(3)*x >= -sqrt(3).
    if ly + math.sqrt(3)*lx < -math.sqrt(3): return False
    
    return True

def _hex_distance(hex1, hex2):
    """
    Calculate the minimum distance between two hexagons.
    hex1, hex2 are tuples (cx, cy, angle). Side length is 1.
    Returns a negative value if they overlap (penetration depth), 0 if touching, positive if separated.
    This is expensive. We need a fast approximate or exact check.
    For SA, we need a fast penalty.
    Approximation: Distance between centers minus sum of inradii?
    Inradius of unit hexagon is sqrt(3)/2.
    If dist(centers) < 2 * inradius, they might overlap, but not necessarily if rotated.
    Actually, for convex polygons, we can compute the distance between them.
    However, for 11 hexagons, 55 pairs.
    Let's use a simple center-distance heuristic for the penalty, and verify validity at the end?
    No, we need to ensure non-overlap.
    
    Better: Compute the minimum distance between the two convex polygons.
    Since N is small (11), we can afford a slightly more expensive check during SA if it's vectorized or optimized.
    But SA does many iterations.
    
    Let's use the fact that the hexagons are regular.
    The distance between two regular hexagons can be computed by checking the distance from the center of one to the edges of the other, etc.
    Or simply:
    1. Check if centers are far enough.
    2. If not, check vertex-edge and edge-vertex distances.
    
    Let's implement a fast `min_dist_convex` for two hexagons.
    """
    c1x, c1y, a1 = hex1
    c2x, c2y, a2 = hex2
    
    dx = c2x - c1x
    dy = c2y - c1y
    d = math.sqrt(dx*dx + dy*dy)
    
    # Inradius r = sqrt(3)/2. If d > 2*r + 2*r = 2*sqrt(3) ~ 3.46, they are definitely separate?
    # No, circumradius is 1. Max distance from center to vertex is 1.
    # If d > 2, they might still overlap if oriented unfavorably?
    # If d > 2 * (sqrt(3)/2) = sqrt(3) ~ 1.732, they might overlap.
    # If d > 2, they are definitely separate?
    # The minimum distance between two unit hexagons is 0 (touching).
    # The maximum distance between centers when touching is 2 (vertex-vertex) or sqrt(3) (edge-edge).
    # So if d > 2, they are separate.
    if d > 2.0:
        return d - 2.0 # Approximation, but we only care about penalty > 0
    
    # Get vertices
    v1 = _hex_vertices(c1x, c1y, 1.0, a1)
    v2 = _hex_vertices(c2x, c2y, 1.0, a2)
    
    min_d = float('inf')
    
    # Check distance from each vertex of 1 to edges of 2
    for i in range(6):
        p = v1[i]
        # Edges of 2
        for j in range(6):
            e_start = v2[j]
            e_end = v2[(j+1)%6]
            # Point to segment distance
            vx = e_end[0] - e_start[0]
            vy = e_end[1] - e_start[1]
            wx = p[0] - e_start[0]
            wy = p[1] - e_start[1]
            
            c1 = vx*wx + vy*wy
            if c1 <= 0:
                dpt = math.sqrt(wx*wx + wy*wy)
            else:
                c2 = vx*vx + vy*vy
                if c2 <= c1:
                    dpt = math.sqrt((p[0]-e_end[0])**2 + (p[1]-e_end[1])**2)
                else:
                    t = c1 / c2
                    qx = e_start[0] + t*vx - p[0]
                    qy = e_start[1] + t*vy - p[1]
                    dpt = math.sqrt(qx*qx + qy*qy)
            if dpt < min_d:
                min_d = dpt
                
            # Also check if the point is "inside" the edge normal? 
            # The distance function above gives Euclidean distance.
            # To determine penetration, we need to know if the point is on the "inside" of the edge.
            # This is getting complicated for a fast SA penalty.
            
    # Alternative: Use the separating axis theorem (SAT) to check for overlap and calculate penetration.
    # For two convex polygons, if there is a separating axis, they don't overlap.
    # The minimum translation vector (MTV) gives the penetration depth.
    
    # Let's implement a simple SAT-based distance/penetration for hexagons.
    # Axes are the normals to the edges of both hexagons.
    
    def get_axes(poly):
        axes = []
        for i in range(6):
            p1 = poly[i]
            p2 = poly[(i+1)%6]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # Normal
            n = (-dy, dx)
            length = math.sqrt(n[0]*n[0] + n[1]*n[1])
            if length > 0:
                n = (n[0]/length, n[1]/length)
                axes.append(n)
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    
    min_overlap = float('inf')
    min_axis = None
    
    for ax in axes:
        # Project v1
        proj1 = [p[0]*ax[0] + p[1]*ax[1] for p in v1]
        min1, max1 = min(proj1), max(proj1)
        # Project v2
        proj2 = [p[0]*ax[0] + p[1]*ax[1] for p in v2]
        min2, max2 = min(proj2), max(proj2)
        
        # Check for gap
        if max1 < min2 or max2 < min1:
            # Separated
            gap = min(min2 - max1, min1 - max2)
            return gap # Positive gap
            
        # Overlap on this axis
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            min_axis = ax
            
    # If we get here, they overlap.
    # The penetration depth is min_overlap.
    # We return negative penetration to indicate overlap.
    return -min_overlap

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial configuration: Hexagonal lattice
    # Side length 1. Distance between centers in a lattice is 2 * sqrt(3)/2 = sqrt(3) if touching edges?
    # No, if they touch edges, distance is 2 * inradius = sqrt(3).
    # If they touch vertices, distance is 2.
    # Let's place them in a hexagonal grid.
    # Grid spacing: dx = sqrt(3), dy = 1.5?
    # For a hexagonal tiling:
    # x = i * sqrt(3) + (j % 2) * sqrt(3)/2
    # y = j * 1.5
    # This creates a tiling where hexagons touch edges.
    
    positions = []
    angles = []
    
    # Generate 11 hexagons in a compact cluster
    # Center + 6 neighbors + 4 more?
    # Let's try a 3-4-3 or similar arrangement.
    # Or just a line? No, cluster is better.
    
    # Let's initialize with a random cluster or a specific known good one.
    # For N=11, a 3x3 grid minus corners?
    # Let's use a hexagonal number pattern: 1 + 6 + 4 = 11? No, 1+6=7. 1+6+12=19.
    # So 11 is 1 center + 6 neighbors + 4 more.
    
    # Initial positions:
    # Center: (0,0)
    # 6 neighbors at distance sqrt(3)
    # 4 more surrounding those.
    
    r = math.sqrt(3)
    for i in range(N):
        if i == 0:
            positions.append([0.0, 0.0])
            angles.append(0.0)
        elif i <= 6:
            k = i - 1
            angle = k * math.pi / 3.0
            positions.append([r * math.cos(angle), r * math.sin(angle)])
            angles.append(0.0)
        else:
            # Place remaining 4 arbitrarily close
            k = i - 7
            angle = k * math.pi / 2.0 + math.pi/4
            positions.append([2*r * math.cos(angle), 2*r * math.sin(angle)])
            angles.append(0.0)
    
    # State: list of [x, y, angle]
    state = [[p[0], p[1], a] for p, a in zip(positions, angles)]
    
    best_state = [list(s) for s in state]
    best_penalty = float('inf')
    best_outer_side = float('inf')
    
    def get_outer_side_length(config):
        """Calculate the minimum outer hexagon side length required to contain all inner hexagons."""
        # The outer hexagon is centered at (0,0) and can be rotated.
        # To minimize the side length, we should align the outer hexagon optimally.
        # For a fixed set of points, the minimum enclosing hexagon (with free rotation)
        # can be found by checking the required side length for each of the 3 principal axes
        # of the outer hexagon, but since rotation is free, we need to minimize over rotation.
        
        # However, the outer hexagon is a regular hexagon.
        # A point (x,y) is inside a regular hexagon of side S centered at origin with rotation phi
        # if it satisfies the 6 inequalities.
        
        # To find the minimum S for a given phi:
        # S must be large enough to contain all points.
        # The distance from the center to the edge in direction theta is S * cos(theta - phi_edge) ...
        # Actually, the apothem (distance from center to edge) is a = S * sqrt(3)/2.
        # For a regular hexagon, the support function in direction u is:
        # h(u) = a / cos(alpha) where alpha is the angle between u and the normal of the closest edge.
        # The normals are at angles phi + k * 60 degrees.
        
        # So for a given rotation phi, the required side S is:
        # S = max over all points p and all 6 edge normals n_k of:
        #     (dot(p, n_k) / (sqrt(3)/2 * cos(angle_between(p, n_k))))?
        # No, simpler:
        # The hexagon is the intersection of 6 half-planes: dot(x, n_k) <= a.
        # So a >= max_k max_p dot(p, n_k).
        # And S = a / (sqrt(3)/2).
        
        # So for a fixed phi:
        # a_required = max over k in 0..5 of ( max over p of dot(p, n_k(phi)) )
        # S(phi) = a_required / (sqrt(3)/2)
        
        # We need to minimize S(phi) over phi in [0, 60).
        
        # To speed this up, we can sample phi.
        # Or note that the function S(phi) is periodic and piecewise smooth.
        
        # Let's sample 60 values of phi.
        
        max_a = 0.0
        # Actually, we need to compute this for each phi.
        
        # To make it fast, let's precompute the points.
        pts = []
        for p in config:
            pts.append((p[0], p[1]))
        
        min_s = float('inf')
        
        # Sample rotations
        for phi_deg in range(0, 60, 1): # 1 degree resolution
            phi = phi_deg * math.pi / 180.0
            a_req = 0.0
            for k in range(6):
                angle_n = phi + k * math.pi / 3.0
                nx = math.cos(angle_n)
                ny = math.sin(angle_n)
                
                max_dot = -float('inf')
                for px, py in pts:
                    dot = px * nx + py * ny
                    if dot > max_dot:
                        max_dot = dot
                if max_dot > a_req:
                    a_req = max_dot
            
            s = a_req / (math.sqrt(3) / 2.0)
            if s < min_s:
                min_s = s
                
        return min_s

    def get_penalty(config):
        """Calculate penalty for overlaps."""
        penalty = 0.0
        for i in range(len(config)):
            for j in range(i+1, len(config)):
                d = _hex_distance(config[i], config[j])
                if d < 0:
                    # Overlap
                    # Penalty is proportional to penetration squared
                    penalty += (-d) ** 2
        return penalty

    # Evaluate initial state
    p = get_penalty(state)
    s = get_outer_side_length(state)
    total_cost = s + 100.0 * p # Weight overlap heavily
    
    if p < best_penalty:
        best_penalty = p
        best_outer_side = s
        best_state = [list(x) for x in state]
    
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline and T > T_min:
        
        # Propose a move
        # Randomly pick a hexagon and perturb its position and angle
        idx = random.randint(0, N-1)
        
        new_state = [list(x) for x in state]
        
        # Perturbation size
        step_size = 0.5 * T
        new_state[idx][0] += random.gauss(0, step_size)
        new_state[idx][1] += random.gauss(0, step_size)
        new_state[idx][2] += random.gauss(0, step_size * 0.5)
        
        new_p = get_penalty(new_state)
        new_s = get_outer_side_length(new_state)
        new_cost = new_s + 100.0 * new_p
        
        old_p = get_penalty(state)
        old_s = get_outer_side_length(state)
        old_cost = old_s + 100.0 * old_p
        
        delta_cost = new_cost - old_cost
        
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            state = new_state
            if new_p < best_penalty:
                best_penalty = new_p
                best_outer_side = new_s
                best_state = [list(x) for x in state]
        
        T *= cooling
        
        # Check deadline
        if time.time() > deadline - 5.0:
            break

    # Final refinement: Local search on best_state
    # Try to reduce the outer side length by fine-tuning positions
    # This is optional if time permits.
    
    # Ensure best_state is valid (penalty == 0)
    # If not, we might have to return a valid but suboptimal solution.
    # The task says "disjoint". If penalty > 0, it's invalid.
    # We should try to ensure we return a valid solution.
    # If best_penalty > 0, we should probably try to expand the hexagons apart or return a known valid pattern.
    
    # For now, return best_state.
    # The evaluator checks validity. If invalid, score 0.
    # So we must ensure validity.
    
    # Let's check if best_state is valid.
    if best_penalty > 1e-6:
        # Fallback to a known valid sparse configuration
        # Just spread them out far enough.
        fallback_state = []
        for i in range(N):
            angle = i * 2 * math.pi / N
            r = 10.0
            fallback_state.append([r * math.cos(angle), r * math.sin(angle), 0.0])
        return fallback_state, (0.0, 0.0), 20.0, 0.0

    return best_state, (0.0, 0.0), best_outer_side, 0.0

# EVOLVE-BLOCK-END
