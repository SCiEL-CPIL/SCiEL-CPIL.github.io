"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from math import cos, sin, pi, sqrt

N = 11

# Helper functions for hexagon geometry

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon with center (cx,cy), side length, and rotation in degrees."""
    angle_rad = angle_deg * pi / 180.0
    vertices = []
    for i in range(6):
        theta = angle_rad + i * pi / 3.0
        x = cx + side * cos(theta)
        y = cy + side * sin(theta)
        vertices.append((x, y))
    return vertices

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon with center (cx,cy), side, rotation."""
    # Transform point to hexagon's local coordinate system
    angle_rad = angle_deg * pi / 180.0
    dx = px - cx
    dy = py - cy
    # Rotate point by -angle
    rx = dx * cos(-angle_rad) - dy * sin(-angle_rad)
    ry = dx * sin(-angle_rad) + dy * cos(-angle_rad)
    
    # For a regular hexagon with side length s, centered at origin, with flat top:
    # The hexagon can be described by |x| <= s*sqrt(3)/2 and appropriate y bounds
    # Actually, let's use the standard form: vertices at angles 0, 60, 120, 180, 240, 300
    # The distance from center to each edge is s*sqrt(3)/2
    # Use the property that a point is inside if its distance to each of the 6 edges is <= 0
    
    # Get vertices in local coords
    verts = []
    for i in range(6):
        theta = i * pi / 3.0
        vx = side * cos(theta)
        vy = side * sin(theta)
        verts.append((vx, vy))
    
    # Check if point is on the same side of each edge as the center
    inside = True
    for i in range(6):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % 6]
        # Edge from (x1,y1) to (x2,y2)
        # Normal pointing inward (toward center)
        ex = x2 - x1
        ey = y2 - y1
        # Normal
        nx = -ey
        ny = ex
        # Check if center and point are on same side
        center_side = (0 - x1) * nx + (0 - y1) * ny
        point_side = (rx - x1) * nx + (ry - y1) * ny
        if center_side * point_side < 0:
            inside = False
            break
    return inside

def min_distance_between_hexagons(cx1, cy1, side1, angle1, cx2, cy2, side2, angle2):
    """
    Compute minimum distance between two regular hexagons.
    If they overlap, return negative value (approximate penetration depth).
    """
    # Sample points on both hexagons and compute min distance
    # This is approximate but works for optimization
    
    # Get vertices
    verts1 = hexagon_vertices(cx1, cy1, side1, angle1)
    verts2 = hexagon_vertices(cx2, cy2, side2, angle2)
    
    # Also sample points along edges
    num_samples = 10
    points1 = []
    points2 = []
    
    for i in range(6):
        x1, y1 = verts1[i]
        x2, y2 = verts1[(i + 1) % 6]
        for j in range(num_samples):
            t = j / num_samples
            points1.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
    
    for i in range(6):
        x1, y1 = verts2[i]
        x2, y2 = verts2[(i + 1) % 6]
        for j in range(num_samples):
            t = j / num_samples
            points2.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
    
    # Compute minimum distance between any pair of points
    min_dist = float('inf')
    for p1 in points1:
        for p2 in points2:
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            d = sqrt(dx * dx + dy * dy)
            if d < min_dist:
                min_dist = d
    
    # Check if hexagons overlap by checking if any vertex of one is inside the other
    overlap = False
    for v in verts1:
        if point_in_hexagon(v[0], v[1], cx2, cy2, side2, angle2):
            overlap = True
            break
    if not overlap:
        for v in verts2:
            if point_in_hexagon(v[0], v[1], cx1, cy1, side1, angle1):
                overlap = True
                break
    
    if overlap:
        # Return negative distance (penetration)
        # Approximate by -min_dist if they're close, but this is tricky
        # For optimization, we want a penalty when overlapping
        return -min_dist if min_dist < 0.1 else -0.1
    else:
        return min_dist

def hexagon_radius(side, angle_deg):
    """Circumradius of hexagon."""
    return side

def hexagon_apothem(side):
    """Distance from center to edge (inradius)."""
    return side * sqrt(3) / 2

def distance_from_point_to_hexagon_edge(px, py, cx, cy, side, angle_deg):
    """
    Compute signed distance from point to hexagon boundary.
    Positive if outside, negative if inside.
    """
    # Transform to local coords
    angle_rad = angle_deg * pi / 180.0
    dx = px - cx
    dy = py - cy
    rx = dx * cos(-angle_rad) - dy * sin(-angle_rad)
    ry = dx * sin(-angle_rad) + dy * cos(-angle_rad)
    
    # For hexagon with vertices at angles 0, 60, 120, ...
    # The six edges have normals at angles 30, 90, 150, 210, 270, 330 degrees
    # Distance from center to each edge is apothem = side * sqrt(3)/2
    
    apothem = side * sqrt(3) / 2
    min_dist = float('inf')
    
    for i in range(6):
        # Normal direction for edge i
        theta = (2 * i + 1) * pi / 6.0  # 30, 90, 150, 210, 270, 330 degrees
        nx = cos(theta)
        ny = sin(theta)
        
        # Distance from point to this edge line
        dist = rx * nx + ry * ny - apothem
        
        if dist < min_dist:
            min_dist = dist
    
    # If the point is outside the hexagon's angular range, we need to check distance to vertices
    # Simplified: return min_dist which works for points near the hexagon
    return min_dist

def construct():
    """
    Place N unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    Returns: (inner, center, side, angle)
    - inner: list of [x, y, angle_degrees] for each inner hexagon
    - center: (x, y) of outer hexagon center
    - side: side length of outer hexagon
    - angle: rotation of outer hexagon in degrees
    """
    start_time = time.time()
    deadline = start_time + 130.0  # 130s deadline to be safe
    
    # Initial configuration: hexagonal packing
    # For N hexagons, arrange in a hexagonal pattern
    # The seed uses a linear arrangement with side 40, which is very large
    # We can do much better with a compact hexagonal arrangement
    
    # Start with a hexagonal arrangement of N hexagons
    # For a hexagonal number pattern, the arrangement is compact
    
    # Let's try a simple approach: place hexagons in a hexagonal grid pattern
    # Unit hexagons (side=1) have apothem = sqrt(3)/2 ≈ 0.866
    
    # Hexagonal packing: centers are at distance 2*sqrt(3)/2 = sqrt(3) apart in adjacent rows
    # Actually, for regular hexagons of side 1 touching each other:
    # Horizontal distance between centers in same row: 2 * sqrt(3) / 2 * 2 = ... 
    # Let me think more carefully.
    
    # A regular hexagon with side s has:
    # - Width (flat to flat): s * sqrt(3)
    # - Height (point to point): 2 * s
    # When two hexagons touch edge-to-edge, the distance between centers is 2 * apothem = s * sqrt(3)
    
    # For a hexagonal arrangement:
    # Row 0: centers at (0, 0), (sqrt(3), 0), (2*sqrt(3), 0), ...
    # Row 1: centers at (sqrt(3)/2, 1.5), (3*sqrt(3)/2, 1.5), ...
    # The vertical spacing between rows is 1.5 (for side=1 hexagons)
    # Wait, that's not right either.
    
    # Let me use the standard hexagonal lattice:
    # For hexagons of side s=1:
    # - The distance from center to vertex is 1
    # - The distance from center to edge midpoint is sqrt(3)/2
    # - Two adjacent hexagons (sharing an edge) have centers at distance 2*sqrt(3)/2 = sqrt(3)
    
    # Hexagonal lattice vectors:
    # a1 = (sqrt(3), 0)
    # a2 = (sqrt(3)/2, 3/2)
    
    # So centers are at: i*a1 + j*a2 for integers i, j
    
    sqrt3 = sqrt(3)
    
    # Generate hexagonal lattice points
    def generate_hexagonal_points(n):
        """Generate n points in a hexagonal pattern centered at origin."""
        points = [(0.0, 0.0)]
        if n <= 1:
            return points
        
        # Add points in rings
        ring = 1
        while len(points) < n:
            # Ring r has 6*r points
            for k in range(6 * ring):
                if len(points) >= n:
                    break
                # Position on ring
                # Each side of the hexagonal ring has 'ring' points
                side_idx = k // ring  # 0-5
                pos_on_side = k % ring  # 0 to ring-1
                
                # Directions for the 6 sides
                # Side 0: from (ring, 0) to (ring, ring) in lattice coords
                # Let's use the standard way:
                # Ring r consists of points at hexagonal distance r from origin
                
                # Using axial coordinates:
                # The 6 directions are:
                # (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)
                # For ring r, we go around:
                # Start at (r, 0)
                # Side 0: (r, 0) to (r, -r): step (0, -1), r steps
                # Side 1: (r, -r) to (0, -r): step (-1, 0), r steps  
                # Side 2: (0, -r) to (-r, 0): step (0, 1), r steps
                # Side 3: (-r, 0) to (-r, r): step (1, 0), r steps
                # Side 4: (-r, r) to (0, r): step (0, -1), r steps
                # Side 5: (0, r) to (r, 0): step (1, 0), r steps
                
                # Convert axial to cartesian:
                # x = sqrt(3) * (q + r/2)
                # y = 1.5 * r
                
                q = 0
                rr = 0
                
                # Determine position on ring
                if side_idx == 0:
                    q = ring
                    rr = -pos_on_side
                elif side_idx == 1:
                    q = ring - pos_on_side
                    rr = -ring
                elif side_idx == 2:
                    q = -pos_on_side
                    rr = -ring + pos_on_side
                elif side_idx == 3:
                    q = -ring
                    rr = pos_on_side
                elif side_idx == 4:
                    q = -ring + pos_on_side
                    rr = ring
                elif side_idx == 5:
                    q = pos_on_side
                    rr = ring - pos_on_side
                
                x = sqrt3 * (q + rr / 2.0)
                y = 1.5 * rr
                
                points.append((x, y))
            
            ring += 1
        
        return points[:n]
    
    # Get initial positions
    initial_points = generate_hexagonal_points(N)
    
    # Initial inner hexagons: all at angle 0
    inner = [[p[0], p[1], 0.0] for p in initial_points]
    
    # Compute initial outer hexagon side length
    # The outer hexagon needs to contain all inner hexagons
    # For a hexagon centered at origin with side S, the apothem is S*sqrt(3)/2
    # We need the outer hexagon to contain all inner hexagon vertices
    
    def compute_required_outer_side(inner_hexes, outer_center=(0.0, 0.0), outer_angle=0.0):
        """Compute minimum outer hexagon side to contain all inner hexagons."""
        cx, cy = outer_center
        oa_rad = outer_angle * pi / 180.0
        
        max_dist = 0.0
        for hx, hy, ha in inner_hexes:
            # Get vertices of this inner hexagon
            for i in range(6):
                theta = ha * pi / 180.0 + i * pi / 3.0
                vx = hx + cos(theta)
                vy = hy + sin(theta)
                
                # Distance from outer center
                dx = vx - cx
                dy = vy - cy
                
                # We need to check if this vertex is inside the outer hexagon
                # For outer hexagon with side S, centered at (cx,cy), rotated by outer_angle:
                # Transform vertex to outer hexagon's local frame
                rx = dx * cos(-oa_rad) - dy * sin(-oa_rad)
                ry = dx * sin(-oa_rad) + dy * cos(-oa_rad)
                
                # Check against each edge of outer hexagon
                # Outer hexagon has apothem = S * sqrt(3)/2
                # Edge normals at angles 30, 90, 150, 210, 270, 330 degrees (in local frame)
                for j in range(6):
                    theta_edge = (2 * j + 1) * pi / 6.0
                    nx = cos(theta_edge)
                    ny = sin(theta_edge)
                    
                    # The constraint is: rx * nx + ry * ny <= S * sqrt(3)/2
                    # So S >= (rx * nx + ry * ny) / (sqrt(3)/2)
                    val = (rx * nx + ry * ny) / (sqrt3 / 2.0)
                    if val > max_dist:
                        max_dist = val
        
        return max_dist
    
    # Initial solution
    outer_side = compute_required_outer_side(inner)
    best_solution = (inner.copy(), (0.0, 0.0), outer_side, 0.0)
    best_score = outer_side
    
    # Optimization: use gradient descent to minimize outer side length
    # Variables: positions and angles of inner hexagons
    # We'll use a simple approach: perturb and accept if better
    
    # For efficiency, let's try a few random restarts with local optimization
    
    np.random.seed(42)
    
    for restart in range(5):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # Use the initial hexagonal arrangement
            current_inner = [[p[0], p[1], 0.0] for p in initial_points]
        else:
            # Random initial positions near the hexagonal arrangement
            current_inner = []
            for p in initial_points:
                x = p[0] + np.random.normal(0, 0.5)
                y = p[1] + np.random.normal(0, 0.5)
                angle = np.random.uniform(0, 360)
                current_inner.append([x, y, angle])
        
        # Local optimization using simulated annealing-like approach
        temp = 1.0
        for step in range(2000):
            if time.time() > deadline:
                break
            
            # Pick a random hexagon to move
            idx = np.random.randint(0, N)
            old_x = current_inner[idx][0]
            old_y = current_inner[idx][1]
            old_a = current_inner[idx][2]
            
            # Perturb
            move_size = 0.1 * temp
            new_x = old_x + np.random.normal(0, move_size)
            new_y = old_y + np.random.normal(0, move_size)
            new_a = (old_a + np.random.normal(0, move_size * 10)) % 360
            
            current_inner[idx] = [new_x, new_y, new_a]
            
            # Compute new outer side
            new_side = compute_required_outer_side(current_inner)
            
            if new_side < best_score:
                best_score = new_side
                best_solution = (current_inner.copy(), (0.0, 0.0), new_side, 0.0)
            
            # Accept/reject
            old_side = compute_required_outer_side(
                [current_inner[i] if i != idx else [old_x, old_y, old_a] for i in range(N)]
            )
            
            if new_side < old_side:
                pass  # Accept
            else:
                # Reject with probability
                prob = np.exp(-(new_side - old_side) / max(temp, 0.01))
                if np.random.random() > prob:
                    current_inner[idx] = [old_x, old_y, old_a]
            
            # Cool down
            temp *= 0.999
    
    # Return the best solution found
    inner_best, center_best, side_best, angle_best = best_solution
    return inner_best, center_best, side_best, angle_best

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
