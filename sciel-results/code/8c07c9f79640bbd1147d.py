import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    # Geometry helpers
    def get_hex_vertices(cx, cy, side, angle_deg):
        """
        Returns vertices of a regular hexagon.
        Center: (cx, cy)
        Side: side length
        Angle: rotation in degrees.
        Vertex i is at angle: angle + i * 60 degrees.
        Distance from center to vertex is 'side' (circumradius = side for regular hex).
        """
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle_rad + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return np.array(verts)

    def point_in_hexagon(px, py, hx, hy, side, angle_deg):
        """
        Check if point (px, py) is inside hexagon centered at (hx, hy).
        Uses half-plane intersection logic.
        """
        verts = get_hex_vertices(hx, hy, side, angle_deg)
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            # Edge vector
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            # Vector from p1 to center
            vx_c = hx - p1[0]
            vy_c = hy - p1[1]
            # Vector from p1 to point
            vx_p = px - p1[0]
            vy_p = py - p1[1]
            
            # Cross product z-component
            cross_c = ex * vy_c - ey * vx_c
            cross_p = ex * vy_p - ey * vx_p
            
            if cross_c == 0:
                continue
            if (cross_c > 0 and cross_p < 0) or (cross_c < 0 and cross_p > 0):
                return False
        return True

    def hexagons_overlap(v1, v2, side1, angle1, side2, angle2, tol=1e-9):
        """
        Check if two hexagons overlap.
        v1, v2: centers (x,y)
        """
        # 1. Vertex containment
        verts1 = get_hex_vertices(v1[0], v1[1], side1, angle1)
        for v in verts1:
            if point_in_hexagon(v[0], v[1], v2[0], v2[1], side2, angle2):
                return True
        
        verts2 = get_hex_vertices(v2[0], v2[1], side2, angle2)
        for v in verts2:
            if point_in_hexagon(v[0], v[1], v1[0], v1[1], side1, angle1):
                return True
        
        # 2. Edge intersections
        def segments_intersect(p1, p2, p3, p4):
            def ccw(a, b, c):
                return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
            return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

        for i in range(6):
            p1 = verts1[i]
            p2 = verts1[(i+1)%6]
            for j in range(6):
                p3 = verts2[j]
                p4 = verts2[(j+1)%6]
                if segments_intersect(p1, p2, p3, p4):
                    return True
        return False

    def min_outer_side_for_config(inner_hexes):
        """
        Given a valid set of non-overlapping inner hexagons,
        calculate the minimum side length of the outer hexagon (centered at 0,0, angle 0)
        that contains all of them.
        
        For a regular hexagon centered at origin with side S and angle 0 (vertices at 0, 60...):
        The boundary is defined by 6 lines.
        A point (x,y) is inside if for all 6 normals n_i, dot((x,y), n_i) <= S * cos(30) = S * sqrt(3)/2 ?
        Actually, let's derive the support function.
        For a regular hexagon with circumradius R (which equals side length S),
        the distance from center to the midpoint of an edge (apothem) is S * sqrt(3)/2.
        The hexagon is the intersection of 6 half-planes.
        The normals to the edges are at angles 30, 90, 150, 210, 270, 330 degrees.
        Let's verify:
        Vertices at 0, 60, 120...
        Edge 0 connects (1,0) and (1/2, sqrt(3)/2).
        Midpoint: (3/4, sqrt(3)/4).
        Normal vector pointing outward: (sqrt(3)/2, 1/2) -> angle 30 deg.
        Distance from origin to line: dot(mid, normal) = 3/4*sqrt(3)/2 + sqrt(3)/4*1/2 = 3sqrt(3)/8 + sqrt(3)/8 = 4sqrt(3)/8 = sqrt(3)/2.
        So, for a point (x,y) to be inside, we must have:
        x * cos(30 + i*60) + y * sin(30 + i*60) <= S * sqrt(3)/2
        for i in 0..5.
        
        Thus, S >= (x * cos(theta_i) + y * sin(theta_i)) * 2 / sqrt(3) for all vertices of all inner hexagons.
        """
        max_proj = 0.0
        # Normals at 30, 90, 150, 210, 270, 330 degrees
        normal_angles = [math.radians(30 + i * 60) for i in range(6)]
        
        for ih in inner_hexes:
            cx, cy, angle = ih
            verts = get_hex_vertices(cx, cy, 1.0, angle)
            for v in verts:
                for na in normal_angles:
                    proj = v[0] * math.cos(na) + v[1] * math.sin(na)
                    if proj > max_proj:
                        max_proj = proj
        
        # S * sqrt(3)/2 >= max_proj  => S >= max_proj * 2 / sqrt(3)
        return max_proj * 2.0 / math.sqrt(3)

    # --- Search Strategy ---
    # We assume the optimal configuration has the outer hexagon centered at (0,0) and angle 0.
    # We search for inner hexagon positions and angles.
    
    # Discretization
    # Based on honeycomb, centers are roughly in [-3, 3] x [-3, 3].
    # Let's search x, y in [-3.5, 3.5] with step 0.1
    # Angles: 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330
    # 12 hexagons. 71 x 71 x 12 possibilities per hexagon is too large for full cartesian product.
    # We need a smarter search or a smaller grid.
    
    # Let's try a coarser grid first to find a good candidate, then refine.
    # Or use a local search from a honeycomb initial guess.
    
    # Initial guess: Honeycomb
    sqrt3 = math.sqrt(3)
    coords = [
        (-2, 0), (0, 0), (2, 0),
        (-1, sqrt3), (1, sqrt3), (3, sqrt3),
        (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),
        (-1, 3*sqrt3), (1, 3*sqrt3), (3, 3*sqrt3)
    ]
    # Center the block
    cx_avg = sum(c[0] for c in coords) / N
    cy_avg = sum(c[1] for c in coords) / N
    init_inner = [[c[0] - cx_avg, c[1] - cy_avg, 0.0] for c in coords]
    
    # Validate initial config
    if any(hexagons_overlap(init_inner[i][:2], init_inner[j][:2], 1.0, init_inner[i][2], 1.0, init_inner[j][2]) for i in range(N) for j in range(i+1, N)):
        # Fallback if honeycomb is invalid (should not happen)
        init_inner = [[i*2, 0, 0] for i in range(N)]
        
    best_side = min_outer_side_for_config(init_inner)
    best_inner = init_inner
    
    # Local Search / Grid Refinement
    # We will perform a local search around the current best solution.
    # Step size for x,y: 0.05
    # Step size for angle: 15 degrees (or 30)
    
    step_xy = 0.05
    step_angle = 15.0
    max_iterations = 5000
    iteration = 0
    
    # Precompute normal angles for speed
    normal_angles = [math.radians(30 + i * 60) for i in range(6)]
    
    def is_valid_config(config):
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_overlap(config[i][:2], config[j][:2], 1.0, config[i][2], 1.0, config[j][2]):
                    return False
        return True

    def get_side(config):
        max_proj = 0.0
        for ih in config:
            cx, cy, angle = ih
            # Fast vertex calculation
            angle_rad = math.radians(angle)
            # Vertices
            # v_k = (cx + cos(a + k*pi/3), cy + sin(a + k*pi/3))
            for k in range(6):
                theta = angle_rad + k * math.pi / 3
                vx = cx + math.cos(theta)
                vy = cy + math.sin(theta)
                for na in normal_angles:
                    proj = vx * math.cos(na) + vy * math.sin(na)
                    if proj > max_proj:
                        max_proj = proj
        return max_proj * 2.0 / math.sqrt(3)

    # Simple Hill Climbing
    while time.time() < deadline - 5.0:
        improved = False
        # Try moving each hexagon
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            x, y, a = best_inner[i]
            
            # Try small perturbations
            # X/Y moves
            for dx in [-step_xy, 0, step_xy]:
                for dy in [-step_xy, 0, step_xy]:
                    if dx == 0 and dy == 0:
                        continue
                    new_config = best_inner.copy()
                    new_config[i] = [x + dx, y + dy, a]
                    if is_valid_config(new_config):
                        new_side = get_side(new_config)
                        if new_side < best_side - 1e-5:
                            best_side = new_side
                            best_inner = new_config
                            improved = True
                            
            # Angle moves
            for da in [-step_angle, step_angle]:
                new_a = (a + da) % 360
                new_config = best_inner.copy()
                new_config[i] = [x, y, new_a]
                if is_valid_config(new_config):
                    new_side = get_side(new_config)
                    if new_side < best_side - 1e-5:
                        best_side = new_side
                        best_inner = new_config
                        improved = True
                        
            # Try moving pairs? No, too expensive.
            
        # If no improvement, try larger steps or reset to a different random start?
        # For now, if no improvement, we might be stuck in a local minimum.
        # Let's try a few random restarts if we have time.
        if not improved:
            # Try a random restart
            if time.time() < deadline - 20.0:
                # Generate a random valid-ish config?
                # Just shuffle the honeycomb slightly
                rand_inner = []
                for c in coords:
                    rx = c[0] - cx_avg + np.random.uniform(-0.5, 0.5)
                    ry = c[1] - cy_avg + np.random.uniform(-0.5, 0.5)
                    ra = np.random.choice([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
                    rand_inner.append([rx, ry, ra])
                
                if is_valid_config(rand_inner):
                    rand_side = get_side(rand_inner)
                    if rand_side < best_side:
                        best_side = rand_side
                        best_inner = rand_inner
                        improved = True
                if not improved:
                    # If stuck, break to save time? Or continue with smaller steps?
                    # Let's reduce step size
                    step_xy *= 0.9
                    step_angle *= 0.9
                    if step_xy < 0.01:
                        break

    # Final result
    inner = []
    for ih in best_inner:
        inner.append([ih[0], ih[1], ih[2]])
    
    return inner, (0.0, 0.0), best_side, 0.0

# EVOLVE-BLOCK-END
