import time
import numpy as np

N = 12

def _hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle_rad = np.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees (flat-top orientation)
    # For pointy-top, offset by 30 degrees. We'll use the standard math convention.
    # A regular hexagon with "pointy top" has vertices at 30, 90, 150, 210, 270, 330 degrees.
    # A regular hexagon with "flat top" has vertices at 0, 60, 120, 180, 240, 300 degrees.
    # The problem says "regular hexagons" without specifying orientation, so rotation is a free parameter.
    # We'll define the base hexagon with vertices at angles 0, 60, ..., 300 degrees (flat-top).
    angles = np.radians(np.arange(6) * 60.0) + angle_rad
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack([x, y])

def _point_in_hexagon(pt, center, side, angle_deg):
    """Check if a point is inside or on the boundary of a regular hexagon."""
    # Transform point to hexagon's local coordinate system
    dx = pt[0] - center[0]
    dy = pt[1] - center[1]
    angle_rad = np.radians(angle_deg)
    # Rotate point by -angle_rad
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # In the local frame, a flat-top hexagon of side 's' has vertices at:
    # (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # The hexagon is the intersection of 6 half-planes.
    # For a flat-top hexagon, the constraints are:
    # |lx| <= s
    # |ly| <= s * sqrt(3) / 2
    # |lx| + |ly| / sqrt(3) <= s  (this isn't quite right, let me think more carefully)
    
    # Actually, let's use the 6 half-plane constraints directly.
    # The 6 sides of a flat-top hexagon (centered at origin, side s) are:
    # 1. Right: x = s  ->  x <= s
    # 2. Upper-right: x/2 + y*sqrt(3)/2 = s  ->  x + y*sqrt(3) <= 2s
    # 3. Upper-left: -x/2 + y*sqrt(3)/2 = s  ->  -x + y*sqrt(3) <= 2s
    # 4. Left: x = -s  ->  -x <= s  i.e.  x >= -s
    # 5. Lower-left: -x/2 - y*sqrt(3)/2 = s  ->  -x - y*sqrt(3) <= 2s
    # 6. Lower-right: x/2 - y*sqrt(3)/2 = s  ->  x - y*sqrt(3) <= 2s
    
    # So the constraints are:
    # lx <= s
    # lx >= -s
    # lx + sqrt(3)*ly <= 2s
    # -lx + sqrt(3)*ly <= 2s
    # -lx - sqrt(3)*ly <= 2s
    # lx - sqrt(3)*ly <= 2s
    
    s = side
    sqrt3 = np.sqrt(3.0)
    
    if lx > s + 1e-12: return False
    if lx < -s - 1e-12: return False
    if lx + sqrt3 * ly > 2 * s + 1e-12: return False
    if -lx + sqrt3 * ly > 2 * s + 1e-12: return False
    if -lx - sqrt3 * ly > 2 * s + 1e-12: return False
    if lx - sqrt3 * ly > 2 * s + 1e-12: return False
    
    return True

def _max_projection(vertices, outer_center, outer_side, outer_angle_deg):
    """
    Given vertices of the inner cluster (all points), compute the minimum outer hexagon side length
    needed to contain them, assuming the outer hexagon is centered at outer_center with rotation outer_angle_deg.
    
    Actually, we want to compute: given a set of points and an outer hexagon (center, side, angle),
    check if all points are inside. But for optimization, we need a smooth objective.
    
    Let's reformulate: given inner hexagon centers and angles, and outer hexagon (center, angle),
    find the minimum outer side length S such that all inner hexagons are contained.
    
    For each inner hexagon, we can compute its 6 vertices. Then for the outer hexagon, we need
    all 72 vertices to be inside.
    
    The minimum S such that a point p is inside a hexagon centered at c with rotation a is:
    S >= max over the 6 constraints of the "distance" to each side.
    
    For a flat-top hexagon centered at origin, side S, the 6 constraints for point (x,y) are:
    x <= S, -x <= S, x + sqrt(3)y <= 2S, -x + sqrt(3)y <= 2S, -x - sqrt(3)y <= 2S, x - sqrt(3)y <= 2S
    
    So S >= max(x, -x, (x+sqrt(3)y)/2, (-x+sqrt(3)y)/2, (-x-sqrt(3)y)/2, (x-sqrt(3)y)/2)
    
    For a rotated hexagon, we first rotate the point by -angle, then apply the above.
    
    So for each point, the required S is:
    S_i = max over 6 directions of (dot product with direction vector)
    
    Specifically, if we rotate point p by -a to get p', then:
    S_i = max(p'_x, -p'_x, (p'_x + sqrt(3)*p'_y)/2, (-p'_x + sqrt(3)*p'_y)/2, (-p'_x - sqrt(3)*p'_y)/2, (p'_x - sqrt(3)*p'_y)/2)
    
    This can be simplified. The 6 values are:
    p'_x, -p'_x, p'_x/2 + sqrt(3)*p'_y/2, -p'_x/2 + sqrt(3)*p'_y/2, -p'_x/2 - sqrt(3)*p'_y/2, p'_x/2 - sqrt(3)*p'_y/2
    
    Note that max(p'_x, -p'_x) = |p'_x|
    And the other 4 are: p'_x/2 + sqrt(3)*p'_y/2, -p'_x/2 + sqrt(3)*p'_y/2, -p'_x/2 - sqrt(3)*p'_y/2, p'_x/2 - sqrt(3)*p'_y/2
    
    The max of these 6 is the support function of the hexagon.
    
    Actually, the required S for a point p relative to a hexagon centered at c with rotation a is:
    S = max over 6 unit normals n_k of (p - c) · n_k
    where n_k are the outward unit normals of the 6 sides.
    
    For a flat-top hexagon, the outward unit normals are at angles 0, 60, 120, 180, 240, 300 degrees.
    n_k = (cos(k*60°), sin(k*60°)) for k=0..5
    
    Wait, let me verify. The right side of the hexagon is x = S, so the outward normal is (1, 0). Correct.
    The upper-right side: the line is x + sqrt(3)y = 2S. The normal vector is (1, sqrt(3)), normalized: (1, sqrt(3))/2 = (cos(60°), sin(60°)). Correct.
    
    So for a rotated hexagon by angle a, the normals are at angles a + k*60° for k=0..5.
    
    The required S for point p is:
    S_p = max_k (p - c) · (cos(a + k*60°), sin(a + k*60°))
    
    And the overall S is max over all points p of S_p.
    """
    # This function is for checking, but for optimization we'll use a different approach.
    pass

def _compute_required_side(inner_hexes, outer_center, outer_angle_deg):
    """
    Compute the minimum outer hexagon side length to contain all inner hexagons.
    inner_hexes: list of (cx, cy, angle_deg) for each inner unit hexagon
    outer_center: (cx, cy)
    outer_angle_deg: rotation of outer hexagon
    Returns: minimum side length S
    """
    max_s = 0.0
    sqrt3 = np.sqrt(3.0)
    
    # Outer hexagon normals (unit vectors) for a flat-top hexagon rotated by outer_angle_deg
    outer_angle_rad = np.radians(outer_angle_deg)
    normal_angles = outer_angle_rad + np.radians(np.arange(6) * 60.0)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    for (cx, cy, angle_deg) in inner_hexes:
        # Compute 6 vertices of this inner hexagon
        verts = _hexagon_vertices(cx, cy, 1.0, angle_deg)
        
        for pt in verts:
            # Vector from outer center to this point
            d = pt - np.array(outer_center)
            # Project onto each normal
            projections = normals @ d
            s_req = np.max(projections)
            if s_req > max_s:
                max_s = s_req
    
    return max_s

def _objective(params, inner_hexes_base):
    """
    params: [outer_cx, outer_cy, outer_angle_deg]
    inner_hexes_base: fixed positions of inner hexagons (relative to some origin)
    We'll translate the inner hexagons so their centroid is at the origin, then the outer center is the translation.
    
    Actually, let's define: the inner hexagons have fixed relative positions. The outer center is a parameter.
    The outer angle is a parameter. We compute the required S.
    """
    outer_cx = params[0]
    outer_cy = params[1]
    outer_angle = params[2]
    
    return _compute_required_side(inner_hexes_base, (outer_cx, outer_cy), outer_angle)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second limit, with safety margin
    
    # Define a dense hexagonal lattice arrangement of 12 unit hexagons.
    # In a hexagonal lattice, each hexagon touches 6 neighbors.
    # For 12 hexagons, a good compact arrangement is a "flower" of 7 (center + 6 around)
    # plus 5 more. Or a 2x6 rectangle in hex lattice coordinates.
    # Or a hexagonal shape with side 2 (which has 1+6+12=19, too many).
    # Let's try a few topologies and pick the best.
    
    # Topology 1: 2x6 grid in hex lattice
    # Hex lattice coordinates: (col, row) where odd rows are offset by 0.5
    # For a 2x6 arrangement:
    # Row 0: 6 hexagons at x = 0, 2, 4, 6, 8, 10 (spacing 2 for flat-top)
    # Row 1: 6 hexagons at x = 1, 3, 5, 7, 9, 11 (offset by 1)
    # y spacing: sqrt(3) for flat-top hexagons
    
    # Actually, for flat-top hexagons of side 1:
    # Horizontal spacing: 2 (width of hexagon)
    # Vertical spacing: sqrt(3) (height of hexagon)
    # Odd rows offset by 1 in x
    
    def make_hex_lattice_topology(cols, rows, offset_odd=True):
        hexes = []
        sqrt3 = np.sqrt(3.0)
        for r in range(rows):
            for c in range(cols):
                x = c * 2.0
                if offset_odd and r % 2 == 1:
                    x += 1.0
                y = r * sqrt3
                hexes.append([x, y, 0.0])
        return hexes
    
    # Topology 2: Flower (7) + 5 more
    # Center at (0,0), 6 around at distance 2 (for flat-top, neighbors are at:
    # (2,0), (1, sqrt(3)), (-1, sqrt(3)), (-2, 0), (-1, -sqrt(3)), (1, -sqrt(3))
    # That's 7 hexagons. Add 5 more in a compact way.
    
    def make_flower_topology():
        sqrt3 = np.sqrt(3.0)
        hexes = [[0.0, 0.0, 0.0]]  # center
        # 6 neighbors
        neighbors = [
            [2.0, 0.0],
            [1.0, sqrt3],
            [-1.0, sqrt3],
            [-2.0, 0.0],
            [-1.0, -sqrt3],
            [1.0, -sqrt3]
        ]
        for (x, y) in neighbors:
            hexes.append([x, y, 0.0])
        # Add 5 more: extend the flower
        # Add at (3, 0), (2, sqrt3), (0, 2*sqrt3), (-2, sqrt3), (-3, 0)
        # Let's try a more symmetric extension
        extra = [
            [3.0, 0.0],
            [2.0, sqrt3],
            [0.0, 2.0 * sqrt3],
            [-2.0, sqrt3],
            [-3.0, 0.0]
        ]
        for (x, y) in extra:
            hexes.append([x, y, 0.0])
        return hexes
    
    # Generate candidate topologies
    topologies = []
    
    # 2x6 grid
    topologies.append(make_hex_lattice_topology(6, 2))
    # 3x4 grid
    topologies.append(make_hex_lattice_topology(4, 3))
    # Flower + 5
    topologies.append(make_flower_topology())
    
    # Try a 3x4 with different offset
    topologies.append(make_hex_lattice_topology(4, 3, offset_odd=False))
    
    best_result = None
    best_score = float('inf')
    
    for top_idx, hexes in enumerate(topologies):
        if time.time() > deadline - 10.0:
            break
        
        # Center the topology at origin
        xs = [h[0] for h in hexes]
        ys = [h[1] for h in hexes]
        cx_mean = np.mean(xs)
        cy_mean = np.mean(ys)
        hexes_centered = [[h[0] - cx_mean, h[1] - cy_mean, h[2]] for h in hexes]
        
        # Optimize outer center and angle
        # Since the topology is centered at origin, a good initial guess for outer center is (0,0)
        # and outer angle is 0.
        
        # Use a simple grid search + local refinement
        best_local = float('inf')
        best_params = None
        
        # Grid search over outer angle (0 to 60 degrees, since hexagon has 60° symmetry)
        # and outer center (small range around origin)
        
        # First, try angle = 0 and center = (0,0)
        s0 = _compute_required_side(hexes_centered, (0.0, 0.0), 0.0)
        best_local = s0
        best_params = [0.0, 0.0, 0.0]
        
        # Try different angles
        for angle_deg in np.arange(0.0, 60.0, 5.0):
            if time.time() > deadline - 10.0:
                break
            s = _compute_required_side(hexes_centered, (0.0, 0.0), angle_deg)
            if s < best_local:
                best_local = s
                best_params = [0.0, 0.0, angle_deg]
        
        # Try small translations
        for dx in np.arange(-2.0, 2.1, 0.5):
            for dy in np.arange(-2.0, 2.1, 0.5):
                if time.time() > deadline - 10.0:
                    break
                for angle_deg in np.arange(0.0, 60.0, 10.0):
                    s = _compute_required_side(hexes_centered, (dx, dy), angle_deg)
                    if s < best_local:
                        best_local = s
                        best_params = [dx, dy, angle_deg]
        
        # Local refinement using gradient-free optimization
        # Use Nelder-Mead or a simple hill-climbing
        if time.time() < deadline - 15.0:
            # Simple coordinate descent
            current = best_params.copy()
            current_s = best_local
            step_sizes = [0.1, 0.1, 1.0]
            
            for iteration in range(200):
                if time.time() > deadline - 10.0:
                    break
                improved = False
                for i in range(3):
                    for direction in [-1, 1]:
                        new_params = current.copy()
                        new_params[i] += direction * step_sizes[i]
                        new_s = _compute_required_side(hexes_centered, (new_params[0], new_params[1]), new_params[2])
                        if new_s < current_s - 1e-10:
                            current = new_params
                            current_s = new_s
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    step_sizes = [s * 0.5 for s in step_sizes]
                    if step_sizes[0] < 1e-6:
                        break
            
            if current_s < best_local:
                best_local = current_s
                best_params = current
        
        if best_local < best_score:
            best_score = best_local
            best_result = (hexes_centered, best_params)
        
        if time.time() > deadline - 20.0:
            break
    
    if best_result is None:
        # Fallback to the seed program
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    hexes_centered, best_params = best_result
    outer_cx = best_params[0]
    outer_cy = best_params[1]
    outer_angle = best_params[2]
    outer_side = best_score
    
    # The inner hexagons are defined relative to the outer center.
    # We need to return them in absolute coordinates.
    # The current setup has inner hexagons centered at origin, and outer center at (outer_cx, outer_cy).
    # So the absolute position of each inner hexagon is (h[0] + outer_cx, h[1] + outer_cy, h[2]).
    
    inner_abs = [[h[0] + outer_cx, h[1] + outer_cy, h[2]] for h in hexes_centered]
    
    # Verify the result
    # Check that all inner hexagons are disjoint and inside the outer hexagon
    # For now, just return the result
    
    return inner_abs, (outer_cx, outer_cy), outer_side, outer_angle
