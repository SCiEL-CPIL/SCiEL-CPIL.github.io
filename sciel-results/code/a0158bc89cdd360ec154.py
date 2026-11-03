"""Packing 11 unit hexagons in a regular hexagon using topological graph optimization."""
import numpy as np
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

def hexagon_face_normals(angle_deg):
    """Return the 6 unit outward face normals of a regular hexagon with given rotation."""
    angle_rad = math.radians(angle_deg)
    normals = []
    for i in range(6):
        a = angle_rad + i * math.pi / 3
        # Face normal is perpendicular to edge, pointing outward
        # The face between vertex i and i+1 has normal at angle a + pi/6
        n_angle = a + math.pi / 6
        normals.append((math.cos(n_angle), math.sin(n_angle)))
    return normals

def support_function(points, normals):
    """Compute max dot product of points with each normal direction."""
    max_support = 0.0
    for px, py in points:
        for nx, ny in normals:
            val = px * nx + py * ny
            if val > max_support:
                max_support = val
    return max_support

def min_outer_side(inner_hexagons):
    """Compute minimum outer hexagon side length to contain all inner hexagons.
    Returns (side, cx, cy, angle_deg) where the outer hexagon is centered at origin with angle 0.
    Actually, we need to find the best outer hexagon orientation too.
    """
    # Collect all vertices of all inner hexagons
    all_points = []
    for ix, iy, iangle in inner_hexagons:
        verts = hexagon_vertices(ix, iy, 1.0, iangle)
        all_points.extend(verts)
    
    # The outer hexagon can be rotated. We need to minimize over rotation angle.
    # For a given outer rotation, the required side is max_support / (sqrt(3)/2)
    # where max_support is over the 6 face normals of the outer hexagon.
    
    # Try many outer rotations to find the best
    best_side = float('inf')
    best_angle = 0.0
    for trial in range(360):
        outer_angle = trial * 0.5  # degrees
        normals = hexagon_face_normals(outer_angle)
        sf = support_function(all_points, normals)
        side = sf / (math.sqrt(3) / 2)
        if side < best_side:
            best_side = side
            best_angle = outer_angle
    
    return best_side, 0.0, 0.0, best_angle

def check_no_overlap(inner_hexagons):
    """Check that no two inner hexagons overlap."""
    polys = []
    for ix, iy, iangle in inner_hexagons:
        polys.append(hexagon_vertices(ix, iy, 1.0, iangle))
    
    def point_in_polygon(px, py, polygon):
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
    
    def segments_intersect(p1, p2, p3, p4):
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    for i in range(N):
        for j in range(i+1, N):
            # Check if any vertex of poly i is inside poly j
            for v in polys[i]:
                if point_in_polygon(v[0], v[1], polys[j]):
                    return False
            # Check if any vertex of poly j is inside poly i
            for v in polys[j]:
                if point_in_polygon(v[0], v[1], polys[i]):
                    return False
            # Check edge intersections
            for k in range(6):
                for l in range(6):
                    if segments_intersect(polys[i][k], polys[i][(k+1)%6], polys[j][l], polys[j][(l+1)%6]):
                        return False
    return True

def construct():
    """
    Construct 11 unit hexagons packed in a regular hexagon, minimizing outer side length.
    Returns: (inner, center, side, angle)
    """
    N_local = 11
    sqrt3 = math.sqrt(3)
    
    # Initial configuration: asymmetric defect cluster
    # Base: 1-2-3-2-1 lattice (9 hexagons) + 2 additional
    # Introduce asymmetry by rotating one hexagon 30 degrees
    
    # Standard honeycomb positions for side-1 hexagons
    # Distance between adjacent centers: sqrt(3)
    d = sqrt3
    
    # 1-2-3-2-1 arrangement
    base_positions = [
        (0.0, 2*d, 0.0),           # Row 0: 1 hex
        (-d/2, d, 0.0),            # Row 1: 2 hexes
        (d/2, d, 0.0),
        (-d, 0.0, 0.0),            # Row 2: 3 hexes
        (0.0, 0.0, 0.0),
        (d, 0.0, 0.0),
        (-d/2, -d, 0.0),           # Row 3: 2 hexes
        (d/2, -d, 0.0),
        (0.0, -2*d, 0.0),          # Row 4: 1 hex
    ]
    
    # Add 2 more hexagons at asymmetric positions
    # Place them to break symmetry - one at top-right, one at bottom-left
    base_positions.append((1.5*d, 1.5*d, 0.0))   # Top-right area
    base_positions.append((-1.5*d, -1.5*d, 0.0)) # Bottom-left area
    
    # Introduce defect: rotate the center hexagon by 30 degrees
    base_positions[4] = (0.0, 0.0, 30.0)
    
    # Convert to numpy array for optimization
    # Parameters: for each hex, (x, y, angle_deg)
    x0 = np.array([p for pos in base_positions for p in pos], dtype=float)
    
    def objective(params):
        """Objective: outer side length + penalty for overlaps."""
        inner = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(N_local)]
        
        # Compute minimum outer side
        side, _, _, _ = min_outer_side(inner)
        
        # Check for overlaps - add penalty
        overlap_penalty = 0.0
        polys = [hexagon_vertices(ix, iy, iangle) for ix, iy, iangle in inner]
        
        def point_in_polygon(px, py, polygon):
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
        
        def segments_intersect(p1, p2, p3, p4):
            def ccw(a, b, c):
                return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
            return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
        
        for i in range(N_local):
            for j in range(i+1, N_local):
                overlap = False
                for v in polys[i]:
                    if point_in_polygon(v[0], v[1], polys[j]):
                        overlap = True
                        break
                if not overlap:
                    for v in polys[j]:
                        if point_in_polygon(v[0], v[1], polys[i]):
                            overlap = True
                            break
                if not overlap:
                    for k in range(6):
                        for l in range(6):
                            if segments_intersect(polys[i][k], polys[i][(k+1)%6], polys[j][l], polys[j][(l+1)%6]):
                                overlap = True
                                break
                        if overlap:
                            break
                if overlap:
                    overlap_penalty += 10.0
                    break
        
        return side + overlap_penalty
    
    # Gradient-free optimization with multiple restarts
    best_params = x0.copy()
    best_val = objective(x0)
    
    # Local search via Nelder-Mead
    from scipy.optimize import minimize
    
    np.random.seed(42)
    
    for restart in range(8):
        if restart == 0:
            start = x0.copy()
        else:
            start = x0 + np.random.normal(0, 0.3, size=x0.shape)
            # Keep angles in reasonable range
            for i in range(N_local):
                start[3*i+2] = start[3*i+2] % 360
        
        try:
            result = minimize(
                objective,
                start,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-3, 'fatol': 1e-3, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Extract results
    inner = []
    for i in range(N_local):
        inner.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
    
    # Compute final outer hexagon
    side, cx, cy, angle = min_outer_side(inner)
    
    # Final validation
    if not check_no_overlap(inner):
        # Fallback: spread out horizontally
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N_local // 2), N_local - N_local // 2)], (0.0, 0.0), 40.0, 0.0
    
    return inner, (cx, cy), side, angle

# EVOLVE-BLOCK-START
# (construct function defined above)
# EVOLVE-BLOCK-END
