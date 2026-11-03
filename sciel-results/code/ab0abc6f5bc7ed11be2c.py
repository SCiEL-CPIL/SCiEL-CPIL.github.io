"""Hexagon packing: 12 unit regular hexagons inside a regular hexagon, minimize outer side."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# Helper: get vertices of a regular hexagon with side length s, center (cx, cy), rotation angle (radians)
def hex_vertices(cx, cy, s, angle):
    """Return 6 vertices of a regular hexagon."""
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees from center
    # For a regular hexagon with side length s, the distance from center to vertex is s
    angles = angle + np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    xs = cx + s * np.cos(angles)
    ys = cy + s * np.sin(angles)
    return np.column_stack([xs, ys])

def hex_edge_normals(vertices):
    """Return outward normals and edge midpoints for a convex polygon."""
    n = len(vertices)
    normals = np.zeros((n, 2))
    midpoints = np.zeros((n, 2))
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        # Outward normal: rotate edge 90 degrees clockwise, then normalize
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal /= norm
        normals[i] = normal
        midpoints[i] = (p1 + p2) / 2.0
    return normals, midpoints

def point_in_hexagon(pt, hex_center, hex_side, hex_angle):
    """Check if point pt is inside or on the boundary of a regular hexagon."""
    # Transform point to hexagon's local coordinate system
    dx = pt[0] - hex_center[0]
    dy = pt[1] - hex_center[1]
    # Rotate by -angle
    cos_a = np.cos(-hex_angle)
    sin_a = np.sin(-hex_angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # In local coords, hexagon has vertices at angles 0,60,120,180,240,300 with radius = side
    # The hexagon is the intersection of 6 half-planes
    # For a regular hexagon with side s, the apothem (distance from center to edge) = s * sqrt(3)/2
    apothem = hex_side * np.sqrt(3) / 2.0
    
    # Check against each edge
    for k in range(6):
        angle_k = k * np.pi / 3.0
        # Normal direction for edge k
        nx = np.cos(angle_k)
        ny = np.sin(angle_k)
        # The constraint: nx * lx + ny * ly <= apothem
        if nx * lx + ny * ly > apothem + 1e-9:
            return False
    return True

def hexagon_penetration(v1, v2, s1, a1, s2, a2):
    """
    Compute the minimum translation distance to separate two convex hexagons.
    v1, v2: centers
    s1, s2: side lengths
    a1, a2: rotation angles (radians)
    Returns the penetration depth (0 if no overlap, positive if overlapping).
    Uses separating axis theorem.
    """
    verts1 = hex_vertices(v1[0], v1[1], s1, a1)
    verts2 = hex_vertices(v2[0], v2[1], s2, a2)
    
    normals1, _ = hex_edge_normals(verts1)
    normals2, _ = hex_edge_normals(verts2)
    
    all_normals = np.vstack([normals1, normals2])
    
    max_overlap = 0.0
    for normal in all_normals:
        # Project both hexagons onto this normal
        proj1 = np.dot(verts1, normal)
        proj2 = np.dot(verts2, normal)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for separation
        if max1 < min2 or max2 < min1:
            return 0.0  # Separated
        
        # Overlap on this axis
        overlap = min(max1, max2) - max(min1, min2)
        if overlap > max_overlap:
            max_overlap = overlap
    
    return max_overlap

def outer_hexagon_constraint(v, s, a, outer_center, outer_side, outer_angle):
    """
    Check if inner hexagon (center v, side s, angle a) is inside outer hexagon.
    Returns penetration depth (0 if inside, positive if penetrating).
    """
    inner_verts = hex_vertices(v[0], v[1], s, a)
    outer_verts = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # For each vertex of inner hexagon, check if it's inside outer hexagon
    # Use the half-plane representation of the outer hexagon
    outer_normals, _ = hex_edge_normals(outer_verts)
    
    max_pen = 0.0
    for i in range(6):
        pt = inner_verts[i]
        normal = outer_normals[i]
        midpoint = (outer_verts[i] + outer_verts[(i+1)%6]) / 2.0
        
        # Signed distance from pt to the edge line
        # The edge line passes through midpoint with normal pointing outward
        dist = np.dot(pt - midpoint, normal)
        # If dist > 0, the point is outside the half-plane
        if dist > max_pen:
            max_pen = dist
    
    return max_pen

def objective(params, outer_side, outer_center, outer_angle):
    """
    Objective function: minimize outer_side such that all inner hexagons fit.
    params: flattened array of [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11]
    Each angle is in degrees.
    """
    # Unpack
    positions = []
    angles_rad = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a_deg = params[3*i+2]
        positions.append(np.array([x, y]))
        angles_rad.append(a_deg * np.pi / 180.0)
    
    unit_side = 1.0
    
    # Penalty for inner hexagons penetrating the outer hexagon
    pen_outer = 0.0
    for i in range(N):
        pen = outer_hexagon_constraint(positions[i], unit_side, angles_rad[i], 
                                       outer_center, outer_side, outer_angle)
        if pen > 0:
            pen_outer += pen ** 2
    
    # Penalty for inner hexagons overlapping each other
    pen_inner = 0.0
    for i in range(N):
        for j in range(i+1, N):
            pen = hexagon_penetration(positions[i], positions[j], unit_side, angles_rad[i], 
                                       unit_side, angles_rad[j])
            if pen > 0:
                pen_inner += pen ** 2
    
    return pen_outer + pen_inner

def construct():
    """
    Construct 12 unit hexagons inside a regular hexagon, minimizing outer side length.
    Returns: (inner, center, side, angle)
    inner: list of [x, y, angle_degrees] for each hexagon
    center: (x, y) of outer hexagon
    side: side length of outer hexagon
    angle: rotation angle of outer hexagon (degrees)
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Initial guess: hexagonal lattice arrangement
    # Place hexagons in a hexagonal pattern
    # For 12 hexagons, we can use a centered hexagonal arrangement
    # Rows: 3, 4, 3, 2 or similar
    
    # Let's try a simple grid-like arrangement first
    # Hexagonal lattice spacing: for unit hexagons, the distance between centers in adjacent rows
    # For a flat-top hexagon, the horizontal spacing is 2 * apothem = 2 * sqrt(3)/2 * 1 = sqrt(3)
    # Wait, for a regular hexagon with side 1:
    # - Width (flat to flat) = 2 * apothem = 2 * (sqrt(3)/2) * 1 = sqrt(3)
    # - Height (point to point) = 2 * 1 = 2 (if pointy-top) or width = 2 (if flat-top)
    
    # Let's use pointy-top orientation (angle = 0 means a vertex points up)
    # Actually, let's define angle=0 as having a vertex at angle 0 (pointing right)
    # For hexagonal packing, we want to align them properly
    
    # Initial placement: hexagonal lattice
    # For unit hexagons in a hexagonal lattice:
    # Horizontal distance between centers in same row: 2 * apothem = sqrt(3)
    # Vertical distance between rows: 1.5 (for pointy-top) or 3/2 * sqrt(3) ... let me think
    
    # For a regular hexagon with side s=1, pointy-top:
    # The vertices are at angles 30, 90, 150, 210, 270, 330 degrees from center
    # Height = 2 * s = 2
    # Width = sqrt(3) * s = sqrt(3)
    
    # For hexagonal packing of pointy-top hexagons:
    # Row spacing (vertical): 1.5 * s = 1.5
    # Horizontal spacing in same row: sqrt(3) * s = sqrt(3)
    # Offset between rows: sqrt(3)/2
    
    # Let's place them in rows: 2, 3, 3, 3, 1 or some arrangement that fits 12
    
    # Simpler: just use a 3x4 grid with hexagonal offset
    positions_init = []
    angles_init = []
    
    # Use a hexagonal lattice pattern
    # Rows of 4, 3, 4, 1 = 12
    row_sizes = [4, 3, 4, 1]
    row_y = 0.0
    for r, size in enumerate(row_sizes):
        for c in range(size):
            x = c * np.sqrt(3)
            y = row_y
            positions_init.append([x, y])
            angles_init.append(0.0)
        row_y += 1.5
    
    # Center the arrangement
    xs = [p[0] for p in positions_init]
    ys = [p[1] for p in positions_init]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    
    params_init = []
    for i in range(N):
        params_init.append(positions_init[i][0] - cx)
        params_init.append(positions_init[i][1] - cy)
        params_init.append(angles_init[i])
    
    params_init = np.array(params_init)
    
    # Start with a large outer hexagon and try to shrink it
    # Initial outer side estimate
    outer_side = 5.0
    outer_center = [0.0, 0.0]
    outer_angle = 0.0
    
    best_params = params_init.copy()
    best_outer_side = outer_side
    best_penalty = np.inf
    
    # Iteratively try to reduce outer side
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Check if current configuration fits in outer_side
        penalty = objective(best_params, outer_side, outer_center, outer_angle)
        
        if penalty < best_penalty:
            best_penalty = penalty
        
        if penalty < 1e-6:
            # Configuration fits, try to shrink outer side
            outer_side *= 0.95
        else:
            # Try to optimize positions to fit
            # Use scipy minimize to reduce penalty
            result = minimize(
                lambda p: objective(p, outer_side, outer_center, outer_angle),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            
            if result.success or result.fun < best_penalty:
                best_params = result.x
                best_penalty = result.fun
            
            # If we've been stuck, try a smaller outer side
            if penalty > 0.1:
                # Try to reduce outer side slightly
                outer_side *= 0.98
    
    # Final polish: binary search on outer side
    # Find the minimum outer side that works
    lo = 2.0
    hi = outer_side * 1.2
    
    # First, make sure hi works
    for _ in range(20):
        if time.time() > deadline:
            break
        mid = (lo + hi) / 2.0
        penalty = objective(best_params, mid, outer_center, outer_angle)
        if penalty < 1e-4:
            hi = mid
        else:
            lo = mid
    
    outer_side = hi
    
    # Build result
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([float(x), float(y), float(a)])
    
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
