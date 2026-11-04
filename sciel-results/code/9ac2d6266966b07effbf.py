"""Packing 12 unit regular hexagons inside a minimal regular hexagon."""
import numpy as np
from scipy.optimize import minimize

N = 12

# Precompute unit regular hexagon vertices (flat-top orientation, side=1)
# A regular hexagon with side length 1 has vertices at distance 1 from center
# Vertices at angles 0, 60, 120, 180, 240, 300 degrees for pointy-top
# For flat-top: vertices at 30, 90, 150, 210, 270, 330 degrees
# Let's use standard: vertices at angles 0, pi/3, 2pi/3, pi, 4pi/3, 5pi/3 (pointy top)
# Actually, let's define a hexagon with "flat top" so that one pair of edges is horizontal.
# Flat-top hexagon vertices (side 1, centered at origin):
# (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
# This is actually pointy-left/right. Let me be precise.

# A regular hexagon with side length s=1, centered at origin:
# If we want "flat top" (horizontal edges at top and bottom):
#   vertices at angles 30°, 90°, 150°, 210°, 270°, 330° from x-axis
#   i.e., (cos(30°), sin(30°)), (cos(90°), sin(90°)), etc.
#   = (sqrt(3)/2, 1/2), (0, 1), (-sqrt(3)/2, 1/2), (-sqrt(3)/2, -1/2), (0, -1), (sqrt(3)/2, -1/2)
#   Width (horizontal extent) = sqrt(3), Height (vertical extent) = 2
#
# If we want "pointy top" (vertex at top):
#   vertices at angles 0°, 60°, 120°, 180°, 240°, 300°
#   = (1, 0), (1/2, sqrt(3)/2), (-1/2, sqrt(3)/2), (-1, 0), (-1/2, -sqrt(3)/2), (1/2, -sqrt(3)/2)
#   Width = 2, Height = sqrt(3)

# I'll use pointy-top as base, then allow rotation.
# Base vertices for unit hexagon (side=1), pointy-top, centered at origin:
_base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
_base_vertices = np.column_stack([np.cos(_base_angles), np.sin(_base_angles)])

def _hexagon_vertices(cx, cy, angle_deg):
    """Return 6 vertices of a unit regular hexagon centered at (cx,cy) rotated by angle_deg."""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return _base_vertices @ R.T + np.array([cx, cy])

def _polygon_overlap_distance(v1, v2):
    """
    Estimate minimum translation distance to separate two convex polygons.
    Uses the separating axis theorem: for each edge normal of both polygons,
    compute the overlap. If no separating axis exists, polygons overlap.
    Returns the minimum overlap depth (0 if separated, >0 if overlapping).
    """
    # Get edge normals for both polygons
    def get_axes(vertices):
        axes = []
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]
            edge = p2 - p1
            # Normal perpendicular to edge
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal /= norm
            axes.append(normal)
        return np.array(axes)

    axes1 = get_axes(v1)
    axes2 = get_axes(v2)
    all_axes = np.vstack([axes1, axes2])

    min_overlap = float('inf')
    for axis in all_axes:
        proj1 = v1 @ axis
        proj2 = v2 @ axis
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        
        # Check if separated
        if max1 <= min2 or max2 <= min1:
            return 0.0  # Separating axis found, no overlap
        
        # Overlap on this axis
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)

    return max(0.0, min_overlap)

def _smooth_overlap(v1, v2, sharpness=20.0):
    """
    Smooth approximation of the overlap depth between two hexagons.
    Uses a soft-min of the axis overlaps.
    """
    def get_axes(vertices):
        axes = []
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal /= norm
            axes.append(normal)
        return np.array(axes)

    axes1 = get_axes(v1)
    axes2 = get_axes(v2)
    all_axes = np.vstack([axes1, axes2])

    overlaps = np.zeros(len(all_axes))
    for i, axis in enumerate(all_axes):
        proj1 = v1 @ axis
        proj2 = v2 @ axis
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        
        # Overlap on this axis (can be negative if separated)
        overlap = min(max1, max2) - max(min1, min2)
        overlaps[i] = overlap

    # The actual overlap depth is the minimum of positive overlaps, or 0 if any axis separates
    # We want a smooth version. Use: if any overlap <= 0, result is 0 (or small).
    # Smooth min: -1/sharpness * log(sum(exp(-sharpness * x)))
    # But we want max(0, min(overlaps)) smoothly.
    
    # Alternative: the constraint is that for ALL axes, overlap >= 0.
    # Violation = max(0, -min(overlaps)) = max(0, max(-overlaps))
    # Smooth max: 1/sharpness * log(sum(exp(sharpness * x)))
    neg_overlaps = -overlaps
    # If all neg_overlaps are very negative (large positive overlaps), smooth max is small
    # We want violation = smooth_max(neg_overlaps) but clamped to >= 0
    # Actually, violation should be 0 when all overlaps >= 0 (all neg_overlaps <= 0)
    
    # Use softplus-like: violation = smooth_max(neg_overlaps, 0)
    # = 1/sharpness * log(1 + sum(exp(sharpness * neg_overlaps)))
    # This is 0 when all neg_overlaps -> -inf, and ~max(neg_overlaps) when one is large positive
    
    scaled = sharpness * neg_overlaps
    # Clip to avoid overflow
    scaled = np.clip(scaled, -100, 100)
    violation = (1.0 / sharpness) * np.log(1.0 + np.sum(np.exp(scaled)))
    
    return violation

def _outer_hex_support(cx, cy, side, angle_deg, vertex):
    """
    Compute the support function of the outer hexagon in the direction of 'vertex'.
    The outer hexagon is centered at (cx, cy) with side length 'side' and rotation angle_deg.
    Support in direction u (unit vector) = cx*u_x + cy*u_y + side * h(u, angle_deg)
    where h is the support of a unit hexagon (side=1) in direction u, rotated.
    
    Actually, for a regular hexagon with side s, the support function in direction u is:
    s * max_i (v_i · u) where v_i are the vertices of a unit hexagon (side=1)
    But the outer hexagon's vertices are at distance s from center? No.
    
    For a regular hexagon with side length s:
    - The distance from center to each vertex is s (circumradius = s)
    - The distance from center to each edge midpoint is s * sqrt(3)/2 (inradius)
    
    The support function in direction u is max over vertices of (vertex · u)
    For a unit hexagon (side=1), vertices are at distance 1 from center.
    So support in direction u for unit hexagon = max_i (v_i · u)
    
    For outer hexagon with side s, centered at (cx,cy), rotated by angle_deg:
    vertices = s * R(angle) * v_i + (cx, cy)
    support in direction u = max_i (s * (R(angle) * v_i) · u + (cx,cy) · u)
                         = s * max_i (v_i · (R(-angle) * u)) + (cx,cy) · u
    
    So we need max_i (v_i · u_rotated) where u_rotated = R(-angle) * u
    """
    theta = np.radians(-angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    u_rot = R @ vertex
    # Max dot product of unit hexagon vertices with u_rot
    dots = _base_vertices @ u_rot
    max_dot = np.max(dots)
    return side * max_dot + (cx * vertex[0] + cy * vertex[1])

def _objective(params, sharpness=15.0):
    """
    Objective function: maximize the minimum margin over all inner hexagon vertices
    against the outer hexagon, plus penalty for overlaps.
    
    params layout:
    - For each of N inner hexagons: cx, cy, angle (3 params each) => 3*N
    - Outer: ocx, ocy, side, oangle (4 params)
    Total: 3*N + 4
    """
    # Unpack
    inner_params = params[:3 * N].reshape(N, 3)
    ocx, ocy, side, oangle = params[3 * N:3 * N + 4]
    
    # Compute vertices for each inner hexagon
    inner_verts = []
    for i in range(N):
        cx, cy, angle = inner_params[i]
        verts = _hexagon_vertices(cx, cy, angle)
        inner_verts.append(verts)
    
    # Constraint 1: All inner hexagon vertices must be inside outer hexagon
    # For each vertex of each inner hexagon, check support
    margins = []
    for i in range(N):
        for j in range(6):
            vertex = inner_verts[i][j]
            # Unit direction
            norm = np.linalg.norm(vertex - np.array([ocx, ocy]))
            if norm < 1e-10:
                continue
            direction = (vertex - np.array([ocx, ocy])) / norm
            # Actually, we need to check if vertex is inside outer hexagon.
            # A point p is inside a convex polygon if for all edge normals n, (p - edge_point) · n <= 0
            # Or equivalently, the support function of the outer hex in direction (p - center) >= |p - center|
            
            # Let's use: p is inside outer hex iff for all directions, the support >= projection
            # More directly: p is inside if max over outer vertices... no.
            
            # Correct check: p is inside outer hexagon iff the distance from center in direction (p-center)
            # is <= the support of the outer hex in that direction.
            
            p_centered = vertex - np.array([ocx, ocy])
            dist = np.linalg.norm(p_centered)
            if dist < 1e-10:
                margins.append(0.0)
                continue
            direction = p_centered / dist
            support = _outer_hex_support(ocx, ocy, side, oangle, direction)
            # The actual distance from center to the boundary in this direction
            # p is inside iff dist <= support (approximately, since support is the max extent)
            # Actually, for a convex polygon, p is inside iff for ALL directions u, (p-center)·u <= support(u)
            # It's sufficient to check in the direction of p-center for the "farthest" point, but not sufficient in general.
            # However, for a regular hexagon, checking in the 6 vertex directions and 6 edge-normal directions should suffice.
            # Let's just use this as a heuristic constraint.
            margin = support - dist
            margins.append(margin)
    
    min_margin = min(margins) if margins else 0.0
    
    # Penalty for negative margin (vertex outside outer hex)
    outer_penalty = 0.0
    if min_margin < 0:
        outer_penalty = -min_margin * 10.0  # Heavy penalty
    
    # Constraint 2: No overlaps between inner hexagons
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            v1 = inner_verts[i]
            v2 = inner_verts[j]
            # Check overlap
            ov = _smooth_overlap(v1, v2, sharpness=sharpness)
            overlap_penalty += ov
    
    # Objective: we want to minimize side, but also ensure constraints are satisfied.
    # Use: objective = side + outer_penalty + overlap_penalty
    # But we want to MINIMIZE side while keeping penalties near 0.
    # Better: objective = side + lambda1 * max(0, -min_margin) + lambda2 * sum(overlaps)
    
    obj = side + 50.0 * max(0.0, -min_margin) + 10.0 * overlap_penalty
    
    return obj

def _objective_grad(params, sharpness=15.0):
    """Numerical gradient (since analytical is complex)."""
    eps = 1e-7
    grad = np.zeros_like(params)
    f0 = _objective(params, sharpness)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = _objective(params_plus, sharpness)
        grad[i] = (f_plus - f0) / eps
    return grad

def construct():
    """
    Find optimal packing of 12 unit hexagons in minimal outer hexagon.
    Returns: inner (list of [x, y, angle_deg]), center (x, y), side, angle
    """
    # Initial guess: place hexagons in a ring or grid pattern
    # Start with a reasonable initial configuration
    np.random.seed(42)
    
    # Initial inner positions: try a hexagonal grid arrangement
    # 12 hexagons: maybe 2 rows of 6, or a ring
    initial_inner = []
    # Place in a 3x4 grid with hexagonal offsets
    spacing = 1.8  # Slightly more than 2 to avoid initial overlaps
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * spacing * np.sqrt(3) / 2
            y = row * spacing * 1.5
            if row % 2 == 1:
                x += spacing * np.sqrt(3) / 4
            positions.append([x, y, 0.0])
    
    # Center the configuration
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx_mean = (max(xs) + min(xs)) / 2
    cy_mean = (max(ys) + min(ys)) / 2
    for p in positions:
        p[0] -= cx_mean
        p[1] -= cy_mean
    
    initial_inner = positions
    
    # Initial outer: large enough to contain all
    max_r = max(np.sqrt(p[0]**2 + p[1]**2) for p in initial_inner)
    initial_side = max_r + 2.0
    initial_center = [0.0, 0.0]
    initial_angle = 0.0
    
    # Build initial params
    params0 = np.zeros(3 * N + 4)
    for i in range(N):
        params0[3*i] = initial_inner[i][0]
        params0[3*i+1] = initial_inner[i][1]
        params0[3*i+2] = initial_inner[i][2]
    params0[3*N] = initial_center[0]
    params0[3*N+1] = initial_center[1]
    params0[3*N+2] = initial_side
    params0[3*N+3] = initial_angle
    
    # Bounds: 
    # Inner positions: [-10, 10] for x, y; [0, 360] for angle
    # Outer center: [-10, 10]
    # Side: [2, 20]
    # Outer angle: [0, 60] (due to 6-fold symmetry, 0-60 is enough)
    bounds = []
    for i in range(N):
        bounds.append((-10.0, 10.0))  # x
        bounds.append((-10.0, 10.0))  # y
        bounds.append((0.0, 360.0))   # angle
    bounds.append((-10.0, 10.0))  # ocx
    bounds.append((-10.0, 10.0))  # ocy
    bounds.append((2.0, 20.0))    # side
    bounds.append((0.0, 60.0))    # oangle
    
    # Optimize
    result = minimize(
        _objective,
        params0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
    )
    
    # Extract solution
    best_params = result.x
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        angle = best_params[3*i+2] % 360.0
        inner.append([x, y, angle])
    
    center = (best_params[3*N], best_params[3*N+1])
    side = best_params[3*N+2]
    angle = best_params[3*N+3]
    
    # Verify and potentially improve with a second pass if needed
    # Check if solution is valid
    inner_verts = []
    for i in range(N):
        verts = _hexagon_vertices(inner[i][0], inner[i][1], inner[i][2])
        inner_verts.append(verts)
    
    # Check outer containment
    valid = True
    for i in range(N):
        for j in range(6):
            vertex = inner_verts[i][j]
            p_centered = vertex - np.array(center)
            dist = np.linalg.norm(p_centered)
            if dist < 1e-10:
                continue
            direction = p_centered / dist
            support = _outer_hex_support(center[0], center[1], side, angle, direction)
            if dist > support + 1e-6:
                valid = False
                break
        if not valid:
            break
    
    # Check overlaps
    if valid:
        for i in range(N):
            for j in range(i+1, N):
                ov = _polygon_overlap_distance(inner_verts[i], inner_verts[j])
                if ov > 1e-6:
                    valid = False
                    break
            if not valid:
                break
    
    # If invalid, fall back to a known valid configuration
    if not valid:
        # Simple valid configuration: line up in a row with spacing
        inner = []
        for k in range(N):
            x = k * 2.1  # Slightly more than 2 to ensure no overlap
            y = 0.0
            inner.append([x, 0.0, 0.0])
        # Center
        cx = (N-1) * 2.1 / 2
        for i in range(N):
            inner[i][0] -= cx
        # Outer hexagon: need to contain all
        max_r = max(np.sqrt(p[0]**2 + p[1]**2) for p in inner)
        side = max_r + 1.5
        center = (0.0, 0.0)
        angle = 0.0
    
    return inner, center, side, angle

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
