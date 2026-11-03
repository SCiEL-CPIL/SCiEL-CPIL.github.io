import numpy as np
from scipy.optimize import minimize

N = 12
sqrt3 = np.sqrt(3)

def hex_support(point, angle_deg):
    """
    Return the support function value of a unit regular hexagon (side 1)
    centered at origin and rotated by angle_deg, evaluated at a unit direction 'point'.
    For a regular hexagon of side 1, the support in direction theta is:
        max over 6 vertices of dot(vertex_i, theta)
    Vertices of unit hexagon (side 1) are at radius 1 (circumradius = side for regular hex).
    Actually for a regular hexagon with side length s, circumradius = s.
    Vertices at angles 0, 60, 120, 180, 240, 300 degrees, radius = 1.
    So vertex_i = (cos(k*60°), sin(k*60°)) for k=0..5, then rotate by angle_deg.
    """
    k = np.arange(6)
    base_angles = k * np.pi / 3.0
    rot = np.radians(angle_deg)
    angles = base_angles + rot
    verts = np.column_stack([np.cos(angles), np.sin(angles)])
    return np.max(verts @ point)

def outer_hex_support(point, angle_deg):
    """
    Support function of outer regular hexagon (side 1, centered at origin) in direction 'point',
    with rotation angle_deg. Same as hex_support but we use it for the outer boundary.
    The outer hexagon has side length S; support scales linearly: S * hex_support(point, angle_deg).
    We'll handle scaling outside.
    """
    return hex_support(point, angle_deg)

def max_overlap_violation(inner, outer_center, outer_side, outer_angle):
    """
    For each pair of inner hexagons, compute the minimum distance between them
    using the Minkowski sum / support function approach.
    Two convex polygons P and Q are disjoint iff max_theta (h_P(theta) + h_Q(-theta)) < |p_q - q_q| * something...
    Actually, the standard way: two convex bodies A and B are disjoint iff for all unit vectors u,
        h_A(u) + h_B(-u) < distance between centers projected... no.
    
    Better: The minimum distance between two convex polygons can be found via:
        min over points... but we can use the fact that for regular hexagons,
        the Minkowski sum is also a regular hexagon (or a 12-gon if rotations differ).
    
    For two regular hexagons of side 1 with rotations a1, a2 and center difference d = c2 - c1,
    they are disjoint iff for all unit vectors u:
        h_1(u) + h_2(-u) <= |d|  ... no, that's not quite right either.
    
    Actually, the correct condition: Two convex sets A = a + P, B = b + Q are disjoint iff
        max_u (h_P(u) + h_Q(-u) - u·(b-a)) < 0
    i.e., the maximum over unit u of [h_P(u) + h_Q(-u) - u·d] < 0.
    
    Equivalently, they overlap iff max_u (h_P(u) + h_Q(-u) - u·d) >= 0.
    
    For our purposes, we need to check if they are disjoint. We'll compute:
        f(u) = h_P(u) + h_Q(-u) - u·d
    and check if max_u f(u) < 0.
    
    For regular hexagons, the maximum occurs at one of the 12 edge normals (6 from each).
    So we can evaluate at those 12 directions.
    """
    # We'll do a simpler check: compute the 12 relevant directions and check.
    # For hexagon 1 with rotation a1, its edge normals are at angles a1 + k*60°, k=0..5
    # For hexagon 2 with rotation a2, its edge normals are at angles a2 + k*60°, k=0..5
    # The Minkowski sum's edge normals are the union, so 12 directions total.
    
    pass  # We'll implement a simpler distance check for regular hexagons

def hex_hex_min_distance(c1, a1_deg, c2, a2_deg):
    """
    Compute the minimum distance between two unit regular hexagons centered at c1, c2
    with rotations a1_deg, a2_deg. If they overlap, return negative value.
    """
    d = np.array(c2) - np.array(c1)
    # For two convex polygons, min distance = max_u (h1(u) + h2(-u) - u·d)
    # If this max is < 0, they are disjoint (gap = -max).
    # The maximum is achieved at a vertex normal of the Minkowski sum.
    # For two regular hexagons, check 12 directions.
    
    dirs = []
    for k in range(6):
        ang1 = np.radians(a1_deg) + k * np.pi / 3.0
        dirs.append(np.array([np.cos(ang1), np.sin(ang1)]))
        ang2 = np.radians(a2_deg) + k * np.pi / 3.0
        dirs.append(np.array([np.cos(ang2), np.sin(ang2)]))
    
    # Also include -u for each
    all_dirs = []
    for u in dirs:
        all_dirs.append(u)
        all_dirs.append(-u)
    
    max_val = -np.inf
    for u in all_dirs:
        h1 = hex_support(u, a1_deg)
        h2 = hex_support(-u, a2_deg)
        val = h1 + h2 - np.dot(u, d)
        if val > max_val:
            max_val = val
    
    # If max_val < 0, they are disjoint with gap = -max_val
    # If max_val >= 0, they overlap
    return -max_val  # positive if disjoint, negative if overlap

def outer_constraint_violation(inner, outer_center, outer_side, outer_angle):
    """
    Check if all inner hexagons are inside the outer hexagon.
    For each inner hexagon, check if all its vertices are inside the outer hexagon.
    A point p is inside a regular hexagon (side S, center c, rotation a) iff
        h_hex(p - c, a) <= S  for all ... no.
    Actually, point p is inside iff for all edge normals u, (p-c)·u <= S * h_unit(u, a)
    where h_unit is the support of unit hexagon in direction u.
    Simpler: check all 6 vertices of inner hex against all 6 half-planes of outer hex.
    """
    # Outer hexagon: center oc, side S, rotation oa
    # Its 6 edges: each edge has a normal direction and offset S * h_unit(normal, oa)
    # For the outer hex, the 6 outward normals are at angles oa + k*60°, k=0..5
    # A point p is inside iff (p - oc)·n_k <= S * h_unit(n_k, oa) for all k
    
    oc = np.array(outer_center)
    
    # Get vertices of each inner hexagon
    for cx, cy, ang_deg in inner:
        cc = np.array([cx, cy])
        # Vertices of inner hexagon
        for k in range(6):
            theta = np.radians(ang_deg) + k * np.pi / 3.0
            vx = cx + np.cos(theta)
            vy = cy + np.sin(theta)
            p = np.array([vx, vy])
            
            # Check against all 6 half-planes of outer hex
            for m in range(6):
                n_ang = np.radians(outer_angle) + m * np.pi / 3.0
                n = np.array([np.cos(n_ang), np.sin(n_ang)])
                lhs = np.dot(p - oc, n)
                rhs = outer_side * hex_support(n, outer_angle)
                if lhs > rhs + 1e-10:
                    return lhs - rhs  # violation amount
    
    return 0.0

def total_violation(inner, outer_center, outer_side, outer_angle):
    """
    Compute total violation: sum of overlap violations between inner hexagons
    plus outer constraint violations.
    """
    viol = 0.0
    
    # Inner-inner overlaps
    for i in range(N):
        for j in range(i+1, N):
            c1 = [inner[i][0], inner[i][1]]
            a1 = inner[i][2]
            c2 = [inner[j][0], inner[j][1]]
            a2 = inner[j][2]
            dist = hex_hex_min_distance(c1, a1, c2, a2)
            if dist < 0:
                viol += (-dist) ** 2  # penalize overlap
    
    # Outer constraints
    outer_viol = outer_constraint_violation(inner, outer_center, outer_side, outer_angle)
    if outer_viol > 0:
        viol += outer_viol ** 2
    
    return viol

def objective(params):
    """
    Objective function for optimization.
    params: [c1x, c1y, a1, c2x, c2y, a2, ..., outer_cx, outer_cy, outer_side, outer_angle]
    We optimize over inner positions/rotations and outer parameters.
    To minimize outer_side while keeping all inner hexagons inside and non-overlapping.
    """
    # Unpack
    inner_params = params[:3*N]
    outer_params = params[3*N:]
    outer_cx, outer_cy, outer_side, outer_angle = outer_params
    
    inner = []
    for i in range(N):
        x = inner_params[3*i]
        y = inner_params[3*i+1]
        ang = inner_params[3*i+2]
        inner.append([x, y, ang])
    
    # Add penalty for large side (we want to minimize it)
    side_penalty = outer_side ** 2 * 0.01
    
    viol = total_violation(inner, (outer_cx, outer_cy), outer_side, outer_angle)
    
    return viol + side_penalty

def construct():
    # Initial guess: place hexagons in a honeycomb-like pattern
    # For 12 hexagons, a 3-4-3 or 2-4-4-2 arrangement might work.
    # Let's start with a hexagonal arrangement.
    
    np.random.seed(42)
    
    # Initial positions: spread them out
    params = np.zeros(3*N + 4)
    
    # Place inner hexagons in a rough grid
    positions = []
    # 3 rows: 4, 4, 4 or 3, 4, 3, etc.
    # Let's try a pattern: offsets in a hex grid
    for i in range(N):
        # Rough circular placement
        angle = 2 * np.pi * i / N
        r = 2.0
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        positions.append([x, y, 0.0])
    
    for i, (x, y, a) in enumerate(positions):
        params[3*i] = x
        params[3*i+1] = y
        params[3*i+2] = a
    
    # Outer hexagon: center at origin, side large enough, rotation 0
    params[3*N] = 0.0      # outer_cx
    params[3*N+1] = 0.0    # outer_cy
    params[3*N+2] = 5.0    # outer_side (will be optimized)
    params[3*N+3] = 0.0    # outer_angle
    
    # Use Nelder-Mead or Powell for optimization
    result = minimize(objective, params, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    # Extract solution
    inner = []
    for i in range(N):
        x = result.x[3*i]
        y = result.x[3*i+1]
        ang = result.x[3*i+2]
        inner.append([float(x), float(y), float(ang)])
    
    outer_cx = float(result.x[3*N])
    outer_cy = float(result.x[3*N+1])
    outer_side = float(result.x[3*N+2])
    outer_angle = float(result.x[3*N+3])
    
    # Ensure outer_side is at least a minimum value
    outer_side = max(outer_side, 2.0)
    
    return inner, (outer_cx, outer_cy), outer_side, outer_angle
