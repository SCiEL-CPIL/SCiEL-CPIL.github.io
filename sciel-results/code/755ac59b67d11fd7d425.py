"""Place 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
DEADLINE = 115.0  # safety margin before 120s kill

# Vertices of a unit regular hexagon (side 1), centered at origin, flat-top orientation
# Angle 0 means flat side on top/bottom? Let's use pointy-top or flat-top consistently.
# A regular hexagon with side length s has vertices at angles 0, 60, 120, 180, 240, 300 degrees
# if pointy-top, or 30, 90, 150, 210, 270, 330 if flat-top.
# Let's use: vertex i at angle (60*i + offset) degrees, radius = side_length (for regular hex, circumradius = side)
# For a regular hexagon with side length 1, circumradius = 1.
# Vertices: (cos(theta_i), sin(theta_i)) for theta_i = pi/3 * i + offset

def hex_vertices(angle_deg):
    """Return 6 vertices of unit hexagon rotated by angle_deg."""
    a = np.radians(angle_deg)
    thetas = np.pi / 3 * np.arange(6) + a
    return np.column_stack([np.cos(thetas), np.sin(thetas)])

def outer_hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of outer regular hexagon."""
    a = np.radians(angle_deg)
    thetas = np.pi / 3 * np.arange(6) + a
    verts = side * np.column_stack([np.cos(thetas), np.sin(thetas)])
    return verts + np.array([cx, cy])

def point_in_polygon(px, py, verts):
    """Check if point (px,py) is inside polygon with vertices verts (array of shape (6,2))."""
    n = len(verts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def hexagon_in_hexagon(inner_cx, inner_cy, inner_angle, outer_cx, outer_cy, outer_side, outer_angle):
    """Check if all vertices of inner hex are inside outer hex."""
    inner_verts = hex_vertices(inner_angle) + np.array([inner_cx, inner_cy])
    outer_verts = outer_hex_vertices(outer_cx, outer_cy, outer_side, outer_angle)
    for vx, vy in inner_verts:
        if not point_in_polygon(vx, vy, outer_verts):
            return False
    return True

def min_outer_side_for_config(inner_positions, outer_cx, outer_cy, outer_angle):
    """Given inner hex positions (fixed), find minimum outer side that contains all.
    We can do this by computing the maximum 'distance' from outer center in each direction.
    Actually, for a fixed outer hex (center, rotation), the minimum side length such that
    all inner vertices are inside can be computed by finding, for each inner vertex, the 
    required side length, and taking the max.
    
    For a regular hexagon centered at origin with rotation angle a, a point p is inside 
    if for all 6 sides, the point is on the correct side.
    
    The side lines of a regular hexagon with circumradius R (side = R) at rotation a:
    The normal to side i points at angle (pi/3 * i + a + pi/6) ... let me think.
    
    Actually, let's use the support function approach. For a convex polygon (outer hex) 
    centered at origin with vertices at R*(cos(theta_i), sin(theta_i)), a point p is inside 
    iff for all edges, p is on the inner side.
    
    Alternatively: the minimum R such that all points are inside a hexagon of circumradius R 
    (centered at origin, rotation a) is the maximum over all points p of the "hexagon norm" of p.
    
    The hexagon norm: for a regular hexagon with flat sides (or pointy), the norm is defined by
    the 6 linear constraints.
    
    Let me just do a binary search on side length for a given configuration, or better,
    compute it directly.
    
    For a regular hexagon centered at origin with circumradius R and rotation angle a,
    the 6 supporting lines have normals at angles: a + pi/3 * i + pi/6 for i=0..5? 
    Let's derive: vertices at angles a + pi/3 * i. The edges connect vertex i to vertex i+1.
    The outward normal of edge i (between vertex i and i+1) bisects the angle, so it's at
    angle a + pi/3 * i + pi/6.
    
    The distance from origin to each edge is R * cos(pi/6) = R * sqrt(3)/2 (the inradius).
    
    So the constraint for point p to be inside is:
    For each i=0..5: p · n_i <= inradius = R * sqrt(3)/2
    where n_i = (cos(a + pi/3*i + pi/6), sin(a + pi/3*i + pi/6))
    
    So R >= (p · n_i) / (sqrt(3)/2) for all i where p · n_i > 0.
    
    The minimum R is max over all points p and all i of (p · n_i) / (sqrt(3)/2).
    """
    a = np.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + np.pi / 3 * i + np.pi / 6
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    normals = np.array(normals)  # shape (6, 2)
    
    # All inner vertices
    all_verts = []
    for (cx, cy, ang) in inner_positions:
        verts = hex_vertices(ang) + np.array([cx, cy])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)  # shape (6*N, 2)
    
    # Translate by outer center
    all_verts = all_verts - np.array([outer_cx, outer_cy])
    
    # Compute p · n_i for all points and all normals
    # all_verts @ normals.T -> shape (6N, 6)
    dots = all_verts @ normals.T  # shape (6N, 6)
    
    # R >= dots / (sqrt(3)/2) for positive dots
    inradius = np.sqrt(3) / 2
    max_val = np.max(dots)
    if max_val <= 0:
        return 0.0
    return max_val / inradius

def total_outer_side(inner_positions, outer_cx, outer_cy, outer_angle):
    """Compute the minimum outer side length for given configuration."""
    return min_outer_side_for_config(inner_positions, outer_cx, outer_cy, outer_angle)

def construct():
    start_time = time.time()
    
    # Initial configuration: hexagonal packing
    # Place 11 hexagons in a honeycomb-like pattern
    # A hexagonal arrangement of 11: center + 6 in first ring + 4 more? 
    # Actually, let's start with a simple grid-like arrangement and optimize.
    
    # Honeycomb: center at (0,0), then 6 around at distance sqrt(3) (for touching hexagons)
    # For unit hexagons (side 1), the distance between centers of touching hexagons is sqrt(3) * side = sqrt(3)
    # Wait, for two regular hexagons of side 1 to be tangent, the distance between centers is 2 * inradius = sqrt(3).
    
    # Let's create a hexagonal cluster: 1 in center, 6 in first ring, 4 in second ring (partial)
    # Or just use a compact arrangement.
    
    # Initial positions: hexagonal lattice
    # Center: (0,0)
    # First ring (6 hexagons) at distance sqrt(3) from center
    # Second ring: pick 4 from the 12 positions
    
    d = np.sqrt(3)  # distance between touching hex centers
    
    # Hexagonal lattice points
    # Using axial coordinates converted to Cartesian
    # For pointy-top hexagons, but our hexagons can be any orientation.
    # Let's use a simple arrangement first.
    
    initial_positions = []
    # Center
    initial_positions.append((0.0, 0.0, 0.0))
    
    # First ring: 6 hexagons
    for i in range(6):
        angle = np.pi / 3 * i
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        initial_positions.append((x, y, 0.0))
    
    # Second ring: 4 more hexagons (to make 11 total)
    # Place them at the "corners" of the hexagonal arrangement
    for i in range(4):
        angle = np.pi / 3 * i + np.pi / 6  # offset by 30 degrees
        x = 2 * d * np.cos(angle)
        y = 2 * d * np.sin(angle)
        initial_positions.append((x, y, 0.0))
    
    # That's 1 + 6 + 4 = 11
    
    best_side = float('inf')
    best_config = initial_positions
    best_outer = (0.0, 0.0, 0.0)
    
    # Phase 1: Optimize inner positions only (fix outer at origin, angle 0)
    # Parameters: 3 per hexagon (x, y, angle)
    n_params = 3 * N
    
    def unpack_inner(params):
        positions = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i + 1]
            ang = params[3*i + 2]
            positions.append((x, y, ang))
        return positions
    
    def objective_inner(params):
        positions = unpack_inner(params)
        return total_outer_side(positions, 0.0, 0.0, 0.0)
    
    # Initial parameters
    init_params = []
    for (x, y, ang) in initial_positions:
        init_params.extend([x, y, ang])
    init_params = np.array(init_params)
    
    # Optimize inner positions
    try:
        result = minimize(objective_inner, init_params, method='L-BFGS-B', 
                         options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
        if result.success or result.fun < best_side:
            best_side = result.fun
            best_config = unpack_inner(result.x)
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 30:
        # Return best so far
        return [(x, y, ang) for (x, y, ang) in best_config], best_outer, best_side, 0.0
    
    # Phase 2: Joint optimization (inner positions + outer center and angle)
    # Parameters: 3*N + 3 (outer cx, cy, angle)
    n_params_joint = 3 * N + 3
    
    def unpack_joint(params):
        positions = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i + 1]
            ang = params[3*i + 2]
            positions.append((x, y, ang))
        outer_cx = params[3*N]
        outer_cy = params[3*N + 1]
        outer_angle = params[3*N + 2]
        return positions, outer_cx, outer_cy, outer_angle
    
    def objective_joint(params):
        positions, ocx, ocy, oang = unpack_joint(params)
        return total_outer_side(positions, ocx, ocy, oang)
    
    # Initial joint params
    init_joint = list(init_params) + [0.0, 0.0, 0.0]
    init_joint = np.array(init_joint)
    
    try:
        result2 = minimize(objective_joint, init_joint, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12})
        if result2.fun < best_side:
            best_side = result2.fun
            best_config, best_outer[0], best_outer[1], best_outer[2] = unpack_joint(result2.x)
            best_outer = (best_outer[0], best_outer[1], best_outer[2])
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 20:
        return [(x, y, ang) for (x, y, ang) in best_config], best_outer, best_side, best_outer[2]
    
    # Phase 3: Nelder-Mead polish
    try:
        init_nm = list(result2.x) if 'result2' in dir() and result2.fun < best_side else list(init_joint)
        result3 = minimize(objective_joint, np.array(init_nm), method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
        if result3.fun < best_side:
            best_side = result3.fun
            best_config, ox, oy, oang = unpack_joint(result3.x)
            best_outer = (ox, oy, oang)
    except Exception:
        pass
    
    # Final validation
    final_positions = [(x, y, ang) for (x, y, ang) in best_config]
    final_outer = best_outer
    final_side = best_side
    
    return final_positions, final_outer, final_side, final_outer[2]

# Test
if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Outer side: {side}")
    print(f"Outer center: {center}")
    print(f"Outer angle: {angle}")
    for i, (x, y, a) in enumerate(inner):
        print(f"  Hex {i}: ({x:.4f}, {y:.4f}, {a:.4f})")
