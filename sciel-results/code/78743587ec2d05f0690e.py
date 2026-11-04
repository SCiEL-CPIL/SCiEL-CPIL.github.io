import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 12
DEADLINE = 140.0  # Keep a safety margin before the 150s evaluator kill

def _hex_vertices(s, angle_deg, cx=0.0, cy=0.0):
    """Return the 6 vertices of a regular hexagon with side s, rotated by angle_deg, centered at (cx, cy)."""
    a = np.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees
    base_angles = np.arange(6) * np.pi / 3
    angles = base_angles + a
    r = s  # circumradius = side length for regular hexagon
    vx = cx + r * np.cos(angles)
    vy = cy + r * np.sin(angles)
    return np.column_stack([vx, vy])

def _hexagon_circumscribed_radius(s):
    return s

def _point_in_hexagon(pt, center, s, angle_deg):
    """Check if a point is inside a regular hexagon using the six half-planes."""
    cx, cy = center
    a = np.radians(angle_deg)
    # The hexagon is the intersection of 6 half-planes.
    # For a regular hexagon centered at origin with circumradius s,
    # the apothem (inradius) is s * cos(pi/6) = s * sqrt(3)/2.
    # The normal vectors point outward at angles 0, 60, 120, 180, 240, 300 + a.
    # Actually, for a flat-top hexagon, the normals are at 30, 90, 150, 210, 270, 330 + a.
    # Let's use the standard: vertices at 0,60,...,300.
    # The edges are perpendicular to directions 30, 90, 150, 210, 270, 330.
    # Normal vectors (outward) at angles 30+30k for k=0..5, plus rotation a.
    normals_angles = np.arange(6) * np.pi / 3 + np.pi / 6 + a
    normals = np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])
    # Distance from center to each edge = apothem = s * cos(pi/6) = s * sqrt(3)/2
    apothem = s * np.cos(np.pi / 6)
    # For point pt, the signed distance from center along each normal must be <= apothem
    d = pt - np.array(center)
    dists = d @ normals.T
    return np.all(dists <= apothem + 1e-10)

def _hexagon_area(s):
    return (3 * np.sqrt(3) / 2) * s * s

def _min_dist_between_hexagons(c1, s1, a1, c2, s2, a2):
    """
    Compute the minimum distance between two regular hexagons.
    Returns 0 if they intersect, positive distance if disjoint.
    We use the fact that the distance between two convex polygons is the minimum
    distance between any pair of edges/vertices. For efficiency in optimization,
    we can compute this via the Minkowski difference approach.
    
    Actually, for a smooth surrogate, let's compute the minimum distance
    between the two sets of vertices and edges. But for gradient-based optimization,
    we need a smooth function. Let's use the distance between the two convex hulls
    computed via the support function approach.
    
    For simplicity and smoothness, let's compute the minimum distance between
    all pairs of vertices and edges. This is not smooth at contact points,
    but we can use a soft-min.
    """
    v1 = _hex_vertices(s1, a1, c1[0], c1[1])
    v2 = _hex_vertices(s2, a2, c2[0], c2[1])
    
    # Compute minimum distance between two convex polygons
    # Use the separating axis theorem: if there's a separating axis, the distance
    # is the projection gap. Otherwise, they intersect and distance is 0.
    
    # Get all edge normals from both polygons
    def get_normals(vertices):
        normals = []
        n = len(vertices)
        for i in range(n):
            e = vertices[(i+1) % n] - vertices[i]
            # Perpendicular to edge
            normal = np.array([-e[1], e[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-12)
            normals.append(normal)
        return np.array(normals)
    
    n1 = get_normals(v1)
    n2 = get_normals(v2)
    all_normals = np.vstack([n1, n2])
    
    min_dist = np.inf
    for normal in all_normals:
        # Project both polygons onto this normal
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        # If intervals overlap, no separation along this axis
        if proj1.max() < proj2.min() or proj2.max() < proj1.min():
            # Separating axis found
            gap = max(proj2.min() - proj1.max(), proj1.min() - proj2.max())
            min_dist = min(min_dist, abs(gap))
    
    if np.isinf(min_dist):
        # Polygons intersect
        return 0.0
    return min_dist

def _compute_outer_side_length(inner_positions, outer_center, outer_angle):
    """
    Compute the minimum side length of a regular hexagon centered at outer_center
    with rotation outer_angle that contains all inner hexagons.
    
    We need to find the minimum R such that all vertices of all inner hexagons
    are inside the outer hexagon.
    
    For a regular hexagon with circumradius R, centered at (cx, cy), rotated by angle a,
    a point (px, py) is inside iff for all 6 normals (at angles 30+30k+a),
    the projection of (px-cx, py-cy) onto the normal is <= R * cos(pi/6) = R * sqrt(3)/2.
    
    So R >= max_over_points, max_over_normals (projection / cos(pi/6))
    """
    cx, cy = outer_center
    a = np.radians(outer_angle)
    normals_angles = np.arange(6) * np.pi / 3 + np.pi / 6 + a
    normals = np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])
    apothem_factor = np.cos(np.pi / 6)
    
    max_proj = 0.0
    for (px, py, ang) in inner_positions:
        # Get all vertices of this inner hexagon
        verts = _hex_vertices(1.0, ang, px, py)
        d = verts - np.array([cx, cy])
        projections = d @ normals.T  # shape (6, 6)
        # For each normal, the max projection must be <= R * apothem_factor
        max_proj_per_normal = np.max(projections, axis=0)
        max_proj = max(max_proj, np.max(max_proj_per_normal))
    
    R = max_proj / apothem_factor
    return R

def construct():
    t_start = time.time()
    
    # Initial configuration: place 12 hexagons in a hexagonal lattice pattern
    # within a larger hexagonal region.
    # Let's start with a reasonable initial guess.
    
    # Place them in a 3-4-3-2 pattern or similar.
    # Actually, let's use a simple initial layout: a hexagonal cluster.
    # For N=12, a good initial layout is a hexagon of hexagons.
    # Center at origin, with 1 in center, 6 around it, and 5 more.
    
    # Let's try a simple linear arrangement first as a baseline,
    # but we want to optimize.
    
    # Initial positions: place in a hexagonal lattice
    positions = []
    # Central hexagon
    positions.append((0.0, 0.0, 0.0))
    # 6 surrounding
    for k in range(6):
        angle = k * np.pi / 3
        x = 1.75 * np.cos(angle)  # 1.75 = 2 * (sqrt(3)/2) * 1 = sqrt(3) ~ 1.732, use 1.75 for slight gap
        y = 1.75 * np.sin(angle)
        positions.append((x, y, 0.0))
    # 5 more: place in the next ring
    for k in range(5):
        angle = k * 2 * np.pi / 5 + np.pi / 5  # Offset to avoid symmetry conflicts
        x = 3.3 * np.cos(angle)
        y = 3.3 * np.sin(angle)
        positions.append((x, y, 0.0))
    
    best_result = (positions, (0.0, 0.0), 10.0, 0.0)
    best_score = _compute_outer_side_length(positions, (0.0, 0.0), 0.0)
    
    # We'll optimize the positions and the outer hexagon parameters
    # Variables: 12 * 3 (x, y, angle) + 3 (outer cx, cy, angle) = 39 variables
    # We want to minimize the outer side length R.
    
    def objective(vars):
        if time.time() - t_start > DEADLINE:
            return best_score + 1e6
        
        # Extract positions
        pos_vars = vars[:36]
        pos = [(pos_vars[i*3], pos_vars[i*3+1], pos_vars[i*3+2]) for i in range(12)]
        outer_cx = vars[36]
        outer_cy = vars[37]
        outer_angle = vars[38]
        
        # Compute outer side length
        R = _compute_outer_side_length(pos, (outer_cx, outer_cy), outer_angle)
        
        # Add penalty for overlapping inner hexagons
        penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                d = _min_dist_between_hexagons(
                    (pos[i][0], pos[i][1]), 1.0, pos[i][2],
                    (pos[j][0], pos[j][1]), 1.0, pos[j][2]
                )
                if d < 0.01:
                    penalty += (0.01 - d) ** 2 * 100
        
        return R + penalty
    
    # Initial variables
    x0 = np.zeros(39)
    for i, (px, py, ang) in enumerate(positions):
        x0[i*3] = px
        x0[i*3+1] = py
        x0[i*3+2] = ang
    x0[36] = 0.0
    x0[37] = 0.0
    x0[38] = 0.0
    
    # Try optimization
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-8})
        if result.fun < best_score:
            pos_vars = result.x[:36]
            pos = [(pos_vars[i*3], pos_vars[i*3+1], pos_vars[i*3+2]) for i in range(12)]
            outer_cx = result.x[36]
            outer_cy = result.x[37]
            outer_angle = result.x[38]
            R = _compute_outer_side_length(pos, (outer_cx, outer_cy), outer_angle)
            if R < best_score:
                best_score = R
                best_result = (pos, (outer_cx, outer_cy), R, outer_angle)
    except Exception:
        pass
    
    # Try a few random restarts if time permits
    num_restarts = 0
    while time.time() - t_start < DEADLINE - 10:
        num_restarts += 1
        # Generate a random initial configuration
        random_pos = []
        for i in range(12):
            x = np.random.uniform(-5, 5)
            y = np.random.uniform(-5, 5)
            ang = np.random.uniform(0, 360)
            random_pos.append((x, y, ang))
        
        random_x0 = np.zeros(39)
        for i, (px, py, ang) in enumerate(random_pos):
            random_x0[i*3] = px
            random_x0[i*3+1] = py
            random_x0[i*3+2] = ang
        random_x0[36] = np.random.uniform(-2, 2)
        random_x0[37] = np.random.uniform(-2, 2)
        random_x0[38] = np.random.uniform(0, 360)
        
        try:
            result = minimize(objective, random_x0, method='L-BFGS-B', 
                              options={'maxiter': 100, 'ftol': 1e-7})
            if result.fun < best_score:
                pos_vars = result.x[:36]
                pos = [(pos_vars[i*3], pos_vars[i*3+1], pos_vars[i*3+2]) for i in range(12)]
                outer_cx = result.x[36]
                outer_cy = result.x[37]
                outer_angle = result.x[38]
                R = _compute_outer_side_length(pos, (outer_cx, outer_cy), outer_angle)
                if R < best_score:
                    best_score = R
                    best_result = (pos, (outer_cx, outer_cy), R, outer_angle)
        except Exception:
            pass
    
    # Format the best result
    inner, outer_center, side, angle = best_result
    # Convert to the required format
    inner_list = [[x, y, a] for (x, y, a) in inner]
    return inner_list, outer_center, side, angle

# Provide a simple fallback if optimization fails
def _fallback():
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
