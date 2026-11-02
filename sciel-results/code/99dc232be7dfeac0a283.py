"""Radial Slicing formulation for 12 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize

N = 12

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    angle_rad = np.deg2rad(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append((x, y))
    return np.array(vertices)

def _support_function_distance(point, vertices):
    """
    Compute the minimum distance from point to the convex polygon defined by vertices.
    Uses the fact that for a convex polygon, the distance from an external point can be
    computed by checking edges and vertices.
    For internal points, distance is 0.
    """
    # Check if point is inside the polygon
    # For a convex polygon, point is inside if it's on the same side of all edges
    n = len(vertices)
    inside = True
    for i in range(n):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n]
        # Cross product to check side
        dx = v2[0] - v1[0]
        dy = v2[1] - v1[1]
        px = point[0] - v1[0]
        py = point[1] - v1[1]
        cross = dx * py - dy * px
        if cross < -1e-10:
            inside = False
            break
    
    if inside:
        return 0.0
    
    # Compute minimum distance to any edge
    min_dist = float('inf')
    for i in range(n):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n]
        # Project point onto line segment v1-v2
        d = v2 - v1
        d_len_sq = np.dot(d, d)
        if d_len_sq < 1e-12:
            dist = np.linalg.norm(point - v1)
        else:
            t = np.dot(point - v1, d) / d_len_sq
            t = np.clip(t, 0.0, 1.0)
            closest = v1 + t * d
            dist = np.linalg.norm(point - closest)
        min_dist = min(min_dist, dist)
    
    return min_dist

def _compute_required_outer_side(inner_positions, inner_angles, outer_center, outer_angle):
    """
    Given inner hexagon positions/angles and outer hexagon center/angle,
    compute the minimum outer side length such that all inner hexagons fit inside.
    
    The inner hexagons are unit side length (side=1).
    We need to find the minimum S such that each inner hexagon is contained in the outer hexagon.
    
    For a regular hexagon with center C, side S, and rotation angle A, a point P is inside if:
    For each of the 6 half-planes defined by the edges, P satisfies the inequality.
    
    The support function approach: the outer hexagon must contain all vertices of all inner hexagons.
    The minimum S is determined by the maximum over all inner hexagon vertices of the 
    "radial extent" in the direction of each of the 6 normal directions of the outer hexagon.
    
    For a regular hexagon with side S, the distance from center to each edge (apothem) is S * sqrt(3)/2.
    The distance from center to each vertex is S.
    
    A point P is inside the outer hexagon (center C, side S, angle A) if and only if:
    For each of the 6 directions (normals to edges), the projection of (P - C) onto that normal
    is at most S * sqrt(3)/2.
    
    The 6 normal directions are at angles A + i*pi/3 for i=0..5, but actually the normals to the
    edges of a regular hexagon are at angles A + i*pi/3 where A is the rotation angle.
    
    Wait, let me be more careful. A regular hexagon with "pointy top" vs "flat top" matters.
    Let's define: if angle_deg=0, the hexagon has vertices at angles 0, 60, 120, 180, 240, 300 degrees.
    The edges are between consecutive vertices. The normal to the edge between vertex at angle 0 and 
    vertex at angle 60 is at angle 30 degrees.
    
    So the 6 outward normals are at angles: A + 30 + i*60 degrees for i=0..5.
    
    The apothem (distance from center to each edge) is S * sqrt(3)/2.
    
    So for point P to be inside: for each normal direction n_i, (P - C) · n_i <= S * sqrt(3)/2.
    
    Therefore, the minimum S required is:
    S_min = max over all inner vertices P, and all 6 normal directions n_i of:
           ((P - C) · n_i) / (sqrt(3)/2)
    
    But we also need ((P - C) · n_i) to be positive; if it's negative, that constraint is not binding.
    Actually, we need all 6 constraints, so we take the max of ((P-C)·n_i) for each i, then divide by sqrt(3)/2.
    """
    outer_c = np.array(outer_center)
    outer_angle_rad = np.deg2rad(outer_angle)
    
    # The 6 outward normal directions
    normals = []
    for i in range(6):
        theta = outer_angle_rad + np.pi / 6 + i * np.pi / 3
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    
    max_projection = 0.0
    
    for (cx, cy, angle) in inner_positions:
        verts = _hex_vertices(cx, cy, 1.0, angle)
        for v in verts:
            for n in normals:
                proj = np.dot(v - outer_c, n)
                if proj > max_projection:
                    max_projection = proj
    
    # S * sqrt(3)/2 >= max_projection
    # S >= max_projection / (sqrt(3)/2)
    if max_projection <= 0:
        return 0.0
    return max_projection / (np.sqrt(3) / 2)

def _objective(params):
    """
    Objective function: minimize the outer side length.
    
    Parameters:
    - params[0]: ring1_radius (distance from center to the 5 inner ring hexagons)
    - params[1]: ring2_radius (distance from center to the 6 outer ring hexagons)
    - params[2]: global_rotation (rotation of the entire arrangement, in degrees)
    
    The arrangement:
    - 1 hexagon at center (0, 0)
    - 5 hexagons at radius ring1_radius, at angles 0, 72, 144, 216, 288 + global_rotation
    - 6 hexagons at radius ring2_radius, at angles 0, 60, 120, 180, 240, 300 + global_rotation
    
    Each inner hexagon is oriented to face outward (or we can optimize their individual rotations too,
    but to keep the search space small, we'll fix their rotation relative to the radial direction).
    
    Actually, let's allow each ring to have a rotation offset.
    Let's use:
    - params[0]: ring1_radius
    - params[1]: ring2_radius
    - params[2]: global_rotation
    - params[3]: ring1_hex_rotation (rotation of hexagons in ring 1 relative to radial direction)
    - params[4]: ring2_hex_rotation (rotation of hexagons in ring 2 relative to radial direction)
    
    But the directive says 3 continuous parameters. Let's stick with 3 for now and see.
    """
    ring1_r = params[0]
    ring2_r = params[1]
    global_rot = params[2]
    
    # Build inner positions
    inner = []
    
    # Center hexagon
    inner.append((0.0, 0.0, 0.0))
    
    # Ring 1: 5 hexagons
    for i in range(5):
        theta = np.deg2rad(global_rot + i * 72.0)
        x = ring1_r * np.cos(theta)
        y = ring1_r * np.sin(theta)
        # Orient each hexagon to face outward (rotation = theta in degrees)
        angle_deg = np.rad2deg(theta)
        inner.append((x, y, angle_deg))
    
    # Ring 2: 6 hexagons
    for i in range(6):
        theta = np.deg2rad(global_rot + i * 60.0)
        x = ring2_r * np.cos(theta)
        y = ring2_r * np.sin(theta)
        angle_deg = np.rad2deg(theta)
        inner.append((x, y, angle_deg))
    
    # Outer hexagon centered at origin with rotation 0 (we can optimize this too, but let's fix it)
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    S = _compute_required_outer_side(inner, [0.0]*12, outer_center, outer_angle)
    
    # Add a penalty if hexagons overlap (they should be disjoint)
    # Check pairwise distances between centers; for unit hexagons, minimum center distance
    # depends on orientation. A rough lower bound is 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732 for parallel edges.
    # For safety, let's use a penalty if distance < 1.7.
    penalty = 0.0
    centers = [(p[0], p[1]) for p in inner]
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1.7:
                penalty += (1.7 - dist) ** 2
    
    return S + penalty

def construct():
    """
    Construct 12 unit hexagons inside a regular hexagon, minimizing the outer side length.
    Returns: (inner, center, side, angle)
    """
    # Initial guess
    # For 12 hexagons in a hexagonal packing, the optimal outer side is around 5-6.
    # Ring 1 radius ~ 1.7, Ring 2 radius ~ 3.4
    x0 = np.array([1.73, 3.46, 0.0])
    
    bounds = [(0.5, 10.0), (1.0, 15.0), (-180.0, 180.0)]
    
    result = minimize(_objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    ring1_r = result.x[0]
    ring2_r = result.x[1]
    global_rot = result.x[2]
    
    # Build final inner positions
    inner = []
    
    # Center hexagon
    inner.append([0.0, 0.0, 0.0])
    
    # Ring 1: 5 hexagons
    for i in range(5):
        theta = np.deg2rad(global_rot + i * 72.0)
        x = ring1_r * np.cos(theta)
        y = ring1_r * np.sin(theta)
        angle_deg = np.rad2deg(theta)
        inner.append([x, y, angle_deg])
    
    # Ring 2: 6 hexagons
    for i in range(6):
        theta = np.deg2rad(global_rot + i * 60.0)
        x = ring2_r * np.cos(theta)
        y = ring2_r * np.sin(theta)
        angle_deg = np.rad2deg(theta)
        inner.append([x, y, angle_deg])
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    S = _compute_required_outer_side(inner, [0.0]*12, outer_center, outer_angle)
    
    return inner, outer_center, S, outer_angle

# EVOLVE-BLOCK-START
# (The construct function above is the entry point)
# EVOLVE-BLOCK-END
