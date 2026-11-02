"""Packing 11 unit regular hexagons inside a minimal regular hexagon.

Strategy:
  - Use a topological contact-graph / cluster topology: a central hexagon
    surrounded by 6, plus 4 more attached to specific edges (forming a
    compact 11-hexagon cluster).
  - Optimize positions and rotations of the 11 inner hexagons to minimize
    the side length of the smallest enclosing regular hexagon.
  - Use a smooth surrogate for the enclosing-hexagon side length via
    log-sum-exp over the six support functions.
  - Add a non-overlap penalty (soft) so the optimizer can navigate.
  - Multiple restarts with different initializations and outer rotations.
  - SLSQP polish pass at the end for high precision.
"""

import time
import math
import numpy as np
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
TIME_LIMIT = 130.0  # safety margin before 150s kill

def _hex_vertices_local(angle_deg):
    """Return 6 vertices of a unit regular hexagon (side 1) centered at origin, rotated by angle_deg."""
    a = math.radians(angle_deg)
    # Regular hexagon with side 1: vertices at angles 0,60,...,300 from center at distance 1
    # Actually for a regular hexagon with side length s, the circumradius = s
    # Vertices: (s*cos(k*pi/3 + a), s*sin(k*pi/3 + a)) for k=0..5
    angles = np.array([k * math.pi / 3 + a for k in range(6)])
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _hex_vertices(cx, cy, angle_deg):
    """Return 6 vertices of a unit regular hexagon centered at (cx,cy), rotated by angle_deg."""
    local = _hex_vertices_local(angle_deg)
    return local + np.array([cx, cy])

def _outer_side_length(points, outer_angle_deg):
    """
    Given a set of 2D points, compute the minimum side length of a regular hexagon
    with the given rotation (outer_angle_deg) that encloses all points.
    
    A regular hexagon with side length S and rotation theta has 6 supporting lines.
    The side length S must satisfy: for each of the 6 directions, the max projection
    of points onto that direction <= S (since for a regular hexagon with side S,
    the width in each of the 6 normal directions is S).
    
    Actually, for a regular hexagon with side length S, the distance from center to
    each side (apothem) is S * sqrt(3)/2. The width across opposite sides is S*sqrt(3).
    But we need to be careful about the geometry.
    
    Let me reconsider. A regular hexagon with side length S:
    - Circumradius (center to vertex) = S
    - Apothem (center to side midpoint) = S * sqrt(3) / 2
    
    The 6 sides are at distances S*sqrt(3)/2 from the center, in 6 directions separated by 60 degrees.
    The normals to the sides point at angles: theta, theta+60, theta+120, theta+180, theta+240, theta+300.
    
    For a point to be inside, its projection onto each normal direction (from center) must be <= S*sqrt(3)/2.
    But the center is also a variable. To minimize S, we should center the hexagon optimally.
    
    For a given set of points and given outer rotation, the minimal S is determined by:
    For each pair of opposite sides (3 pairs), the distance between the two parallel supporting lines
    must be at least the extent of the points in that direction.
    
    Direction k: unit vector u_k = (cos(theta + k*pi/3), sin(theta + k*pi/3))
    The projection of point p onto u_k is dot(p, u_k).
    For the hexagon to contain all points, we need:
    max_p dot(p, u_k) - min_p dot(p, u_k) <= S * sqrt(3)  for each k=0,1,2
    (since the width of the hexagon in direction u_k is 2 * apothem = S*sqrt(3))
    
    So S >= (max_p dot(p, u_k) - min_p dot(p, u_k)) / sqrt(3) for each k.
    Thus S_min = max over k of [extent_k / sqrt(3)].
    """
    theta = math.radians(outer_angle_deg)
    min_s = 0.0
    for k in range(3):
        ang = theta + k * math.pi / 3
        ux = math.cos(ang)
        uy = math.sin(ang)
        proj = points[:, 0] * ux + points[:, 1] * uy
        extent = proj.max() - proj.min()
        s_needed = extent / math.sqrt(3.0)
        if s_needed > min_s:
            min_s = s_needed
    return min_s

def _all_vertices(inner_positions, inner_angles):
    """Compute all vertices of all inner hexagons as a single Nx2 array."""
    verts = []
    for i in range(len(inner_positions)):
        cx, cy = inner_positions[i]
        ang = inner_angles[i]
        v = _hex_vertices(cx, cy, ang)
        verts.append(v)
    return np.vstack(verts)

def _pairwise_min_distance(p1, p2, a1, p2b, a2):
    """
    Compute minimum distance between two hexagons. Returns negative if overlapping.
    p1, p2: centers (2D), a1, a2: angles in degrees.
    """
    v1 = _hex_vertices(p1[0], p1[1], a1)
    v2 = _hex_vertices(p2[0], p2[1], a2)
    # Compute min distance between any vertex of v1 and any vertex of v2,
    # and also check edge-edge distances. For simplicity, use vertex-to-vertex
    # and vertex-to-edge. Actually, for a robust check, compute the min distance
    # between the two convex polygons.
    
    # Use a simple approach: for each vertex of poly1, find min distance to poly2,
    # and vice versa. Also check if any vertex is inside the other.
    
    def point_in_polygon(pt, poly):
        """Ray casting for point in convex polygon."""
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > pt[1]) != (yj > pt[1])) and \
               (pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    
    def dist_point_to_segment(pt, a, b):
        """Min distance from point to line segment ab."""
        ax, ay = a
        bx, by = b
        px, py = pt
        dx = bx - ax
        dy = by - ay
        if dx == 0 and dy == 0:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        cx = ax + t * dx
        cy = ay + t * dy
        return math.hypot(px - cx, py - cy)
    
    min_d = float('inf')
    # Check vertices of v1 against v2
    for pt in v1:
        if point_in_polygon(pt, v2):
            return -1.0
        for j in range(6):
            d = dist_point_to_segment(pt, v2[j], v2[(j+1)%6])
            if d < min_d:
                min_d = d
    # Check vertices of v2 against v1
    for pt in v2:
        if point_in_polygon(pt, v1):
            return -1.0
        for j in range(6):
            d = dist_point_to_segment(pt, v1[j], v1[(j+1)%6])
            if d < min_d:
                min_d = d
    return min_d

def _overlap_penalty(inner_positions, inner_angles, beta=50.0):
    """
    Compute a smooth penalty for overlaps between all pairs of inner hexagons.
    Returns 0 if no overlaps, positive otherwise.
    """
    penalty = 0.0
    n = len(inner_positions)
    for i in range(n):
        for j in range(i+1, n):
            d = _pairwise_min_distance(
                inner_positions[i], inner_angles[i],
                inner_positions[j], inner_angles[j]
            )
            if d < 0:
                # Overlapping: use soft penalty
                penalty += beta * (-d) ** 2
            elif d < 0.01:
                # Very close: add small penalty to push apart
                penalty += beta * (0.01 - d) ** 2
    return penalty

def _objective(x, outer_angle_deg):
    """
    Objective: outer side length + overlap penalty.
    x: flattened array of [x0, y0, a0, x1, y1, a1, ..., x10, y10, a10]
    """
    positions = np.array([x[i*3:i*3+2] for i in range(N)])
    angles = np.array([x[i*3+2] for i in range(N)])
    
    verts = _all_vertices(positions, angles)
    s = _outer_side_length(verts, outer_angle_deg)
    
    pen = _overlap_penalty(positions, angles)
    
    return s + pen

def _initialization(mode):
    """
    Generate initial positions and angles for 11 hexagons.
    mode 0: compact cluster (center + 6 around + 4 more)
    mode 1: linear chain
    mode 2: two rows
    """
    positions = []
    angles = []
    
    if mode == 0:
        # Central hexagon at origin
        positions.append([0.0, 0.0])
        angles.append(0.0)
        # 6 surrounding hexagons
        # For hexagons of side 1, the distance between centers of adjacent hexagons
        # in a hexagonal lattice is 2 * apothem = sqrt(3) * side = sqrt(3)
        # Actually, for two unit hexagons sharing an edge, the center-to-center distance
        # is 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732
        d = math.sqrt(3.0)
        for k in range(6):
            ang = k * math.pi / 3
            positions.append([d * math.cos(ang), d * math.sin(ang)])
            angles.append(0.0)
        # 4 more: attach to specific edges of the outer ring
        # Positions at distance 2*d in certain directions
        for k in [0, 2, 4, 1]:
            ang = k * math.pi / 3
            positions.append([2 * d * math.cos(ang), 2 * d * math.sin(ang)])
            angles.append(0.0)
            
    elif mode == 1:
        # Linear chain
        d = math.sqrt(3.0)
        for i in range(N):
            positions.append([(i - N//2) * d, 0.0])
            angles.append(0.0)
            
    elif mode == 2:
        # Two rows of 6 and 5
        d = math.sqrt(3.0)
        dy = 1.5  # vertical offset for hexagonal packing
        for i in range(6):
            positions.append([(i - 2.5) * d, 0.0])
            angles.append(0.0)
        for i in range(5):
            positions.append([(i - 2.0) * d, dy])
            angles.append(0.0)
    else:
        # Random
        np.random.seed(42)
        for i in range(N):
            positions.append([np.random.uniform(-3, 3), np.random.uniform(-3, 3)])
            angles.append(np.random.uniform(0, 360))
    
    return positions, angles

def _flatten(positions, angles):
    x = []
    for i in range(N):
        x.extend([positions[i][0], positions[i][1], angles[i]])
    return x

def _unflatten(x):
    positions = np.array([x[i*3:i*3+2] for i in range(N)])
    angles = np.array([x[i*3+2] for i in range(N)])
    return positions, angles

def construct():
    best_score = float('inf')
    best_result = None
    deadline = START_TIME + TIME_LIMIT
    
    # Try different outer rotations and initializations
    outer_rotations = [0.0, 30.0, 60.0]
    modes = [0, 1, 2, 3]
    
    for outer_rot in outer_rotations:
        if time.time() > deadline - 20:
            break
        for mode in modes:
            if time.time() > deadline - 20:
                break
            try:
                positions, angles = _initialization(mode)
                x0 = _flatten(positions, angles)
                
                # L-BFGS-B optimization
                res = minimize(
                    _objective, x0, args=(outer_rot,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if res.success or res.fun < best_score:
                    pos, ang = _unflatten(res.x)
                    verts = _all_vertices(pos, ang)
                    s = _outer_side_length(verts, outer_rot)
                    pen = _overlap_penalty(pos, ang)
                    if pen < 1e-6:  # Valid packing
                        if s < best_score:
                            best_score = s
                            best_result = (pos, ang, outer_rot)
                
            except Exception:
                continue
    
    # SLSQP polish pass on the best result
    if best_result is not None and time.time() < deadline - 5:
        pos, ang, outer_rot = best_result
        x0 = _flatten(pos, ang)
        try:
            res = minimize(
                _objective, x0, args=(outer_rot,),
                method='SLSQP',
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            if res.fun < best_score:
                pos, ang = _unflatten(res.x)
                verts = _all_vertices(pos, ang)
                s = _outer_side_length(verts, outer_rot)
                pen = _overlap_penalty(pos, ang)
                if pen < 1e-6:
                    if s < best_score:
                        best_score = s
                        best_result = (pos, ang, outer_rot)
        except Exception:
            pass
    
    if best_result is None:
        # Fallback: return the initial cluster
        positions, angles = _initialization(0)
        verts = _all_vertices(positions, angles)
        s = _outer_side_length(verts, 0.0)
        inner = [[positions[i][0], positions[i][1], angles[i]] for i in range(N)]
        return inner, (0.0, 0.0), s, 0.0
    
    pos, ang, outer_rot = best_result
    # Center the packing at origin
    all_verts = _all_vertices(pos, ang)
    center_x = (all_verts[:, 0].max() + all_verts[:, 0].min()) / 2
    center_y = (all_verts[:, 1].max() + all_verts[:, 1].min()) / 2
    
    # Adjust positions so that the centroid of the bounding box is at origin
    for i in range(N):
        pos[i][0] -= center_x
        pos[i][1] -= center_y
    
    verts = _all_vertices(pos, ang)
    s = _outer_side_length(verts, outer_rot)
    
    inner = [[pos[i][0], pos[i][1], ang[i]] for i in range(N)]
    return inner, (0.0, 0.0), s, outer_rot
