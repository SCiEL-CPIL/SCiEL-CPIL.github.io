"""Optimized packing of 12 unit hexagons in a minimal enclosing regular hexagon."""
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START

def _hexagon_vertices(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon with center (cx,cy), side s, rotation angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + np.radians(60 * i)
        verts.append((cx + s * np.cos(theta), cy + s * np.sin(theta)))
    return np.array(verts)


def _support_function_projection(vertices, direction):
    """Max projection of a set of vertices onto a direction vector."""
    return np.max(vertices @ direction)


def _compute_min_enclosing_hexagon(inner_positions, inner_angles):
    """
    For fixed inner hexagon positions and angles, compute the minimal enclosing regular hexagon.
    We use the support function approach: the outer hexagon must contain all inner vertices.
    
    The outer hexagon is defined by center (cx, cy), side length s, and rotation phi.
    A point p is inside the outer hexagon iff for all 6 outward normals n_i of the outer hexagon,
    p . n_i <= cx . n_i + s * sqrt(3) / 2  (the apothem is s*sqrt(3)/2)
    
    Wait, let's be precise. For a regular hexagon with side length s, the distance from center to
    each side (apothem) is s * sqrt(3) / 2. The outward normal directions are at angles
    phi + 60*i degrees.
    
    For containment: for each inner vertex v, and for each of the 6 outward normals n_j of the
    outer hexagon, we need:
        v . n_j <= center . n_j + apothem
    
    Equivalently: center . n_j >= v . n_j - apothem for all v, j.
    And also: center . n_j <= v . n_j + apothem is not needed; we just need the max over v of
    (v . n_j - center . n_j) <= apothem for all j.
    
    So: max_j [ max_v (v . n_j) - center . n_j ] <= apothem = s * sqrt(3) / 2
    
    To minimize s, for a given center and phi, we need:
        s >= (2 / sqrt(3)) * max_j [ max_v (v . n_j) - center . n_j ]
    
    But we also need the inner hexagons to be disjoint. Let's first handle containment, then
    add non-overlap constraints.
    
    Actually, let's think about this differently. Given positions and angles of inner hexagons,
    the minimal enclosing regular hexagon (with some rotation phi) can be found by:
    
    1. Collect all vertices of all inner hexagons.
    2. For a given phi, the required apothem is:
       a(phi) = max over j=0..5 of [ max over all vertices v of (v . n_j(phi)) - center . n_j(phi) ]
       where n_j(phi) = (cos(phi + 60j), sin(phi + 60j))
    3. Then s = 2 * a(phi) / sqrt(3)
    
    But we can also choose the center. For a fixed phi, the optimal center minimizes the max
    projection difference. This is a Chebyshev-type problem.
    
    Actually, for a fixed set of points and fixed phi, the minimal enclosing hexagon with that
    orientation is centered at the centroid of the "dual" problem. Let's just optimize over
    center and phi jointly.
    
    Let's define the objective as the side length s, which depends on center (cx, cy) and phi.
    """
    # Collect all vertices
    all_verts = []
    for (x, y, ang) in inner_positions:
        v = _hexagon_vertices(x, y, 1.0, ang)
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    def objective(params):
        cx, cy, phi = params
        center = np.array([cx, cy])
        apothem = 0.0
        for j in range(6):
            theta = phi + np.radians(60 * j)
            n = np.array([np.cos(theta), np.sin(theta)])
            # max projection of all vertices onto n
            proj = np.max(all_verts @ n)
            diff = proj - center @ n
            if diff > apothem:
                apothem = diff
        return 2.0 * apothem / np.sqrt(3)
    
    # Initial guess: center at centroid of all vertices, phi = 0
    cx0 = np.mean(all_verts[:, 0])
    cy0 = np.mean(all_verts[:, 1])
    res = minimize(objective, [cx0, cy0, 0.0], method='Nelder-Mead',
                   options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
    return res.x, res.fun


def _hexagon_overlap(v1, v2, tol=1e-9):
    """Check if two unit hexagons overlap (their interiors intersect)."""
    # For two regular hexagons of side 1, they overlap if and only if
    # the distance between centers is less than 2 (since the inradius is sqrt(3)/2,
    # and two hexagons touch when center distance = 2 * inradius = sqrt(3) ~ 1.732? No.)
    
    # Actually, for two regular hexagons of side length 1, the minimum distance between
    # centers for non-overlap depends on their relative orientations. The worst case (most
    # overlap-prone) is when they are aligned, in which case the center distance must be
    # at least 2 (the diameter across opposite vertices is 2, but the distance across
    # opposite sides is sqrt(3)).
    
    # More precisely: two hexagons of side 1 do NOT overlap if the distance between
    # centers is >= 2 * inradius = sqrt(3) when aligned with parallel sides? No.
    
    # Let's use a simpler check: two convex polygons overlap if and only if they share
    # a common interior point. We can check this by seeing if any vertex of one is inside
    # the other, or if any edges intersect.
    
    # For efficiency, first check center distance. If distance > 2, definitely no overlap.
    # The maximum distance from center to any point in a unit hexagon is 1 (circumradius = side length).
    # So if center distance > 2, no overlap.
    
    c1 = np.array(v1[:2])
    c2 = np.array(v2[:2])
    dist = np.linalg.norm(c1 - c2)
    if dist > 2.0:
        return False
    
    # Get vertices
    verts1 = _hexagon_vertices(v1[0], v1[1], 1.0, v1[2])
    verts2 = _hexagon_vertices(v2[0], v2[1], 1.0, v2[2])
    
    # Check if any vertex of 1 is inside 2, or any vertex of 2 is inside 1
    def point_in_hexagon(p, verts):
        """Check if point p is inside or on boundary of hexagon with vertices verts."""
        # Use cross product test
        n = len(verts)
        for i in range(n):
            a = verts[i]
            b = verts[(i + 1) % n]
            cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            if cross < -tol:
                return False
        return True
    
    for v in verts1:
        if point_in_hexagon(v, verts2):
            return True
    for v in verts2:
        if point_in_hexagon(v, verts1):
            return True
    
    # Check edge intersections
    def segments_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects segment p3-p4."""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        
        if ((d1 > tol and d2 < -tol) or (d1 < -tol and d2 > tol)) and \
           ((d3 > tol and d4 < -tol) or (d3 < -tol and d4 > tol)):
            return True
        return False
    
    n1 = len(verts1)
    n2 = len(verts2)
    for i in range(n1):
        for j in range(n2):
            if segments_intersect(verts1[i], verts1[(i+1)%n1], verts2[j], verts2[(j+1)%n2]):
                return True
    
    return False


def _check_disjoint(inner_positions):
    """Check if all inner hexagons are pairwise disjoint."""
    n = len(inner_positions)
    for i in range(n):
        for j in range(i + 1, n):
            if _hexagon_overlap(inner_positions[i], inner_positions[j]):
                return False
    return True


def _penalty_function(params, inner_positions_template, inner_angles):
    """
    Penalty function that returns 0 if valid, positive if invalid.
    params: flattened array of 3*N values: x1, y1, x2, y2, ...
    """
    positions = []
    for i in range(N):
        positions.append([params[3*i], params[3*i+1], inner_angles[i]])
    
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            if _hexagon_overlap(positions[i], positions[j]):
                penalty += 1.0
    
    return penalty


def construct():
    """
    Construct 12 unit hexagons inside a minimal enclosing regular hexagon.
    Returns: inner (list of [x, y, angle_degrees]), center (x, y), side_length, angle_degrees
    """
    # Start with a hexagonal lattice arrangement
    # 12 hexagons: arrange in a 4-3-3-2 pattern or similar
    
    # Let's try a honeycomb-like arrangement
    # Row 0: 4 hexagons
    # Row 1: 3 hexagons (offset)
    # Row 2: 3 hexagons (offset)
    # Row 3: 2 hexagons (offset)
    
    # For unit hexagons in honeycomb, horizontal spacing is 1.5 (for pointy-top)
    # or sqrt(3) (for flat-top). Let's use flat-top hexagons (angle 0).
    
    # Flat-top hexagon: vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # Horizontal distance between centers in same row: 2 * cos(30) = sqrt(3) ~ 1.732
    # Vertical distance between rows: 1.5
    # Offset in alternating rows: sqrt(3)/2
    
    positions = []
    angles = [0.0] * N
    
    # Honeycomb arrangement
    # Let's place them in a compact cluster
    # Using axial coordinates for hex grid
    
    # Center hexagon at origin
    positions.append([0.0, 0.0, 0.0])
    
    # 6 neighbors at distance sqrt(3) in directions 0, 60, 120, 180, 240, 300
    for i in range(6):
        theta = np.radians(60 * i)
        x = np.sqrt(3) * np.cos(theta)
        y = np.sqrt(3) * np.sin(theta)
        positions.append([x, y, 0.0])
    
    # 5 more hexagons: place them at distance 2*sqrt(3) in some directions
    # To make a compact shape, place them in the "gaps"
    # In a honeycomb, the next ring has 12 positions. We need 5 more.
    # Place them at:
    for i in range(5):
        theta = np.radians(60 * i)
        x = 2 * np.sqrt(3) * np.cos(theta)
        y = 2 * np.sqrt(3) * np.sin(theta)
        positions.append([x, y, 0.0])
    
    # Check if this arrangement is valid (disjoint)
    if not _check_disjoint(positions):
        # Try a different arrangement
        # Linear arrangement as fallback
        positions = [[3.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    
    # Now optimize the positions to minimize the enclosing hexagon size
    # while maintaining disjointness
    
    # Use the support function approach to compute the enclosing hexagon
    # and optimize positions
    
    def total_objective(params):
        """
        Objective: side length of enclosing hexagon + penalty for overlaps
        """
        pos = []
        for i in range(N):
            pos.append([params[3*i], params[3*i+1], angles[i]])
        
        # Compute enclosing hexagon
        all_verts = []
        for (x, y, ang) in pos:
            v = _hexagon_vertices(x, y, 1.0, ang)
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        cx = np.mean(all_verts[:, 0])
        cy = np.mean(all_verts[:, 1])
        
        def side_length_for_phi(phi):
            apothem = 0.0
            for j in range(6):
                theta = phi + np.radians(60 * j)
                n = np.array([np.cos(theta), np.sin(theta)])
                proj = np.max(all_verts @ n)
                diff = proj - (cx * np.cos(theta) + cy * np.sin(theta))
                if diff > apothem:
                    apothem = diff
            return 2.0 * apothem / np.sqrt(3)
        
        # Optimize over phi
        phi_res = minimize(side_length_for_phi, [0.0], method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        side = phi_res.fun
        
        # Penalty for overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                if _hexagon_overlap(pos[i], pos[j]):
                    penalty += 100.0
        
        return side + penalty
    
    # Initial parameters
    x0 = []
    for i in range(N):
        x0.extend([positions[i][0], positions[i][1]])
    
    # Optimize
    res = minimize(total_objective, x0, method='Nelder-Mead',
                  options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    # Extract optimized positions
    opt_positions = []
    for i in range(N):
        opt_positions.append([res.x[3*i], res.x[3*i+1], angles[i]])
    
    # Verify disjointness
    if not _check_disjoint(opt_positions):
        # Fall back to original
        opt_positions = positions
    
    # Compute final enclosing hexagon
    all_verts = []
    for (x, y, ang) in opt_positions:
        v = _hexagon_vertices(x, y, 1.0, ang)
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    cx = np.mean(all_verts[:, 0])
    cy = np.mean(all_verts[:, 1])
    
    def side_length_for_phi(phi):
        apothem = 0.0
        for j in range(6):
            theta = phi + np.radians(60 * j)
            n = np.array([np.cos(theta), np.sin(theta)])
            proj = np.max(all_verts @ n)
            diff = proj - (cx * np.cos(theta) + cy * np.sin(theta))
            if diff > apothem:
                apothem = diff
        return 2.0 * apothem / np.sqrt(3)
    
    phi_res = minimize(side_length_for_phi, [0.0], method='Nelder-Mead',
                      options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
    side = phi_res.fun
    phi_deg = np.degrees(phi_res.x[0])
    
    # Format output
    inner = [[round(p[0], 10), round(p[1], 10), round(p[2], 10)] for p in opt_positions]
    center = (round(cx, 10), round(cy, 10))
    
    return inner, center, round(side, 10), round(phi_deg, 10)

# EVOLVE-BLOCK-END
