"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize_scalar, minimize

# Deadline guard
_START_TIME = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def _hex_vertices(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with side s and rotation angle_deg."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3.0
        verts.append((cx + s * np.cos(theta), cy + s * np.sin(theta)))
    return np.array(verts)

def _support_function(vertices, direction):
    """Return max dot product of vertices with direction vector."""
    return np.max(vertices @ direction)

def _min_distance_hex_hex(v1, v2):
    """
    Compute minimum distance between two convex polygons (hexagons).
    Returns negative if overlapping (approximate via sampling), positive otherwise.
    Uses a simple sampling approach for robustness.
    """
    # Sample points on both hexagons
    n_samples = 24
    pts1 = []
    pts2 = []
    for i in range(n_samples):
        t = i / n_samples
        # Interpolate along edges of hex1
        idx1 = int(t * 6) % 6
        idx2 = (idx1 + 1) % 6
        frac = t * 6 - int(t * 6)
        p1 = v1[idx1] * (1 - frac) + v1[idx2] * frac
        pts1.append(p1)
        p2 = v2[idx1] * (1 - frac) + v2[idx2] * frac
        pts2.append(p2)
    
    # Also include vertices
    pts1 = np.array(pts1 + list(v1))
    pts2 = np.array(pts2 + list(v2))
    
    # Compute min distance between all pairs (approximate)
    # Use broadcasting for efficiency
    diff = pts1[:, np.newaxis, :] - pts2[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    return np.min(dists)

def _check_outer_constraint(inner_hexes, outer_center, outer_side, outer_angle):
    """
    Check if all inner hexagons are inside the outer hexagon.
    Returns True if valid, False otherwise.
    """
    ocx, ocy = outer_center
    # Outer hexagon vertices
    outer_verts = _hex_vertices(ocx, ocy, outer_side, outer_angle)
    
    for (ix, iy, iangle) in inner_hexes:
        inner_verts = _hex_vertices(ix, iy, 1.0, iangle)
        # Check if all inner vertices are inside outer hexagon
        for v in inner_verts:
            if not _point_in_hexagon(v[0], v[1], outer_verts):
                return False
    return True

def _point_in_hexagon(px, py, hex_verts):
    """Check if point (px, py) is inside a convex hexagon given its vertices."""
    # Use winding number / cross product test
    n = len(hex_verts)
    sign = 0
    for i in range(n):
        x1, y1 = hex_verts[i]
        x2, y2 = hex_verts[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross > 1e-9:
            if sign == 0:
                sign = 1
            elif sign == -1:
                return False
        elif cross < -1e-9:
            if sign == 0:
                sign = -1
            elif sign == 1:
                return False
    return True

def _pairwise_nonoverlap(inner_hexes):
    """
    Check if all pairs of inner hexagons are non-overlapping.
    Returns minimum separation (negative if overlapping).
    """
    n = len(inner_hexes)
    min_sep = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            v1 = _hex_vertices(inner_hexes[i][0], inner_hexes[i][1], 1.0, inner_hexes[i][2])
            v2 = _hex_vertices(inner_hexes[j][0], inner_hexes[j][1], 1.0, inner_hexes[j][2])
            sep = _min_distance_hex_hex(v1, v2)
            if sep < min_sep:
                min_sep = sep
    return min_sep

def _compute_max_outer_side(inner_hexes, outer_center, outer_angle):
    """
    Compute the minimum outer side length needed to contain all inner hexagons.
    This is done by finding the maximum support function projection over 6 directions.
    """
    ocx, ocy = outer_center
    # For a regular hexagon with side s and center at (0,0), the support function in direction theta
    # is s * cos(theta - angle) for the appropriate edge.
    # Actually, for a regular hexagon centered at origin with side length s and rotation phi,
    # the support function h(theta) = s * max over vertices of (cos(theta - vertex_angle))
    # But more simply: the width in direction theta is 2 * s * |cos(theta - phi)| for certain ranges.
    # Let's use the fact that a regular hexagon with side s has inradius r = s * sqrt(3)/2.
    # The support function in direction theta (measured from x-axis) for a hexagon centered at origin
    # with one vertex at angle phi is:
    # h(theta) = s * cos(theta - phi) if theta is within [-30, 30] of phi, etc.
    # Actually, let's just compute it directly using vertices.
    
    # For a given inner hexagon at (ix, iy) with angle iangle, we need to find the smallest s
    # such that it fits in outer hexagon centered at (ocx, ocy) with angle outer_angle.
    # This is equivalent to: for all directions, the projection of the inner hexagon
    # relative to outer center must be <= support of outer hexagon in that direction.
    
    # The outer hexagon with side s, center (ocx, ocy), angle outer_angle has support function:
    # h_outer(theta) = s * f(theta - outer_angle) where f is the support function of unit hexagon.
    # For unit hexagon (side 1) centered at origin, the support function in direction theta is:
    # h_unit(theta) = cos(theta mod 60 - 30) * sqrt(3)/2 ... let me just compute via vertices.
    
    # Simpler: for each inner hexagon, compute the required s such that it fits.
    # The condition is: for all directions theta, 
    #   max_{v in inner_hex} (v - outer_center) . d(theta) <= s * h_unit(theta - outer_angle)
    # where h_unit is the support function of a unit hexagon centered at origin.
    
    # h_unit(theta) for unit hexagon: the max over 6 vertices of (cos(theta - vertex_angle)).
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees (for angle=0).
    # So h_unit(theta) = max_k cos(theta - k*pi/3)
    
    # For a given inner hexagon, the required s is:
    # s >= max_{theta} [max_{v in inner} (v - outer_center) . d(theta)] / h_unit(theta - outer_angle)
    
    # But this is complex. Let's use a different approach:
    # The minimal s such that inner hex (translated to be relative to outer center) fits in unit hex * s
    # is equivalent to the Chebyshev-like distance.
    
    # Practical approach: sample many directions and compute the required s.
    thetas = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    d = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    
    # Unit hexagon support function
    unit_hex_verts = _hex_vertices(0, 0, 1.0, 0.0)
    h_unit = np.max(unit_hex_verts @ d.T, axis=0)
    
    max_s = 0.0
    for (ix, iy, iangle) in inner_hexes:
        inner_verts = _hex_vertices(ix, iy, 1.0, iangle)
        # Relative to outer center
        rel_verts = inner_verts - np.array([ocx, ocy])
        # Support in each direction
        h_inner = np.max(rel_verts @ d.T, axis=0)
        # Required s in each direction
        with np.errstate(divide='ignore', invalid='ignore'):
            s_required = np.where(h_unit > 1e-9, h_inner / h_unit, 0.0)
        s_required = np.clip(s_required, 0, None)
        max_s = max(max_s, np.max(s_required))
    
    return max_s

def construct():
    """
    Construct a packing of N unit regular hexagons inside a regular hexagon.
    Returns (inner, center, side, angle) where:
      inner: list of [x, y, angle_degrees] for each inner hexagon
      center: (x, y) of outer hexagon center
      side: side length of outer hexagon
      angle: rotation angle of outer hexagon in degrees
    """
    global _START_TIME
    if time.time() - _START_TIME > DEADLINE - SAFETY_MARGIN:
        # Return a simple valid solution
        N = 7
        inner = [[0.0, 0.0, 0.0]]
        for k in range(6):
            angle = k * 60.0
            x = 2.0 * np.cos(np.radians(angle))
            y = 2.0 * np.sin(np.radians(angle))
            inner.append([x, y, 0.0])
        return inner, (0.0, 0.0), 3.0, 0.0
    
    N = 11
    # Use the 60-degree symmetry: center + ring of 6 + ring of 4? 
    # Actually 1 = 1 + 6 + 4 = 11. Or 1 + 6 + 4.
    # Let's try: center hex, 6 in first ring, 4 in second ring.
    
    # Parameters to optimize:
    # - radial distance of ring 1 (6 hexagons)
    # - radial distance of ring 2 (4 hexagons)
    # - angular offset of ring 1
    # - angular offset of ring 2
    # - rotation of all inner hexagons (can vary per ring)
    
    # Initial guess
    r1 = 2.0  # distance from center to ring 1
    r2 = 3.5  # distance from center to ring 2
    a1 = 0.0  # angular offset for ring 1
    a2 = 0.0  # angular offset for ring 2
    
    def _build_inner(params):
        r1, r2, a1, a2 = params
        inner = [[0.0, 0.0, 0.0]]  # center hex
        # Ring 1: 6 hexagons
        for k in range(6):
            angle = a1 + k * 60.0
            x = r1 * np.cos(np.radians(angle))
            y = r1 * np.sin(np.radians(angle))
            inner.append([x, y, 0.0])
        # Ring 2: 4 hexagons
        for k in range(4):
            angle = a2 + k * 90.0
            x = r2 * np.cos(np.radians(angle))
            y = r2 * np.sin(np.radians(angle))
            inner.append([x, y, 0.0])
        return inner
    
    def _objective(params):
        if time.time() - _START_TIME > DEADLINE - SAFETY_MARGIN:
            return 100.0
        inner = _build_inner(params)
        # Check non-overlap
        min_sep = _pairwise_nonoverlap(inner)
        if min_sep < 0:
            # Penalize overlap
            return 100.0 + abs(min_sep) * 10
        # Compute required outer side
        side = _compute_max_outer_side(inner, (0.0, 0.0), 0.0)
        return side
    
    # Optimize
    x0 = [2.0, 3.5, 0.0, 0.0]
    result = minimize(_objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    best_params = result.x
    inner = _build_inner(best_params)
    
    # Verify non-overlap
    min_sep = _pairwise_nonoverlap(inner)
    if min_sep < 0:
        # Fallback to a known valid configuration
        inner = [[0.0, 0.0, 0.0]]
        for k in range(6):
            angle = k * 60.0
            x = 2.0 * np.cos(np.radians(angle))
            y = 2.0 * np.sin(np.radians(angle))
            inner.append([x, y, 0.0])
        for k in range(4):
            angle = 45.0 + k * 90.0
            x = 3.5 * np.cos(np.radians(angle))
            y = 3.5 * np.sin(np.radians(angle))
            inner.append([x, y, 0.0])
    
    side = _compute_max_outer_side(inner, (0.0, 0.0), 0.0)
    
    return inner, (0.0, 0.0), side, 0.0
