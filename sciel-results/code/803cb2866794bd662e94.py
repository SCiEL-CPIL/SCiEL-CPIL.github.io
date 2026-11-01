"""Rigid-body cluster optimization for packing 12 unit hexagons in a regular hexagon."""
import numpy as np
import time
from scipy.spatial import ConvexHull
import math

N = 12
DEADLINE = 150.0  # seconds, evaluator kills at 150s, we use 145s for safety

# Pre-compute unit hexagon vertices (flat-top orientation, side length 1)
# Vertices at angles 0, 60, 120, 180, 240, 300 degrees
_UNIT_HEX_VERTS = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side length and rotation."""
    a = math.radians(angle_deg)
    R = side
    verts = np.array([[math.cos(math.radians(k * 60 + angle_deg)), math.sin(math.radians(k * 60 + angle_deg))] for k in range(6)]) * R
    return verts + np.array([cx, cy])

def _rotate_point(p, angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([c * p[0] - s * p[1], s * p[0] + c * p[1]])

def _hexagon_contains_point(hx, hy, hside, hangle_deg, px, py):
    """Check if point (px,py) is inside or on boundary of hexagon centered at (hx,hy)."""
    # Transform point to hexagon's local frame
    local_x = px - hx
    local_y = py - hy
    a = math.radians(-hangle_deg)
    lx = math.cos(a) * local_x - math.sin(a) * local_y
    ly = math.sin(a) * local_x + math.cos(a) * local_y
    
    # For a regular hexagon with side s, the inscribed radius (apothem) is s * sqrt(3)/2
    # The hexagon in local coords (flat-top, angle 0) has vertices at (s,0), (s/2, s*sqrt(3)/2), etc.
    # A point is inside if it satisfies all 6 half-plane inequalities.
    # For flat-top hexagon with side s:
    #   |x| <= s
    #   |y| <= s*sqrt(3)/2
    #   |x/2 + y*sqrt(3)/2| <= s  (wait, let me think more carefully)
    
    # Actually, for a regular hexagon with vertices at angles 0,60,120,180,240,300 and circumradius s:
    # The six edge normals point at 30, 90, 150, 210, 270, 330 degrees
    # The apothem (distance from center to each edge) = s * cos(30°) = s * sqrt(3)/2
    # Point is inside if for all 6 edges: dot(point, normal) <= apothem
    
    apothem = hside * math.sqrt(3) / 2
    for k in range(6):
        angle = math.radians(60 * k + 30)
        nx = math.cos(angle)
        ny = math.sin(angle)
        if nx * lx + ny * ly > apothem + 1e-12:
            return False
    return True

def _hexagon_pair_disjoint(h1x, h1y, h1angle, h2x, h2y, h2angle, side=1.0):
    """Check if two unit hexagons are disjoint (no interior overlap). Allow touching."""
    # Quick bounding box check
    r = side * math.sqrt(3)  # diameter
    if abs(h1x - h2x) > 2*r or abs(h1y - h2y) > 2*r:
        return True
    
    # Check if any vertex of hex1 is inside hex2, or any vertex of hex2 is inside hex1
    v1 = _hex_vertices(h1x, h1y, side, h1angle)
    v2 = _hex_vertices(h2x, h2y, side, h2angle)
    
    for p in v1:
        if _hexagon_contains_point(h2x, h2y, side, h2angle, p[0], p[1]):
            return False
    for p in v2:
        if _hexagon_contains_point(h1x, h1y, side, h1angle, p[0], p[1]):
            return False
    
    # Check edge-edge intersections
    for i in range(6):
        a1 = v1[i]
        b1 = v1[(i + 1) % 6]
        for j in range(6):
            a2 = v2[j]
            b2 = v2[(j + 1) % 6]
            if _segments_intersect(a1, b1, a2, b2):
                return False
    
    return True

def _segments_intersect(p1, p2, p3, p4):
    """Check if line segment p1-p2 intersects line segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _all_inside_outer(inner_positions, outer_cx, outer_cy, outer_side, outer_angle):
    """Check if all inner hexagon vertices are inside the outer hexagon."""
    for (ix, iy, iangle) in inner_positions:
        verts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in verts:
            if not _hexagon_contains_point(outer_cx, outer_cy, outer_side, outer_angle, v[0], v[1]):
                return False
    return True

def _all_disjoint(inner_positions):
    """Check if all pairs of inner hexagons are disjoint."""
    n = len(inner_positions)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagon_pair_disjoint(inner_positions[i][0], inner_positions[i][1], inner_positions[i][2],
                                          inner_positions[j][0], inner_positions[j][1], inner_positions[j][2]):
                return False
    return True

def _min_enclosing_hexagon(inner_positions):
    """Find minimum side length and orientation of a regular hexagon enclosing all inner hexagons.
    Returns (min_side, best_angle, best_cx, best_cy) or (inf, 0, 0, 0) if failed."""
    # Collect all vertices of all inner hexagons
    all_verts = []
    for (ix, iy, iangle) in inner_positions:
        verts = _hex_vertices(ix, iy, 1.0, iangle)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # The enclosing regular hexagon must contain all these points.
    # For a regular hexagon with center (cx,cy), side s, angle theta:
    # A point (px,py) is inside iff for all k in 0..5:
    #   dot((px-cx, py-cy), n_k) <= s * sqrt(3)/2
    # where n_k = (cos(60k+30+theta), sin(60k+30+theta))
    # So s >= max over all points and all k of dot((px-cx,py-cy), n_k) / (sqrt(3)/2)
    # = (2/sqrt(3)) * max_k dot((px-cx,py-cy), n_k)
    
    # We need to minimize s over (cx, cy, theta).
    # For fixed theta, the optimal (cx, cy) can be found, and s is determined.
    # This is a convex problem in (cx, cy) for fixed theta.
    
    best_side = float('inf')
    best_cx, best_cy, best_theta = 0, 0, 0
    
    # Try many orientations
    for theta_deg in range(0, 360, 5):
        theta = math.radians(theta_deg)
        # Compute the required s for a range of centers
        # For fixed theta, s(cx,cy) = (2/sqrt(3)) * max_k max_p dot((p - c), n_k)
        # This is the maximum over k of: (2/sqrt(3)) * max_p [dot(p, n_k) - dot(c, n_k)]
        # = (2/sqrt(3)) * max_k [max_p dot(p, n_k) - dot(c, n_k)]
        
        # To minimize this over c, we want to maximize dot(c, n_k) for the k that gives the max.
        # This is equivalent to finding the center of the minimal enclosing hexagon.
        
        # For each direction n_k, compute max_p dot(p, n_k)
        max_dots = []
        for k in range(6):
            angle = math.radians(60 * k + 30 + theta_deg)
            nx, ny = math.cos(angle), math.sin(angle)
            dots = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            max_dots.append(np.max(dots))
        
        # For a given center c, the required side is (2/sqrt(3)) * max_k (max_dots[k] - dot(c, n_k))
        # We want to minimize this. The optimal c equalizes the constraints.
        # Let f_k(c) = max_dots[k] - dot(c, n_k). We want min_c max_k f_k(c).
        # This is a linear programming problem: minimize t subject to dot(c, n_k) >= max_dots[k] - t for all k.
        
        # Use scipy to solve this LP, or use a simple approach.
        # For 6 directions, we can try the intersection of pairs of constraints.
        
        # Simpler: use scipy.optimize to minimize the max
        from scipy.optimize import minimize
        
        def objective(c):
            cx, cy = c
            vals = []
            for k in range(6):
                angle = math.radians(60 * k + 30 + theta_deg)
                nx, ny = math.cos(angle), math.sin(angle)
                vals.append(max_dots[k] - (nx * cx + ny * cy))
            return max(vals)
        
        # Initial guess: centroid of all vertices
        c0 = np.array([np.mean(all_verts[:, 0]), np.mean(all_verts[:, 1])])
        
        result = minimize(objective, c0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if result.success:
            cx, cy = result.x
            s = (2 / math.sqrt(3)) * result.fun
            if s < best_side:
                best_side = s
                best_cx, best_cy, best_theta = cx, cy, theta_deg
    
    # Refine with finer angle search around best_theta
    if best_side < float('inf'):
        for theta_deg in np.arange(best_theta - 5, best_theta + 5, 0.5):
            theta = math.radians(theta_deg)
            max_dots = []
            for k in range(6):
                angle = math.radians(60 * k + 30 + theta_deg)
                nx, ny = math.cos(angle), math.sin(angle)
                dots = all_verts[:, 0] * nx + all_verts[:, 1] * ny
                max_dots.append(np.max(dots))
            
            from scipy.optimize import minimize
            
            def objective(c):
                cx, cy = c
                vals = []
                for k in range(6):
                    angle = math.radians(60 * k + 30 + theta_deg)
                    nx, ny = math.cos(angle), math.sin(angle)
                    vals.append(max_dots[k] - (nx * cx + ny * cy))
                return max(vals)
            
            c0 = np.array([best_cx, best_cy])
            result = minimize(objective, c0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if result.success:
                cx, cy = result.x
                s = (2 / math.sqrt(3)) * result.fun
                if s < best_side:
                    best_side = s
                    best_cx, best_cy, best_theta = cx, cy, theta_deg
    
    return best_side, best_cx, best_cy, best_theta

def _honeycomb_positions():
    """Generate a compact honeycomb arrangement of 12 unit hexagons.
    In a honeycomb, adjacent hexagons share edges.
    The centers of adjacent hexagons are separated by sqrt(3) (the distance between centers of edge-sharing hexagons).
    
    Layout: 3 rows of 4, or 4 rows of 3, with offset.
    Let's do a 4x3 grid with hexagonal offset.
    Row 0: 4 hexagons
    Row 1: 3 hexagons (offset by half)
    Row 2: 4 hexagons
    Row 3: 1 hexagon (to make 12) -- actually let's do a more balanced arrangement.
    
    Better: a "flower" pattern or a compact cluster.
    Let's use a standard honeycomb tiling pattern.
    """
    # Hexagonal grid coordinates: each hex at (col, row)
    # In a honeycomb, moving right: dx = 1.5, dy = 0
    # Moving to next row: dx = 0.75, dy = sqrt(3)/2
    # Distance between centers of adjacent hexes = sqrt(3) for edge-sharing
    
    # Let's place them in a compact 4-wide, 3-tall arrangement with offsets
    positions = []
    # Row 0: 4 hexagons
    for i in range(4):
        positions.append((1.5 * i, 0.0, 0.0))
    # Row 1: 3 hexagons, offset
    for i in range(3):
        positions.append((1.5 * i + 0.75, math.sqrt(3) / 2, 0.0))
    # Row 2: 4 hexagons
    for i in range(4):
        positions.append((1.5 * i, math.sqrt(3), 0.0))
    # That's 11. Add one more.
    # Row 3: 1 hexagon
    positions.append((0.75, 3 * math.sqrt(3) / 2, 0.0))
    
    return positions[:N]

def _compact_cluster_positions():
    """A more compact arrangement: center + 6 around + 5 more.
    Like a flower with 7 hexagons (center + 6), then add 5 more around.
    """
    positions = []
    # Center
    positions.append((0.0, 0.0, 0.0))
    # 6 around the center
    for k in range(6):
        angle = math.radians(60 * k)
        r = math.sqrt(3)
        positions.append((r * math.cos(angle), r * math.sin(angle), 0.0))
    # That's 7. Need 5 more.
    # Add them in the "gaps" or extend the pattern.
    # The outer ring has 6 hexagons. Between each pair of adjacent outer hexagons,
    # there's a gap where another hexagon can fit.
    # Actually, in a honeycomb, the next ring would be at distance 2*sqrt(3) from center.
    # Let's add 5 hexagons at the second ring positions.
    for k in range(5):
        angle = math.radians(60 * k + 30)
        r = 2 * math.sqrt(3)
        positions.append((r * math.cos(angle), r * math.sin(angle), 0.0))
    
    return positions

def construct():
    start_time = time.time()
    
    # Generate candidate cluster arrangements
    candidates = []
    
    # Honeycomb 4-3-4-1
    cand1 = _honeycomb_positions()
    candidates.append(cand1)
    
    # Compact flower + 5
    cand2 = _compact_cluster_positions()
    candidates.append(cand2)
    
    # Another arrangement: 3-4-3-2
    cand3 = []
    for i in range(3):
        cand3.append((1.5 * i, 0.0, 0.0))
    for i in range(4):
        cand3.append((1.5 * i + 0.75, math.sqrt(3) / 2, 0.0))
    for i in range(3):
        cand3.append((1.5 * i, math.sqrt(3), 0.0))
    for i in range(2):
        cand3.append((1.5 * i + 0.75, 3 * math.sqrt(3) / 2, 0.0))
    candidates.append(cand3[:N])
    
    # 2-4-4-2
    cand4 = []
    for i in range(2):
        cand4.append((1.5 * i + 0.75, 0.0, 0.0))
    for i in range(4):
        cand4.append((1.5 * i, math.sqrt(3) / 2, 0.0))
    for i in range(4):
        cand4.append((1.5 * i + 0.75, math.sqrt(3), 0.0))
    for i in range(2):
        cand4.append((1.5 * i + 0.75, 3 * math.sqrt(3) / 2, 0.0))
    candidates.append(cand4[:N])
    
    # Verify disjointness for each candidate
    valid_candidates = []
    for cand in candidates:
        if len(cand) != N:
            continue
        if _all_disjoint(cand):
            valid_candidates.append(cand)
    
    if not valid_candidates:
        # Fallback: use the seed arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # For each valid candidate, find the minimum enclosing hexagon
    best_result = None
    best_side = float('inf')
    
    for cand in valid_candidates:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        side, cx, cy, angle = _min_enclosing_hexagon(cand)
        
        if side < best_side:
            best_side = side
            best_result = (cand, cx, cy, angle)
    
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner, cx, cy, angle = best_result
    
    # Verify the solution
    if not _all_inside_outer(inner, cx, cy, best_side, angle):
        # Add a small margin
        best_side *= 1.001
    
    if time.time() - start_time > DEADLINE - 5:
        return inner, (cx, cy), best_side, angle
    
    # Try to improve by slightly rotating the entire cluster
    improved = False
    for rot_deg in np.arange(-5, 5.5, 0.5):
        if time.time() - start_time > DEADLINE - 10:
            break
        rotated = []
        for (ix, iy, iangle) in inner:
            new_angle = iangle + rot_deg
            # Rotate position around origin
            a = math.radians(rot_deg)
            nx = math.cos(a) * ix - math.sin(a) * iy
            ny = math.sin(a) * ix + math.cos(a) * iy
            rotated.append((nx, ny, new_angle))
        
        if _all_disjoint(rotated):
            side, cx2, cy2, angle2 = _min_enclosing_hexagon(rotated)
            if side < best_side:
                best_side = side
                inner = rotated
                cx, cy, angle = cx2, cy2, angle2
                improved = True
    
    # Verify final solution
    if not _all_inside_outer(inner, cx, cy, best_side, angle):
        best_side *= 1.002
    
    return inner, (cx, cy), best_side, angle
