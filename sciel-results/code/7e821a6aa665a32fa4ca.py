"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import linprog, minimize

N = 11

# EVOLVE-BLOCK-START

def _hexagon_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon (side s) centered at (cx,cy) rotated by angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return np.array(verts)


def _outer_hexagon_vertices(cx, cy, s, angle_deg):
    return _hexagon_vertices(cx, cy, s, angle_deg)


def _point_in_hexagon(p, center, s, angle_deg):
    """Check if point p is inside the hexagon with given center, side, angle."""
    cx, cy = center
    a = math.radians(angle_deg)
    # Transform point to hexagon local coords
    dx = p[0] - cx
    dy = p[1] - cy
    # Rotate by -angle
    ca = math.cos(-a)
    sa = math.sin(-a)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    # Hexagon with flat-top orientation (vertices at angles 0, 60, ...)
    # The hexagon has vertices at (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # So in local coords, the hexagon is defined by |lx| <= s, |ly| <= s*sqrt(3)/2, and |lx| + |ly|/sqrt(3) <= s ... 
    # Actually for a regular hexagon with vertex at (s,0), the constraints are:
    # |lx| <= s
    # |ly| <= s * sqrt(3)/2
    # |lx|/1 + |ly|/sqrt(3) <= s  ... no
    # Let me think: the hexagon has 6 sides. The normal vectors are at angles 30, 90, 150, 210, 270, 330 degrees.
    # Side at angle 30: normal (cos30, sin30), distance from center = s * cos(30) = s*sqrt(3)/2
    # So constraint: lx*cos(30) + ly*sin(30) <= s*sqrt(3)/2
    # Similarly for all 6 normals.
    for k in range(6):
        theta = k * math.pi / 3.0 + math.pi / 6.0
        nx = math.cos(theta)
        ny = math.sin(theta)
        d = s * math.cos(math.pi / 6.0)  # apothem = s * sqrt(3)/2
        if lx * nx + ly * ny > d + 1e-9:
            return False
    return True


def _hexagon_distance(h1, h2):
    """Minimum distance between two hexagons (positive if disjoint, 0 if touching, negative if overlapping).
    h = (cx, cy, angle_deg, s)"""
    cx1, cy1, a1, s1 = h1
    cx2, cy2, a2, s2 = h2
    v1 = _hexagon_vertices(cx1, cy1, s1, a1)
    v2 = _hexagon_vertices(cx2, cy2, s2, a2)
    
    # Check if any vertex of one is inside the other
    for v in v1:
        if _point_in_hexagon(v, (cx2, cy2), s2, a2):
            return -1.0  # overlapping
    for v in v2:
        if _point_in_hexagon(v, (cx1, cy1), s1, a1):
            return -1.0  # overlapping
    
    # Compute minimum distance between edges
    min_dist = float('inf')
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        for j in range(6):
            p3 = v2[j]
            p4 = v2[(j + 1) % 6]
            d = _segment_distance(p1, p2, p3, p4)
            if d < min_dist:
                min_dist = d
    return min_dist


def _segment_distance(p1, p2, p3, p4):
    """Minimum distance between segments p1-p2 and p3-p4."""
    # Check all 4 endpoint-to-segment distances
    d = min(
        _point_segment_distance(p1, p3, p4),
        _point_segment_distance(p2, p3, p4),
        _point_segment_distance(p3, p1, p2),
        _point_segment_distance(p4, p1, p2)
    )
    return d


def _point_segment_distance(p, a, b):
    """Distance from point p to segment a-b."""
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-15)
    t = max(0.0, min(1.0, t))
    closest = a + t * ab
    return np.linalg.norm(p - closest)


def _min_outer_side(inner_positions, outer_center, outer_angle):
    """Find minimum side length of outer hexagon (centered at outer_center, rotated by outer_angle)
    that contains all inner unit hexagons."""
    cx, cy = outer_center
    a = math.radians(outer_angle)
    
    # For each inner hexagon, compute the maximum projection onto each of the 6 outer hexagon normal directions
    # The outer hexagon with side s, center c, angle alpha has 6 supporting lines at distance s*sqrt(3)/2 from center
    # in directions 30+60k degrees.
    # For an inner hexagon to fit, for each normal direction n_k, the max projection of inner hex vertices onto n_k
    # minus min projection of outer center... 
    # Actually: the outer hexagon is defined by: for k=0..5, n_k · (x - c) <= s * sqrt(3)/2
    # where n_k = (cos(30+60k), sin(30+60k))
    # For all inner hex vertices x_i, we need n_k · (x_i - c) <= s * sqrt(3)/2 for all k
    # So s >= max over all inner vertices i and all k of [n_k · (x_i - c)] / (sqrt(3)/2)
    
    max_proj = 0.0
    for (ix, iy, ia) in inner_positions:
        verts = _hexagon_vertices(ix, iy, 1.0, ia)
        for v in verts:
            for k in range(6):
                theta = k * math.pi / 3.0 + math.pi / 6.0
                nx = math.cos(theta)
                ny = math.sin(theta)
                proj = (v[0] - cx) * nx + (v[1] - cy) * ny
                if proj > max_proj:
                    max_proj = proj
    
    return max_proj / (math.sqrt(3) / 2.0)


def _check_validity(inner_positions, outer_center, outer_side, outer_angle):
    """Check all constraints: inner hexagons are disjoint and inside outer hexagon."""
    # Check inner hexagons are inside outer
    for (ix, iy, ia) in inner_positions:
        verts = _hexagon_vertices(ix, iy, 1.0, ia)
        for v in verts:
            if not _point_in_hexagon(v, outer_center, outer_side, outer_angle):
                return False
    
    # Check inner hexagons are disjoint
    n_inner = len(inner_positions)
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            h1 = (inner_positions[i][0], inner_positions[i][1], inner_positions[i][2], 1.0)
            h2 = (inner_positions[j][0], inner_positions[j][1], inner_positions[j][2], 1.0)
            d = _hexagon_distance(h1, h2)
            if d < -1e-9:
                return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 130s safety margin
    
    best_score = float('inf')
    best_result = None
    
    # Strategy: Use a hexagonal lattice arrangement of unit hexagons
    # A unit regular hexagon has width 2 (flat-to-flat = sqrt(3), point-to-point = 2)
    # In a hexagonal lattice, the spacing between centers is 2*sqrt(3) for adjacent hexagons
    # (for the standard tiling where hexagons touch)
    
    # Let's try a compact hexagonal arrangement
    # For N hexagons, arrange them in a hexagonal cluster
    # A hexagonal number arrangement: 1, 7, 19, 37, 61, ...
    # For N=11, we can do a central hexagon + ring of 6 + 4 more
    
    # Actually, let's use a systematic approach:
    # Place hexagons in a hexagonal grid pattern
    # The natural tiling has centers at distance 2*sqrt(3) apart in 3 directions
    
    # Generate hexagonal lattice positions
    def hex_lattice_positions(n):
        """Generate n positions in a hexagonal lattice pattern (compact cluster)."""
        positions = []
        # Use axial coordinates (q, r)
        # Center is (0,0)
        # Ring k has 6k hexagons
        positions.append((0, 0))
        ring = 1
        while len(positions) < n:
            # Ring ring has 6*ring positions
            # Start at (ring, 0) and go around
            # In axial coords, ring r: 
            #   (r, 0), (r, -1), ..., (r, -r), (0, -r), (-r, -r+r), ... 
            # Actually let's use cube coordinates
            # For ring r, the 6r positions are:
            #   (r, -k, k-r) for k=0..r-1  [top-right edge]
            #   (-k, r, k-r) for k=0..r-1  [top-left edge]
            #   (-r, k, r-k) for k=1..r    [bottom-left edge]
            #   (k-r, -r, r-k) for k=1..r  [bottom edge]
            #   (r-k, -r+k, 0) for k=1..r-1  [bottom-right edge]
            #   (0, r-k, -r+k) for k=1..r-1  [right edge]
            # 
            # Simpler: use the 6 directions
            directions = [
                (1, 0), (1, -1), (0, -1),
                (-1, 0), (-1, 1), (0, 1)
            ]
            # Start position for ring r
            q, r_coord = r, 0
            for i in range(6 * ring):
                positions.append((q, r_coord))
                if len(positions) >= n:
                    break
                # Move to next position
                dir_idx = i // ring
                if i % ring == ring - 1:
                    dir_idx = (i // ring) + 1
                dq, dr = directions[dir_idx % 6]
                q += dq
                r_coord += dr
        
        return positions[:n]
    
    # Convert axial to Cartesian (for flat-top hexagons with side 1)
    def axial_to_cartesian(q, r):
        x = 2 * math.sqrt(3) * (q + r / 2.0)
        y = 3 * r
        return x, y
    
    # Initial arrangement
    lattice_pos = hex_lattice_positions(N)
    inner = []
    for (q, r) in lattice_pos:
        x, y = axial_to_cartesian(q, r)
        inner.append([x, y, 0.0])
    
    # Optimize: try to minimize outer side length
    # First, try different outer angles and centers
    
    best_local = None
    
    for outer_angle in [0, 30, 60, 90, 120, 150]:
        # Try to find optimal outer center
        # The optimal center is roughly the centroid of the inner hexagons
        centroid_x = sum(p[0] for p in inner) / N
        centroid_y = sum(p[1] for p in inner) / N
        
        # Try a few center offsets
        for dx in [-0.5, -0.25, 0.0, 0.25, 0.5]:
            for dy in [-0.5, -0.25, 0.0, 0.25, 0.5]:
                oc = (centroid_x + dx, centroid_y + dy)
                side = _min_outer_side(inner, oc, outer_angle)
                if side < best_score:
                    # Verify validity
                    if _check_validity(inner, oc, side, outer_angle):
                        best_score = side
                        best_result = (inner, oc, side, outer_angle)
    
    # Local perturbation to improve
    if best_result is not None:
        inner_best, oc_best, side_best, angle_best = best_result
        # Perturb inner positions slightly
        for iteration in range(50):
            if time.time() > deadline:
                break
            new_inner = [list(p) for p in inner_best]
            # Random small perturbation
            idx = np.random.randint(0, N)
            new_inner[idx][0] += np.random.uniform(-0.1, 0.1)
            new_inner[idx][1] += np.random.uniform(-0.1, 0.1)
            
            # Check if still valid (disjoint)
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    h1 = (new_inner[i][0], new_inner[i][1], new_inner[i][2], 1.0)
                    h2 = (new_inner[j][0], new_inner[j][1], new_inner[j][2], 1.0)
                    d = _hexagon_distance(h1, h2)
                    if d < -1e-9:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                for outer_angle in [angle_best, (angle_best + 15) % 180, (angle_best - 15) % 180]:
                    for dx in [-0.2, 0.0, 0.2]:
                        for dy in [-0.2, 0.0, 0.2]:
                            oc = (oc_best[0] + dx, oc_best[1] + dy)
                            side = _min_outer_side(new_inner, oc, outer_angle)
                            if side < best_score - 1e-9:
                                if _check_validity(new_inner, oc, side, outer_angle):
                                    best_score = side
                                    best_result = (new_inner, oc, side, outer_angle)
    
    if best_result is None:
        # Fallback: use a very large hexagon
        centroid_x = sum(p[0] for p in inner) / N
        centroid_y = sum(p[1] for p in inner) / N
        return inner, (centroid_x, centroid_y), 40.0, 0.0
    
    return best_result

# EVOLVE-BLOCK-END
