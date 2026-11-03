"""Structural redesign: hexagonal lattice packing with rotational symmetry and local search."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 12
DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def hexagon_vertices(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with side s, rotated by angle_deg."""
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    pts = []
    for i in range(6):
        theta = np.pi / 3 * i + a
        x = cx + s * np.cos(theta)
        y = cy + s * np.sin(theta)
        pts.append([x, y])
    return np.array(pts)

def polygon_area_signed(pts):
    """Shoelace formula for signed area."""
    n = len(pts)
    area = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * area

def polygon_centroid(pts):
    area = polygon_area_signed(pts)
    if abs(area) < 1e-15:
        return np.mean(pts, axis=0)
    cx = cy = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    return np.array([cx, cy]) / (6.0 * area)

def point_in_polygon(p, poly):
    """Ray casting: is point p inside polygon poly (CCW)?"""
    x, y = p
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def polygons_intersect(poly1, poly2, eps=1e-9):
    """Check if two convex polygons intersect (including edge contact)."""
    # Check if any vertex of poly1 is inside poly2
    for v in poly1:
        if point_in_polygon(v, poly2):
            return True
    # Check if any vertex of poly2 is inside poly1
    for v in poly2:
        if point_in_polygon(v, poly1):
            return True
    # Check edge intersections
    n1, n2 = len(poly1), len(poly2)
    for i in range(n1):
        for j in range(n2):
            if segments_intersect(poly1[i], poly1[(i+1)%n1], poly2[j], poly2[(j+1)%n2]):
                return True
    return False

def compute_outer_side(inner_hexes, outer_center, outer_angle_deg):
    """
    Given inner hexagons (list of [cx, cy, angle_deg]), find minimum outer hexagon
    side length (centered at outer_center, rotated by outer_angle_deg) that contains all.
    
    Returns (side_length, valid)
    """
    cx, cy = outer_center
    # Vertices of outer hexagon with side s
    # We need to find minimum s such that all inner hexagon vertices are inside outer hexagon.
    # For a regular hexagon centered at (cx,cy) with side s and angle a, 
    # a point (x,y) is inside iff for all 6 face normals n_i, n_i . ((x,y)-(cx,cy)) <= s * apothem
    # where apothem = s * sqrt(3)/2
    
    # Collect all vertices of inner hexagons
    all_verts = []
    for hx, hy, ha in inner_hexes:
        verts = hexagon_vertices(hx, hy, 1.0, ha)
        all_verts.append(verts)
    
    all_verts = np.vstack(all_verts)
    
    # Outer hexagon face normals
    a = np.radians(outer_angle_deg)
    normals = []
    for i in range(6):
        theta = np.pi / 3 * i + a + np.pi / 2  # normal to each face
        normals.append([np.cos(theta), np.sin(theta)])
    normals = np.array(normals)
    
    # For each normal, compute max projection of inner vertices relative to center
    rel = all_verts - np.array([cx, cy])
    projections = rel @ normals.T  # shape (num_verts, 6)
    max_proj = np.max(projections, axis=0)  # max projection along each normal
    
    # Apothem must be >= max_proj for all normals
    apothem_needed = np.max(max_proj)
    if apothem_needed < 0:
        apothem_needed = 0.0
    
    side = 2.0 * apothem_needed / np.sqrt(3.0)
    return side, True

def check_validity(inner_hexes, outer_center, outer_side, outer_angle_deg, eps=1e-7):
    """Verify all inner hexagons fit inside outer and don't overlap each other."""
    # Check inner hexagons are inside outer
    outer_verts = hexagon_vertices(outer_center[0], outer_center[1], outer_side, outer_angle_deg)
    # Ensure CCW
    if polygon_area_signed(outer_verts) < 0:
        outer_verts = outer_verts[::-1]
    
    for hx, hy, ha in inner_hexes:
        verts = hexagon_vertices(hx, hy, 1.0, ha)
        for v in verts:
            if not point_in_polygon(v, outer_verts):
                return False
    
    # Check no overlaps between inner hexagons
    polys = []
    for hx, hy, ha in inner_hexes:
        verts = hexagon_vertices(hx, hy, 1.0, ha)
        if polygon_area_signed(verts) < 0:
            verts = verts[::-1]
        polys.append(verts)
    
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_intersect(polys[i], polys[j], eps):
                return False
    
    return True

def make_hex_lattice_config(radius_rings, start_angle_deg=0.0):
    """
    Create a hexagonal lattice arrangement of N=12 unit hexagons.
    Returns list of [cx, cy, angle_deg].
    """
    # Hexagonal lattice: center at origin, then rings of 6, 12, ...
    # For 12 hexagons: 1 center + 6 in first ring + 5 in second ring (partial)
    # Better: just use 12 positions from hex lattice
    
    positions = []
    # Center
    positions.append([0.0, 0.0])
    
    # First ring: 6 hexagons at distance 2 (since unit hexagons have inradius sqrt(3)/2,
    # and adjacent hexagons in a tiling are 2*inradius = sqrt(3) apart center-to-center)
    # Actually for unit side hexagons, the distance between centers of adjacent hexagons in a tiling is 2 * apothem = sqrt(3)
    # Wait: apothem of unit hexagon = sqrt(3)/2. Adjacent hexagons touch at edges, so center distance = 2 * apothem = sqrt(3).
    
    d = np.sqrt(3)  # distance between adjacent hex centers
    
    # First ring (6 hexagons)
    for i in range(6):
        theta = np.radians(60 * i + start_angle_deg)
        x = d * np.cos(theta)
        y = d * np.sin(theta)
        positions.append([x, y])
    
    # Second ring: need 5 more to reach 12
    # Second ring hexagons are at distance 2*d from center, at angles 30, 90, 150, 210, 270, 330 (offset by 30)
    # Or at distance d*sqrt(3) in axial coordinates... let's just place them at distance 2*d at 30-degree offsets
    d2 = 2 * d
    for i in range(5):
        theta = np.radians(60 * i + 30 + start_angle_deg)
        x = d2 * np.cos(theta)
        y = d2 * np.sin(theta)
        positions.append([x, y])
    
    # All hexagons have same orientation (0 degrees) for simplicity
    config = [[x, y, 0.0] for x, y in positions]
    return config

def local_search(config, outer_center, outer_angle_deg, max_iters=2000, step_size=0.05):
    """
    Local search to minimize outer side length.
    Perturb positions and angles of inner hexagons.
    """
    current_config = [list(c) for c in config]
    current_side, _ = compute_outer_side(current_config, outer_center, outer_angle_deg)
    
    if not check_validity(current_config, outer_center, current_side, outer_angle_deg):
        return current_config, current_side
    
    best_config = [list(c) for c in current_config]
    best_side = current_side
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 10:
            break
        
        # Pick a random hexagon to perturb
        idx = np.random.randint(len(current_config))
        old = list(current_config[idx])
        
        # Perturb position and angle
        current_config[idx][0] += np.random.uniform(-step_size, step_size)
        current_config[idx][1] += np.random.uniform(-step_size, step_size)
        current_config[idx][2] += np.random.uniform(-15, 15)
        
        new_side, _ = compute_outer_side(current_config, outer_center, outer_angle_deg)
        
        if new_side < best_side and check_validity(current_config, outer_center, new_side, outer_angle_deg):
            best_side = new_side
            best_config = [list(c) for c in current_config]
        else:
            current_config[idx] = old
    
    return best_config, best_side

def construct():
    """
    Construct 12 unit hexagons inside a regular hexagon, minimizing outer side length.
    Returns (inner, center, side, angle).
    """
    t_start = time.time()
    
    # Start with hexagonal lattice configuration
    config = make_hex_lattice_config(0)
    
    # Initial outer hexagon centered at origin, angle 0
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    side, _ = compute_outer_side(config, outer_center, outer_angle)
    
    if not check_validity(config, outer_center, side, outer_angle):
        # Fallback: place them in a line with large outer hexagon
        config = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        side = 40.0
        return config, outer_center, side, outer_angle
    
    best_config = [list(c) for c in config]
    best_side = side
    
    # Local search to improve
    best_config, best_side = local_search(config, outer_center, outer_angle, max_iters=3000, step_size=0.08)
    
    # Try a few different initial angles for the lattice
    for angle_offset in [30, 60, 15, 45]:
        if time.time() > DEADLINE - 20:
            break
        config2 = make_hex_lattice_config(angle_offset)
        side2, _ = compute_outer_side(config2, outer_center, outer_angle)
        if side2 < best_side and check_validity(config2, outer_center, side2, outer_angle):
            best_config, best_side = local_search(config2, outer_center, outer_angle, max_iters=2000, step_size=0.08)
    
    # Final validity check
    if not check_validity(best_config, outer_center, best_side, outer_angle):
        # Return the safe fallback
        config = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return config, outer_center, 40.0, outer_angle
    
    return best_config, outer_center, best_side, outer_angle
