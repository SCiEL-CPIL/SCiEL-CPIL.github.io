"""Improved packing of 11 unit regular hexagons in a regular hexagon."""
import time
import math
import random
import itertools
import numpy as np

N = 11
T0 = time.time()
DEADLINE = 110.0  # seconds, safety margin before 120s limit

def hex_vertices(center, side, angle_deg):
    """Return 6 vertices of a regular hexagon.
    Vertex k at angle angle_deg + 60k degrees.
    This matches the 'pointy-top at angle' convention."""
    cx, cy = center
    a = math.radians(angle_deg)
    r = side  # circumradius = side for regular hexagon
    pts = []
    for k in range(6):
        th = a + math.radians(60.0) * k
        pts.append((cx + r * math.cos(th), cy + r * math.sin(th)))
    return pts

def convex_hull(points):
    """Andrew's monotone chain. Returns list of (x,y) in CCW order."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def is_convex_hull_ccw(hull):
    """Check if hull is CCW (positive area)."""
    n = len(hull)
    if n < 3:
        return False
    area = 0.0
    for i in range(n):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area > 0

def point_in_polygon(pt, poly):
    """Ray casting. poly is CCW. Returns True if pt is inside or on boundary."""
    x, y = pt
    n = len(poly)
    inside = False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xint = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xint:
                inside = not inside
    return inside

def polygon_containment(inner_pts, outer_poly, tol=1e-9):
    """Check if all inner_pts are inside or on outer_poly.
    inner_pts: list of (x,y). outer_poly: list of (x,y) CCW."""
    for p in inner_pts:
        if not point_in_polygon(p, outer_poly):
            # Check if on edge
            on_edge = False
            n = len(outer_poly)
            for i in range(n):
                x1, y1 = outer_poly[i]
                x2, y2 = outer_poly[(i + 1) % n]
                # Check if p is on segment (x1,y1)-(x2,y2)
                dx = x2 - x1
                dy = y2 - y1
                if abs(dx) > 1e-12 or abs(dy) > 1e-12:
                    t = ((p[0] - x1) * dx + (p[1] - y1) * dy) / (dx * dx + dy * dy)
                    if -tol <= t <= 1.0 + tol:
                        px = x1 + t * dx
                        py = y1 + t * dy
                        if (p[0] - px) ** 2 + (p[1] - py) ** 2 <= 1e-6:
                            on_edge = True
                            break
            if not on_edge:
                return False
    return True

def hexagon_containment(inner_center, inner_angle, outer_center, outer_side, outer_angle, tol=1e-9):
    """Check if unit hexagon (side 1) is inside outer hexagon."""
    inner_pts = hex_vertices(inner_center, 1.0, inner_angle)
    outer_pts = hex_vertices(outer_center, outer_side, outer_angle)
    return polygon_containment(inner_pts, outer_pts, tol)

def pair_disjoint(c1, a1, c2, a2, tol=1e-9):
    """Check if two unit hexagons are disjoint (no overlapping interior).
    They can touch at edges/vertices."""
    # Simple check: center distance >= 2 * inradius * 2? No, that's not right for arbitrary rotation.
    # Better: check if any vertex of one is inside the other, or if edges intersect.
    # For simplicity and speed, use center distance heuristic first, then vertex check.
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    dist = math.sqrt(dx * dx + dy * dy)
    # Minimum distance between centers of two disjoint unit hexagons is at least 2 * apothem = sqrt(3)
    # But with rotation, it can be as low as... well, if they share an edge, center distance is 2 * apothem = sqrt(3)
    # Actually for two unit hexagons sharing an edge, center distance = sqrt(3)
    # The minimum possible center distance for disjoint hexagons is sqrt(3) (sharing edge)
    if dist > 2.0:  # Clearly disjoint
        return True
    if dist < sqrt3 - 0.01:  # Clearly overlapping (sqrt3 ≈ 1.732)
        # But with rotation, minimum can be slightly less? No, inradius is sqrt(3)/2, so 2*inradius = sqrt(3)
        # If dist < sqrt(3) - epsilon, they must overlap
        # Actually this is not strictly true for all rotations. Let's do vertex check.
        pass
    
    # Check vertices of hex1 inside hex2, and vice versa
    pts1 = hex_vertices(c1, 1.0, a1)
    pts2 = hex_vertices(c2, 1.0, a2)
    outer1 = pts1  # use as polygon
    outer2 = pts2
    
    # Check if any vertex of hex1 is strictly inside hex2 (not on boundary)
    for p in pts1:
        if point_in_polygon(p, outer2):
            return False
    for p in pts2:
        if point_in_polygon(p, outer1):
            return False
    
    # Check edge intersections
    def segments_intersect(p1, p2, p3, p4, tol=1e-9):
        def orient(a, b, c):
            val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if abs(val) < tol:
                return 0
            return 1 if val > 0 else -1
        o1 = orient(p1, p2, p3)
        o2 = orient(p1, p2, p4)
        o3 = orient(p3, p4, p1)
        o4 = orient(p3, p4, p2)
        if o1 != o2 and o3 != o4:
            return True
        if o1 == 0 and ((min(p1[0], p2[0]) - tol <= p3[0] <= max(p1[0], p2[0]) + tol) and 
                         (min(p1[1], p2[1]) - tol <= p3[1] <= max(p1[1], p2[1]) + tol)):
            return True
        if o2 == 0 and ((min(p1[0], p2[0]) - tol <= p4[0] <= max(p1[0], p2[0]) + tol) and 
                         (min(p1[1], p2[1]) - tol <= p4[1] <= max(p1[1], p2[1]) + tol)):
            return True
        if o3 == 0 and ((min(p3[0], p4[0]) - tol <= p1[0] <= max(p3[0], p4[0]) + tol) and 
                         (min(p3[1], p4[1]) - tol <= p1[1] <= max(p3[1], p4[1]) + tol)):
            return True
        if o4 == 0 and ((min(p3[0], p4[0]) - tol <= p2[0] <= max(p3[0], p4[0]) + tol) and 
                         (min(p3[1], p4[1]) - tol <= p2[1] <= max(p3[1], p4[1]) + tol)):
            return True
        return False
    
    for i in range(6):
        for j in range(6):
            if segments_intersect(pts1[i], pts1[(i+1)%6], pts2[j], pts2[(j+1)%6]):
                return False
    return True

sqrt3 = math.sqrt(3.0)

def generate_flower_config():
    """Generate a flower-like configuration of 11 hexagons.
    Center + 6 around it + 4 more in a ring pattern.
    Returns list of (x, y, angle_deg) for 11 hexagons, all angle 0."""
    hex_dist = sqrt3  # distance between centers of adjacent hexagons
    configs = []
    
    # Flower: center at (0,0), 6 at distance sqrt3 at 0,60,...,300
    flower = [(0.0, 0.0, 0.0)]
    for k in range(6):
        a = math.radians(60.0 * k)
        flower.append((sqrt3 * math.cos(a), sqrt3 * math.sin(a), 0.0))
    # Add 4 more: at 120-degree positions at distance 2*sqrt3? Or at distance sqrt3*sqrt(3) in diagonal directions
    # Actually, let's add them at positions that form a larger ring
    # For 11, a common pattern is 1 + 6 + 4, where the 4 are at the "gaps"
    # Let's put 4 more at angles 30, 90, 150, 210, 270, 330 minus 2, at distance sqrt3
    # No, that would be 1 + 6 + 6 = 13.
    # For 11: 1 center, 6 in first ring, 4 in second ring at specific positions
    # Let's just use a simple extended pattern
    for k in range(4):
        a = math.radians(60.0 * k + 30)  # 30, 90, 150, 210
        # Place at distance sqrt3 * sqrt(3) = 3 in these directions?
        # Actually, the "second ring" in hex grid is at distance 2*sqrt3 in the 6 axial directions,
        # and sqrt3*sqrt(3) = 3 in the 6 diagonal directions.
        # Let's place 4 at diagonal positions: 30, 90, 150, 270 (skipping some)
        d = sqrt3 * sqrt3  # = 3
        configs.append((d * math.cos(a), d * math.sin(a), 0.0))
    
    all_pts = flower + configs
    return all_pts

def generate_row_config():
    """Generate a row-based configuration: 4-3-4 rows."""
    hex_dist = sqrt3
    v_dist = 1.5  # vertical distance between rows
    h_dist = sqrt3  # horizontal distance between hexagons in same row
    
    pts = []
    # Row 0 (top): 4 hexagons
    for k in range(4):
        x = (k - 1.5) * h_dist
        y = 0.0
        pts.append((x, y, 0.0))
    # Row 1 (middle): 3 hexagons, offset by h_dist/2
    for k in range(3):
        x = (k - 1.0) * h_dist
        y = v_dist
        pts.append((x, y, 0.0))
    # Row 2 (bottom): 4 hexagons
    for k in range(4):
        x = (k - 1.5) * h_dist
        y = 2 * v_dist
        pts.append((x, y, 0.0))
    return pts

def generate_353_config():
    """3-5-3 row configuration."""
    hex_dist = sqrt3
    v_dist = 1.5
    h_dist = sqrt3
    
    pts = []
    # Row 0: 3 hexagons
    for k in range(3):
        x = (k - 1.0) * h_dist
        y = 0.0
        pts.append((x, y, 0.0))
    # Row 1: 5 hexagons, offset
    for k in range(5):
        x = (k - 2.0) * h_dist
        y = v_dist
        pts.append((x, y, 0.0))
    # Row 2: 3 hexagons
    for k in range(3):
        x = (k - 1.0) * h_dist
        y = 2 * v_dist
        pts.append((x, y, 0.0))
    return pts

def compute_min_outer_side(inner_pts, outer_angle=0.0, outer_center=(0.0, 0.0)):
    """Compute minimum side length of outer hexagon (at given angle and center) 
    that contains all inner hexagons (each side 1, at given angles).
    Uses support function approach."""
    # Collect all vertices of all inner hexagons
    all_vertices = []
    for (cx, cy, angle) in inner_pts:
        verts = hex_vertices((cx, cy), 1.0, angle)
        all_vertices.extend(verts)
    
    if not all_vertices:
        return 0.0
    
    # For a regular hexagon with center C and angle a, the constraint for a point P to be inside
    # is: for each of 6 directions, the projection of (P - C) onto that direction <= side * cos(30°) = side * sqrt(3)/2
    # Direction k: angle = a + 60k, but the normal to the edge is at a + 60k + 30
    # Actually, for a regular hexagon with vertices at angles a, a+60, ..., the edges are perpendicular
    # to directions a+30, a+90, a+150, a+210, a+270, a+330.
    # The distance from center to each edge is side * cos(30°) = side * sqrt(3)/2.
    
    a_outer = math.radians(outer_angle)
    # Normal directions to the 6 edges
    normal_angles = [a_outer + math.radians(30 + 60 * k) for k in range(6)]
    
    # For each normal direction n_k, max projection of (P - C) onto n_k must be <= S * sqrt(3)/2
    # So S >= max_k [ max_P (P - C) . n_k ] / (sqrt(3)/2)
    
    cx, cy = outer_center
    max_proj = 0.0
    for na in normal_angles:
        nx = math.cos(na)
        ny = math.sin(na)
        for px, py in all_vertices:
            proj = (px - cx) * nx + (py - cy) * ny
            if proj > max_proj:
                max_proj = proj
    
    side = max_proj / (sqrt3 / 2.0)
    return side

def optimize_outer_for_config(inner_pts):
    """For a given inner configuration, find the best outer hexagon (center, side, angle).
    Try several angles and find minimum side."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # The outer hexagon should be centered at the centroid of inner points
    cx_avg = sum(p[0] for p in inner_pts) / len(inner_pts)
    cy_avg = sum(p[1] for p in inner_pts) / len(inner_pts)
    
    # Try different angles
    for angle in [0.0, 30.0]:
        # Try different centers around the centroid
        for dcx in [-0.5, -0.25, 0.0, 0.25, 0.5]:
            for dcy in [-0.5, -0.25, 0.0, 0.25, 0.5]:
                center = (cx_avg + dcx, cy_avg + dcy)
                side = compute_min_outer_side(inner_pts, angle, center)
                if side < best_side:
                    best_side = side
                    best_center = center
                    best_angle = angle
    
    return best_side, best_center, best_angle

def validate_configuration(inner_pts, outer_center, outer_side, outer_angle, tol=1e-8):
    """Validate that all inner hexagons are inside outer and pairwise disjoint."""
    oc = outer_center
    # Check containment
    for (cx, cy, angle) in inner_pts:
        if not hexagon_containment((cx, cy), angle, oc, outer_side, outer_angle, tol):
            return False
    
    # Check pairwise disjointness
    for i in range(len(inner_pts)):
        for j in range(i + 1, len(inner_pts)):
            c1 = (inner_pts[i][0], inner_pts[i][1])
            a1 = inner_pts[i][2]
            c2 = (inner_pts[j][0], inner_pts[j][1])
            a2 = inner_pts[j][2]
            if not pair_disjoint(c1, a1, c2, a2, tol):
                return False
    
    return True

def sa_optimize(initial_config, max_iters=5000, t_start=1.0, t_end=0.01):
    """Simulated annealing to optimize inner hexagon positions."""
    config = [list(p) for p in initial_config]
    best_config = [list(p) for p in config]
    best_score = float('inf')
    
    def score(cfg):
        side, center, angle = optimize_outer_for_config(cfg)
        # Add penalty if not valid
        if not validate_configuration(cfg, center, side, angle, tol=1e-6):
            return side + 10.0  # penalty
        return side
    
    current_score = score(config)
    best_score = current_score
    best_config = [list(p) for p in config]
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        # Pick random hexagon and perturb
        idx = random.randint(0, len(config) - 1)
        new_config = [list(p) for p in config]
        new_config[idx][0] += random.gauss(0, 0.3)
        new_config[idx][1] += random.gauss(0, 0.3)
        
        new_score = score(new_config)
        t = t_start * (t_end / t_start) ** (it / max_iters)
        
        if new_score < current_score or random.random() < math.exp(-(new_score - current_score) / max(t, 1e-10)):
            config = new_config
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_config = [list(p) for p in config]
    
    return best_config, best_score

def construct():
    global T0, DEADLINE
    start_time = time.time()
    DEADLINE = start_time + 110.0
    
    random.seed(42)
    
    # Generate multiple initial configurations
    initial_configs = [
        generate_flower_config(),
        generate_row_config(),
        generate_353_config(),
    ]
    
    # Add some random perturbations to create variants
    for _ in range(3):
        base = random.choice(initial_configs)
        perturbed = []
        for (x, y, a) in base:
            perturbed.append((x + random.gauss(0, 0.2), y + random.gauss(0, 0.2), a))
        initial_configs.append(perturbed)
    
    best_overall = None
    best_overall_score = float('inf')
    
    for config in initial_configs:
        if time.time() > DEADLINE:
            break
        
        # Optimize outer for this config
        side, center, angle = optimize_outer_for_config(config)
        
        if side < best_overall_score:
            best_overall_score = side
            best_overall = (config, center, angle)
        
        # Run SA for refinement
        if time.time() > DEADLINE:
            break
        optimized_config, sa_score = sa_optimize(config, max_iters=2000, t_start=0.5, t_end=0.01)
        
        if sa_score < best_overall_score:
            side, center, angle = optimize_outer_for_config(optimized_config)
            if side < best_overall_score:
                best_overall_score = side
                best_overall = (optimized_config, center, angle)
    
    # Final high-precision polish
    if best_overall is not None and time.time() < DEADLINE:
        config, center, angle = best_overall
        # Nelder-Mead style local search
        for _ in range(50):
            if time.time() > DEADLINE:
                break
            improved = False
            for i in range(len(config)):
                for dx in [-0.1, -0.05, 0.05, 0.1]:
                    for dy in [-0.1, -0.05, 0.05, 0.1]:
                        new_config = [list(p) for p in config]
                        new_config[i][0] += dx
                        new_config[i][1] += dy
                        side, c, a = optimize_outer_for_config(new_config)
                        if side < best_overall_score - 1e-9:
                            config = new_config
                            center, angle = c, a
                            best_overall_score = side
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    
    if best_overall is None:
        # Fallback
        config = generate_flower_config()
        side, center, angle = optimize_outer_for_config(config)
        best_overall = (config, center, angle)
        best_overall_score = side
    
    config, center, angle = best_overall
    
    # Final validation and adjustment
    # If not valid, try to fix
    if not validate_configuration(config, center, angle, best_overall_score, tol=1e-6):
        # Increase side slightly
        config, center, angle = best_overall
        best_overall_score += 0.1
    
    # Ensure the side is computed correctly
    side = compute_min_outer_side(config, angle, center)
    # Add small margin for safety
    side += 0.01
    
    return [[x, y, a] for (x, y, a) in config], center, side, angle
