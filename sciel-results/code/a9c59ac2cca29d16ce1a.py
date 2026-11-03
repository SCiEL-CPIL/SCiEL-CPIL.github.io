"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

N = 12

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3.0
        verts.append([cx + side * np.cos(theta), cy + side * np.sin(theta)])
    return np.array(verts)

def polygons_overlap_sat(poly1, poly2):
    """Check if two convex polygons overlap using SAT. Returns True if they overlap."""
    polys = [poly1, poly2]
    for poly in polys:
        n = len(poly)
        for i in range(n):
            # Edge from i to (i+1)%n
            edge = poly[(i + 1) % n] - poly[i]
            # Normal to edge
            normal = np.array([-edge[1], edge[0]])
            # Project both polygons onto this normal
            proj1 = poly1 @ normal
            proj2 = poly2 @ normal
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            # Check for gap
            if max1 < min2 or max2 < min1:
                return False  # Separating axis found, no overlap
    return True  # No separating axis, polygons overlap

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside or on boundary of hexagon."""
    verts = hex_vertices(cx, cy, side, angle_deg)
    # Use winding number or ray casting
    inside = False
    n = len(verts)
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def hexagon_inside_hexagon(inner_verts, outer_verts):
    """Check if inner hexagon is completely inside outer hexagon."""
    # All vertices of inner must be inside or on outer
    for v in inner_verts:
        if not point_in_hexagon(v[0], v[1], *outer_center_side_angle(outer_verts)):
            return False
    return True

def outer_center_side_angle(outer_verts):
    """Extract center, side, angle from outer hexagon vertices."""
    cx = np.mean(outer_verts[:, 0])
    cy = np.mean(outer_verts[:, 1])
    # Side length is distance between adjacent vertices
    side = np.linalg.norm(outer_verts[1] - outer_verts[0])
    # Angle: direction from center to first vertex
    angle = np.degrees(np.arctan2(outer_verts[0, 1] - cy, outer_verts[0, 0] - cx))
    return cx, cy, side, angle

def check_configuration(inner_positions, outer_side, outer_angle_deg=0.0, outer_center=(0.0, 0.0)):
    """Check if configuration is valid: all inner hexagons inside outer, no overlaps."""
    outer_verts = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle_deg)
    
    inner_verts_list = []
    for (ix, iy, iangle) in inner_positions:
        iv = hex_vertices(ix, iy, 1.0, iangle)
        # Check inside outer
        for v in iv:
            if not point_in_hexagon(v[0], v[1], outer_center[0], outer_center[1], outer_side, outer_angle_deg):
                return False
        inner_verts_list.append(iv)
    
    # Check pairwise non-overlap
    for i in range(len(inner_verts_list)):
        for j in range(i + 1, len(inner_verts_list)):
            if polygons_overlap_sat(inner_verts_list[i], inner_verts_list[j]):
                return False
    return True

def min_center_distance(a1, a2):
    """Minimum center distance for two unit hexagons with angles a1, a2 (degrees) to be non-overlapping."""
    # For aligned hexagons, min distance is 2.0 (edge to edge)
    # For other angles, it varies. We'll compute by checking overlap at various distances.
    # This is expensive; instead we use a conservative bound.
    # The maximum inradius is sqrt(3)/2, so a conservative min distance is 2*sqrt(3)/2 = sqrt(3) ~ 1.732
    # But for aligned, it's exactly 2.0. Let's use 2.0 as a safe lower bound for any orientation.
    # Actually, for some orientations the min distance can be less than 2.
    # Let's just use SAT in the actual check.
    return 2.0  # Conservative

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # Initial best: use a reasonable guess
    # For 12 hexagons in a hexagonal packing, a good arrangement is a hexagonal cluster.
    # Let's try a hexagonal lattice arrangement.
    
    # Hexagonal lattice positions for 12 unit hexagons:
    # Center + ring of 6 + ring of 9? No, 1+6=7, 1+6+12=19.
    # Let's try: 1 center, 6 around, 5 more in second ring? 
    # Actually, a better approach: use a 3x4 rectangular-ish grid with hexagonal offsets.
    
    # Let me try a simple hexagonal packing:
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 4 hexagons
    # With hexagonal spacing.
    
    # For hexagons with side 1, the distance between centers in a hexagonal lattice:
    # Horizontal: 2 * (3/2) = 3 (for flat-top) or 2 * sqrt(3)/2 * ... 
    # Let's use pointy-top hexagons.
    # For pointy-top, horizontal spacing = sqrt(3), vertical spacing = 1.5, with offset rows shifted by sqrt(3)/2.
    
    # Actually, let's just do a grid search over a bounded region.
    
    best_score = 40.0
    best_config = None
    
    # Strategy: Grid search over positions and angles, then bisection on outer side.
    # Given the time constraint, we need to be efficient.
    
    # Let's try a fixed arrangement and then optimize the outer side via bisection.
    # Fixed arrangement: hexagonal lattice
    
    # Generate hexagonal lattice positions
    positions = []
    # 3 rows of 4
    for row in range(3):
        for col in range(4):
            x = col * np.sqrt(3) + (row % 2) * np.sqrt(3) / 2
            y = row * 1.5
            positions.append([x, y, 0.0])
    
    # Center the configuration
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    centered_positions = [[p[0] - cx, p[1] - cy, p[2]] for p in positions]
    
    # Bisection on outer side length
    lo = 3.0
    hi = 20.0
    
    # Check if hi is feasible
    if not check_configuration(centered_positions, hi):
        # Try larger
        hi = 40.0
        if not check_configuration(centered_positions, hi):
            # Return a trivial valid configuration
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0
    
    # Bisection
    for _ in range(30):
        if time.time() > deadline:
            break
        mid = (lo + hi) / 2
        if check_configuration(centered_positions, mid):
            hi = mid
        else:
            lo = mid
    
    best_score = hi
    best_config = centered_positions
    
    # Try some other arrangements to see if we can do better
    # Try rotating some hexagons
    
    # Try a few angle variations
    angle_options = [0.0, 30.0, 60.0, 15.0, 45.0]
    
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Randomize angles slightly
        test_positions = []
        for p in centered_positions:
            a = angle_options[trial % len(angle_options)]
            test_positions.append([p[0], p[1], a])
        
        # Bisection for this configuration
        lo2 = 3.0
        hi2 = best_score  # Use current best as upper bound
        
        if not check_configuration(test_positions, hi2):
            hi2 = best_score + 5.0
            if not check_configuration(test_positions, hi2):
                continue
        
        for _ in range(20):
            if time.time() > deadline:
                break
            mid2 = (lo2 + hi2) / 2
            if check_configuration(test_positions, mid2):
                hi2 = mid2
            else:
                lo2 = mid2
        
        if hi2 < best_score:
            best_score = hi2
            best_config = test_positions
    
    # Try offsetting the entire configuration
    for dx in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        if time.time() > deadline:
            break
        for dy in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            if time.time() > deadline:
                break
            
            test_positions = [[p[0] + dx, p[1] + dy, p[2]] for p in best_config]
            
            lo3 = 3.0
            hi3 = best_score
            
            if not check_configuration(test_positions, hi3):
                hi3 = best_score + 5.0
                if not check_configuration(test_positions, hi3):
                    continue
            
            for _ in range(15):
                if time.time() > deadline:
                    break
                mid3 = (lo3 + hi3) / 2
                if check_configuration(test_positions, mid3):
                    hi3 = mid3
                else:
                    lo3 = mid3
            
            if hi3 < best_score:
                best_score = hi3
                best_config = test_positions
    
    return best_config, (0.0, 0.0), best_score, 0.0

# EVOLVE-BLOCK-END
