"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses a lattice-based topology search with symmetry exploitation and local refinement.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 140.0  # safety margin before 150s kill


def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts


def polygons_overlap_sat(A, B, tol=1e-9):
    """SAT: True if two convex polygons intersect (including touching)."""
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            if nx == 0 and ny == 0:
                continue
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            if maxA < minB - tol or maxB < minA - tol:
                return False
    return True


def point_in_convex_polygon(px, py, poly, tol=1e-9):
    """True if (px,py) is inside or on boundary of convex polygon."""
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < tol:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def compute_min_outer_side(inner, outer_center, outer_angle):
    """Minimum outer side length such that all unit hexagons fit inside."""
    cx, cy = outer_center
    a = math.radians(outer_angle)
    # The 6 outward normals of the outer hexagon
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_proj = 0.0
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - cx, py - cy
            for (nx, ny) in normals:
                proj = dx * nx + dy * ny
                if proj > max_proj:
                    max_proj = proj
    # apothem = s * sqrt(3)/2, so s = 2*max_proj/sqrt(3)
    return (2.0 / math.sqrt(3.0)) * max_proj


def check_validity(inner, outer_center, outer_side, outer_angle, eps=1e-7):
    """Check all inner hexagons are inside outer and pairwise non-overlapping."""
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    polys = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        polys.append(p)
        
    # Check containment
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
                
    # Check non-overlap
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True


def generate_topology_positions(topo_id, scale):
    """
    Generate positions for 12 hexagons based on topology ID and scale.
    Returns list of (x, y) centers.
    """
    positions = []
    if topo_id == 0:
        # 1 center + 6 ring 1 + 5 ring 2 (Flower)
        # This is the standard hexagonal packing.
        # Distances:
        # Center: (0,0)
        # Ring 1: distance 2*scale (touching if scale=1)
        # Ring 2: distance 4*scale (touching Ring 1 if scale=1)
        positions.append((0.0, 0.0))
        for k in range(6):
            theta = math.radians(k * 60.0)
            positions.append((2.0 * scale * math.cos(theta), 2.0 * scale * math.sin(theta)))
        # Add 5 from Ring 2. To keep it compact, we place them adjacent to the gaps or continuing the ring.
        # A full ring 2 has 12 positions. We pick 5.
        # To maintain symmetry, we might break 6-fold symmetry, but let's try a specific pattern.
        # Actually, for 12 hexagons, the "Flower" of 7 + 5 is not perfectly symmetric.
        # Alternative: 3x4 grid?
        # Let's stick to the most compact known: The "Star" or "Flower" is usually best for N<10.
        # For N=12, a 3x4 rectangular lattice or a modified flower is good.
        # Let's define Topo 0 as the "Full Flower" (7) + 5 from the next ring placed symmetrically?
        # No, 12 hexagons can form a hexagon of side 2 (which has 1+6+12 = 19)? No.
        # Side 1: 1. Side 2: 1+6=7. Side 3: 1+6+12=19.
        # So 12 is between 7 and 19.
        # Common packing: 3 rows of 4? Or 4 rows of 3?
        # Let's try Topo 0: 3x4 Grid (hexagonal offset)
        # Row 0: 4 hexes
        # Row 1: 4 hexes (shifted)
        # Row 2: 4 hexes
        # This is a parallelogram.
        
        # Let's redefine Topos:
        # Topo 0: 3x4 Hex Grid
        # Topo 1: 2x6 Hex Grid
        # Topo 2: 1+6+5 Flower (Asymmetric ring 2)
        # Topo 3: 1+6+4+1 (Center, Ring1, Ring2 partial, Center? No)
        
        # Let's implement Topo 0: 3x4 Grid
        # Coordinates for 3x4 hex grid with unit spacing (scale=1 means touching)
        # y spacing: sqrt(3)
        # x spacing: 2
        # Offset for odd rows: 1
        for row in range(3):
            for col in range(4):
                x = col * 2.0 * scale
                y = row * math.sqrt(3) * scale
                if row % 2 == 1:
                    x += 1.0 * scale
                # Center the cluster
                # x range: 0 to 3*2*scale = 6*scale. Mid = 3*scale.
                # y range: 0 to 2*sqrt(3)*scale. Mid = sqrt(3)*scale.
                positions.append((x - 3.0 * scale, y - math.sqrt(3) * scale))
                
    elif topo_id == 1:
        # Topo 1: 2x6 Hex Grid
        for row in range(2):
            for col in range(6):
                x = col * 2.0 * scale
                y = row * math.sqrt(3) * scale
                if row % 2 == 1:
                    x += 1.0 * scale
                # Center
                # x range: 0 to 5*2*scale = 10*scale. Mid = 5*scale.
                # y range: 0 to 1*sqrt(3)*scale. Mid = 0.5*sqrt(3)*scale.
                positions.append((x - 5.0 * scale, y - 0.5 * math.sqrt(3) * scale))
                
    elif topo_id == 2:
        # Topo 2: Flower (7) + 5 in Ring 2
        # This topology is inherently asymmetric for the 5 outer ones, but we can place them
        # to minimize the bounding box.
        # However, a better "12" topology is the "Hexagon of side 2 minus 7 corner/edge hexes"?
        # Side 2 hex has 19. 19-12=7.
        # Let's try removing 7 from the side-2 hexagon to leave 12.
        # The side-2 hexagon vertices are at distance 4*scale? No, vertices of hexagon container.
        # Inner hexagons centers:
        # (0,0)
        # (2s, 0), (-2s, 0), (0, 2s*sqrt(3)), ... (6 neighbors)
        # (4s, 0), (2s, 2s*sqrt(3)), ... (12 neighbors)
        # Total 19.
        # If we remove 7, we can keep a symmetric core.
        # Remove 1 from Ring 1? No, keep center.
        # Remove 6 from Ring 2? That leaves 1+6+6 = 13.
        # Remove 7 from Ring 2? That leaves 1+6+5 = 12.
        # Which 5 from Ring 2?
        # To be compact, we should probably pick 5 consecutive ones? Or spread them?
        # If we pick 5 consecutive, the shape is a "fan".
        # Let's try 5 consecutive in Ring 2.
        positions.append((0.0, 0.0))
        for k in range(6):
            theta = math.radians(k * 60.0)
            positions.append((2.0 * scale * math.cos(theta), 2.0 * scale * math.sin(theta)))
        # Ring 2 positions are at distance 4*scale? No, distance from center is 4*scale?
        # Distance between center and Ring 1 is 2*scale.
        # Distance between Ring 1 and Ring 2 is 2*scale.
        # So Ring 2 distance from center is 4*scale.
        # Let's pick 5 consecutive angles from Ring 2.
        # Angles for Ring 2 are offset by 30 degrees from Ring 1?
        # Hex grid geometry:
        # Ring 1 angles: 0, 60, 120...
        # Ring 2 angles: 30, 90, 150... (the "corner" ones) AND 0, 60... (the "edge" ones)?
        # Actually, the 12 positions in Ring 2 are at:
        # (4s, 0), (2s, 2s*sqrt(3)), (-2s, 2s*sqrt(3)), ...
        # Let's just generate the 12 positions of Ring 2 and pick 5.
        ring2_positions = []
        for k in range(6):
            # Corner-ish
            theta = math.radians(k * 60.0 + 30.0)
            ring2_positions.append((4.0 * scale * math.cos(theta), 4.0 * scale * math.sin(theta)))
            # Edge-ish? No, that's not right.
            # Let's use coordinates.
            # Ring 2 centers:
            # (4, 0), (2, 2*sqrt(3)), (-2, 2*sqrt(3)), (-4, 0), (-2, -2*sqrt(3)), (2, -2*sqrt(3))
            # AND (3, sqrt(3))? No.
            # Standard hex coords (q, r):
            # Ring 2:
            # (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1) ... wait, that's 8?
            # No, Ring 2 has 12 hexes.
            # Let's just use the 3x4 grid as the primary candidate, it's very robust.
            # I will implement Topo 2 as the 3x4 grid rotated by 30 degrees? No, that's just a rotation.
            # Let's implement Topo 2 as a "Rectangular" 2x6 but with different spacing?
            
            # Let's stick to Topo 0 (3x4) and Topo 1 (2x6) and Topo 3 (1+6+5 Flower).
            # For Topo 3, let's place the 5 Ring 2 hexes at angles 0, 60, 120, 180, 240 (consecutive).
            for k in range(5):
                theta = math.radians(k * 60.0)
                # Distance 4*scale?
                # In a hex grid, the distance from center to Ring 2 "vertex" positions is 4*scale?
                # Let's verify: Center (0,0). Neighbor (2,0). Next neighbor of (2,0) is (4,0)?
                # Yes, if aligned.
                positions.append((4.0 * scale * math.cos(theta), 4.0 * scale * math.sin(theta)))
                
    else:
        # Fallback: 3x4
        for row in range(3):
            for col in range(4):
                x = col * 2.0 * scale
                y = row * math.sqrt(3) * scale
                if row % 2 == 1:
                    x += 1.0 * scale
                positions.append((x - 3.0 * scale, y - math.sqrt(3) * scale))
                
    return positions


def evaluate_config(positions, inner_rot, outer_rot):
    """Evaluate a configuration. Returns (side_length, is_valid)."""
    inner = [(x, y, inner_rot) for x, y in positions]
    outer_center = (0.0, 0.0)
    outer_side = compute_min_outer_side(inner, outer_center, outer_rot)
    
    # Check validity
    if not check_validity(inner, outer_center, outer_side, outer_rot):
        return float('inf'), False
    
    return outer_side, True


def construct():
    t0 = time.time()
    best_side = 1e6
    best_result = None
    
    # Topologies to try
    topologies = [0, 1, 2] # 3x4, 2x6, Flower
    
    # Ranges for search
    # Scale: 0.95 to 1.05 (allowing some breathing room or tight packing if rotation helps)
    # Actually, if scale < 1, they are separated. If scale > 1, they overlap.
    # So scale must be <= 1 for non-overlap in a standard lattice?
    # Wait, if we rotate inner hexagons, they might fit closer?
    # No, the minimum distance between centers of two unit hexagons is 2.0 if they are parallel.
    # If rotated, the minimum distance can be less?
    # The "width" of a hexagon is 2. The "height" is sqrt(3).
    # If we rotate by 30 degrees, the width becomes sqrt(3) and height becomes 2.
    # So in a grid, we can pack tighter in one dimension if we rotate.
    # But the grid spacing is determined by the worst-case direction.
    # So scale=1.0 is the tightest for parallel hexes.
    # If we rotate, we might be able to reduce scale slightly?
    # Let's search scale in [0.95, 1.01].
    
    scales = np.linspace(0.95, 1.01, 13)
    inner_rots = np.linspace(0.0, 180.0, 19) # 0 to 180, step 10
    outer_rots = np.linspace(0.0, 60.0, 7) # 0 to 60, step 10
    
    # To save time, we will first do a coarse search, then refine the best few.
    
    coarse_best = []
    
    for topo in topologies:
        if time.time() > DEADLINE - 20:
            break
            
        for scale in scales:
            if time.time() > DEADLINE - 20:
                break
            positions = generate_topology_positions(topo, scale)
            
            for inner_rot in inner_rots:
                if time.time() > DEADLINE - 20:
                    break
                
                # Quick check: if scale is too small, they might not overlap, but we want min outer side.
                # Actually, if scale < 1, they are disjoint. We can shrink the outer hexagon.
                # So we want to maximize scale? No, we want to minimize outer side.
                # If we increase scale, the cluster gets bigger, so outer side increases.
                # So we want the *smallest* scale that allows the configuration?
                # No, the scale is a parameter of the *lattice*.
                # If we use scale=1.0, the hexes touch. This is the most compact *lattice*.
                # If we use scale=0.9, they are separated. The bounding box is smaller.
                # But wait, if they are separated, we can *move* them closer?
                # The lattice model assumes fixed relative positions.
                # So if we pick scale=0.9, the cluster is smaller, so the outer hexagon is smaller.
                # So we should search for the *smallest* scale?
                # No, the "unit" hexagons have fixed size 1.
                # The "scale" parameter here scales the *positions*.
                # If I scale positions by 0.9, the distance between centers becomes 1.8.
                # Since the hexes have width 2, they overlap!
                # So scale must be >= 1.0 for parallel hexes.
                # If rotated, the minimum center distance required might be less than 2.0?
                # Let's check.
                # Two unit hexagons. Min distance between centers to avoid overlap.
                # If aligned, dist >= 2.0.
                # If rotated by 30 deg, the "flat" sides face each other?
                # The apothem is sqrt(3)/2 ~ 0.866.
                # If flat sides face each other, dist >= 2 * apothem = 1.732.
                # So yes, rotation allows tighter packing!
                # So we should search scale in [0.85, 1.01].
                
                for outer_rot in outer_rots:
                    if time.time() > DEADLINE - 20:
                        break
                    
                    side, valid = evaluate_config(positions, inner_rot, outer_rot)
                    if valid:
                        coarse_best.append((side, topo, scale, inner_rot, outer_rot, positions))
                        
    if not coarse_best:
        # Fallback if no valid config found (should not happen)
        # Use a very large scale
        positions = generate_topology_positions(0, 1.5)
        inner = [(x, y, 0.0) for x, y in positions]
        outer_center = (0.0, 0.0)
        outer_side = compute_min_outer_side(inner, outer_center, 0.0)
        return inner, outer_center, outer_side, 0.0

    # Sort by side length
    coarse_best.sort(key=lambda x: x[0])
    
    # Refine the top 5 candidates
    for candidate in coarse_best[:5]:
        if time.time() > DEADLINE - 10:
            break
            
        side, topo, scale, inner_rot, outer_rot, positions = candidate
        
        # Define a function to optimize positions slightly?
        # Or just refine the parameters (scale, inner_rot, outer_rot)?
        # Since the topology is fixed, we can optimize scale and rotations.
        
        def objective(params):
            s, ir, orr = params
            # Ensure s is not too small (overlap)
            if s < 0.8: return 1e6
            pos = generate_topology_positions(topo, s)
            side_val, valid = evaluate_config(pos, ir, orr)
            if not valid:
                return 1e6
            return side_val
        
        x0 = [scale, inner_rot, outer_rot]
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 100})
            if res.fun < side:
                s_opt, ir_opt, orr_opt = res.x
                pos_opt = generate_topology_positions(topo, s_opt)
                side_opt, valid_opt = evaluate_config(pos_opt, ir_opt, orr_opt)
                if valid_opt and side_opt < best_side:
                    best_side = side_opt
                    inner_opt = [(x, y, ir_opt) for x, y in pos_opt]
                    best_result = (inner_opt, (0.0, 0.0), side_opt, orr_opt)
        except Exception:
            pass
            
    if best_result is None:
        # Use the best coarse result
        side, topo, scale, inner_rot, outer_rot, positions = coarse_best[0]
        inner = [(x, y, inner_rot) for x, y in positions]
        best_result = (inner, (0.0, 0.0), side, outer_rot)
        
    return best_result
