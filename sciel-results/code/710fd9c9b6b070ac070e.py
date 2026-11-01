"""Packing 11 unit hexagons in a minimal regular hexagon."""
import numpy as np
from itertools import product
import time

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy)."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def _min_enclosing_hex_side(points, angle_deg):
    """Given points and outer hexagon rotation, return the minimum side length
    of a regular hexagon (at that rotation) that encloses all points.
    The hexagon is centered to minimize its size (centered at midpoint of projections)."""
    a = np.radians(angle_deg)
    # 6 normal directions of the hexagon sides
    normals = np.column_stack([np.cos(a + np.arange(6) * np.pi / 3),
                               np.sin(a + np.arange(6) * np.pi / 3)])
    # Project points onto each normal
    proj = points @ normals.T  # (n, 6)
    widths = proj.max(axis=0) - proj.min(axis=0)  # (6,)
    # For a regular hexagon with side s and rotation a, the width in normal direction i
    # is s * sqrt(3) (the distance between opposite sides)
    # Actually: a regular hexagon with side s has inradius = s*sqrt(3)/2,
    # so distance between opposite sides = s*sqrt(3).
    # The width in direction n_i (normal to side i) equals s*sqrt(3) only if the
    # hexagon is aligned. For a regular hexagon, the support function in direction
    # making angle theta with a normal is: h(theta) = s*sqrt(3)/2 * 1/cos(theta mod 60° - 30°)
    # Wait, let me think more carefully.
    # 
    # A regular hexagon with side s, centered at origin, with one vertex at angle 0:
    # vertices at angles 0, 60, 120, 180, 240, 300 degrees, radius = s.
    # The distance from center to each side (inradius) = s * cos(30°) = s * sqrt(3)/2.
    # The width in the direction of a side normal is 2 * inradius = s * sqrt(3).
    # In other directions, the width varies.
    #
    # For a regular hexagon, the support function h(u) where u is a unit vector:
    # h(u) = inradius / cos(delta) where delta is the angle from u to the nearest
    # side normal, capped at 30 degrees.
    # So width in direction u = 2 * h(u) = 2 * inradius / cos(delta) = s*sqrt(3) / cos(delta)
    #
    # For our 6 normals (the side normals), delta = 0, so width = s*sqrt(3).
    # But we need the hexagon to contain all points, so for each of the 6 normal
    # directions, the projection width of the points must be <= s*sqrt(3).
    # Therefore: s >= max(widths) / sqrt(3)
    
    s = np.max(widths) / np.sqrt(3)
    return s

def _best_enclosing_hex_side(points):
    """Find the minimum side length over all rotations of the enclosing hexagon."""
    # The optimal rotation will align one of the hexagon's side normals with
    # a direction that minimizes the maximum projection width.
    # By symmetry, we only need to check angles in [0, 30) degrees.
    # Let's do a fine grid search.
    best_s = float('inf')
    best_angle = 0.0
    for angle in np.linspace(0, 30, 301):
        s = _min_enclosing_hex_side(points, angle)
        if s < best_s:
            best_s = s
            best_angle = angle
    # Refine around best_angle
    for angle in np.linspace(best_angle - 1, best_angle + 1, 201):
        s = _min_enclosing_hex_side(points, angle)
        if s < best_s:
            best_s = s
            best_angle = angle
    return best_s, best_angle

def _compute_score(inner_positions):
    """Given inner hex center positions (all at 0 degrees), compute the
    minimum enclosing hexagon side length and its rotation/center."""
    points = np.array(inner_positions, dtype=float)
    # The enclosing hexagon must contain the entire hexagons, not just centers.
    # A unit hexagon centered at p with orientation 0 has vertices at p + (cos(k*60°), sin(k*60°))
    # So the set of all points in all inner hexagons is the union of their vertices
    # (since hexagons are convex, the extreme points are the vertices).
    
    all_vertices = []
    for (cx, cy, _) in inner_positions:
        verts = _hex_vertices(cx, cy, 1.0, 0.0)
        all_vertices.append(verts)
    all_verts = np.vstack(all_vertices)
    
    # Now find minimum enclosing regular hexagon
    best_s = float('inf')
    best_angle = 0.0
    for angle in np.linspace(0, 30, 301):
        s = _min_enclosing_hex_side(all_verts, angle)
        if s < best_s:
            best_s = s
            best_angle = angle
    for angle in np.linspace(best_angle - 0.5, best_angle + 0.5, 101):
        s = _min_enclosing_hex_side(all_verts, angle)
        if s < best_s:
            best_s = s
            best_angle = angle
    
    # Find the center of the enclosing hexagon
    a = np.radians(best_angle)
    normals = np.column_stack([np.cos(a + np.arange(6) * np.pi / 3),
                               np.sin(a + np.arange(6) * np.pi / 3)])
    proj = all_verts @ normals.T
    centers = (proj.max(axis=0) + proj.min(axis=0)) / 2
    # Reconstruct center from projections
    # The center c satisfies: c . normals[i] = centers[i] for all i
    # Use least squares
    c = np.linalg.lstsq(normals.T, centers, rcond=None)[0]
    
    return best_s, (c[0], c[1]), best_angle

def _generate_topologies():
    """Generate candidate 11-cell polyhex topologies on a hex grid."""
    # Use axial coordinates: (q, r) where the 6 neighbors are:
    # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    neighbors = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    
    def hex_to_cart(q, r):
        # Convert axial to cartesian (for a hex grid with spacing 2 between centers)
        x = 2.0 * q + 1.0 * r
        y = np.sqrt(3) * r
        return x, y
    
    # Generate connected 11-cell shapes via BFS/DFS
    # This is complex; instead, let's use known compact shapes.
    # The most compact 11-hexagon shapes:
    # 1. A "flower" with center + 6 + 4 more
    # 2. A 3x3 block minus 1 corner = 8, plus 3 more
    # 3. Various arrangements
    
    # Let's just enumerate all connected subsets of a larger grid that have 11 cells
    # and are "compact" (small bounding box).
    
    # Simpler: try a few hand-crafted compact topologies
    topologies = []
    
    # Topology 1: center + 6 ring + 4 extra (attached to ring)
    # Center at (0,0), ring at 6 neighbors, then 4 more attached
    # This gives a shape like a plus with extra cells
    cells = [(0,0)]
    cells += neighbors  # 7 cells
    # Add 4 more: attach to some ring cells
    # (1,0) -> (2,0), (2,-1)
    # (-1,1) -> (-2,1), (-2,2)
    cells += [(2,0), (2,-1), (-2,1), (-2,2)]
    topologies.append(cells)
    
    # Topology 2: 3x3 hex block (9 cells) + 2 more
    # A 3x3 block in axial coords:
    # q from -1 to 1, r from -1 to 1, but hex grid isn't square
    # Let's use: all (q,r) with |q|<=1, |r|<=1, |q+r|<=1 gives 7 (the ring+center)
    # For 9: add (2,0), (0,2) -> not symmetric
    # Let's try: center + 6 + 2 opposite
    cells2 = [(0,0)] + neighbors + [(2,0), (-2,0)]
    topologies.append(cells2)
    
    # Topology 3: elongated
    cells3 = [(q, 0) for q in range(-5, 6)]  # line of 11
    topologies.append(cells3)
    
    # Topology 4: 2 rows of 5-6
    cells4 = [(q, r) for q in range(-4, 5) for r in [0, 1] if abs(q) + abs(r) <= 5]
    # That's too many; let me be more careful
    # Row r=0: q=-4..4 (9 cells), Row r=1: q=-3..4 (8 cells) -> 17, too many
    # Let's do: r=0: q=-3..3 (7), r=1: q=-3..2 (6) -> 13, still too many
    # r=0: q=-2..2 (5), r=1: q=-2..3 (6) -> 11
    cells4 = [(q, 0) for q in range(-2, 3)] + [(q, 1) for q in range(-2, 4)]
    topologies.append(cells4)
    
    # Topology 5: compact diamond-like
    cells5 = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),  # center+ring
              (2,0), (2,-1), (-2,1), (-2,2)]  # 4 more
    topologies.append(cells5)
    
    return topologies

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin
    
    best_result = None
    best_score = float('inf')
    
    topologies = _generate_topologies()
    
    for topo in topologies:
        if time.time() > deadline:
            break
        
        # Convert to cartesian
        positions = []
        for (q, r) in topo:
            x = 2.0 * q + 1.0 * r
            y = np.sqrt(3) * r
            positions.append((x, y, 0.0))
        
        # Check disjointness: all pairs should have center distance >= 2
        # (for unit hexes on a grid with spacing 2, this is guaranteed if
        #  no two cells are the same, which they aren't in a valid polyhex)
        # But let's verify
        valid = True
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                if dx*dx + dy*dy < 3.99:  # should be >= 4
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        s, center, angle = _compute_score(positions)
        
        if s < best_score:
            best_score = s
            best_result = (positions, center, s, angle)
    
    if best_result is None:
        # Fallback: linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner, center, s, angle = best_result
    return inner, center, s, angle

# EVOLVE-BLOCK-END
