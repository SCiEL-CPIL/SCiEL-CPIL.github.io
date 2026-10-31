"""Program for placing unit hexagons inside a regular hexagon using hexagonal lattice packing."""
import math
import random

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = math.sqrt(3.0)
    
    # Generate hexagonal lattice coordinates in axial coordinates
    # A hexagon of radius r has cells with max(|q|, |r|, |s|) <= r where s = -q-r
    # We want to place N=11 hexagons. Let's find the minimal hexagon of lattice cells
    # that contains at least 11 cells, and select 11 of them optimally.
    
    # Generate all cells in a hexagon of radius R
    def hexagon_cells(radius):
        cells = []
        for q in range(-radius, radius + 1):
            for r in range(-radius, radius + 1):
                s = -q - r
                if abs(s) <= radius:
                    cells.append((q, r))
        return cells
    
    # Find minimal radius that gives at least N cells
    radius = 0
    while True:
        cells = hexagon_cells(radius)
        if len(cells) >= N:
            break
        radius += 1
    
    all_cells = hexagon_cells(radius)
    
    # Convert axial to Cartesian for unit hexagons
    # Center of unit hexagon at axial (q, r):
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    def axial_to_cartesian(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)
    
    # For each subset of N cells from all_cells, compute the required outer hexagon side length
    # The outer hexagon must contain all unit hexagons. A unit hexagon centered at (cx, cy) with
    # angle 0 has its vertices at distance 1 from center. The outer hexagon of side S centered at
    # origin with angle 0 contains a point (x, y) if the point is inside the hexagon.
    # 
    # A regular hexagon of side S centered at origin with flat-top orientation (angle 0) has:
    # - width (horizontal) = 2*S
    # - height (vertical) = sqrt(3)*S
    # 
    # Actually, let me reconsider the orientation. A regular hexagon with "pointy top" vs "flat top".
    # With angle 0, let's say the hexagon has a flat top and bottom. Then:
    # - The six sides are horizontal (top, bottom) and slanted.
    # - Width = 2*S (from left vertex to right vertex is not correct for flat-top)
    # 
    # Let me use the standard: a regular hexagon with side length S, centered at origin, with one
    # side horizontal at the top. The vertices are at angles 0, 60, 120, 180, 240, 300 degrees
    # from center at distance S (circumradius = S for regular hexagon).
    # 
    # Wait, for a regular hexagon, circumradius = side length. So vertices are at distance S.
    # If vertex is at angle 0: (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), etc.
    # 
    # A point (x, y) is inside this hexagon if it satisfies certain linear inequalities.
    # 
    # For the hexagon with vertices at angles 0, 60, 120, 180, 240, 300:
    # The six sides have outward normals at angles 30, 90, 150, 210, 270, 330.
    # The distance from center to each side (apothem) = S * sqrt(3)/2.
    # 
    # So a point (x, y) is inside if for each normal direction at angle theta:
    # x*cos(theta) + y*sin(theta) <= S * sqrt(3)/2
    # 
    # But we also need the unit hexagons to be fully inside, not just their centers.
    # A unit hexagon centered at (cx, cy) extends up to distance 1 from (cx, cy) in all directions
    # (its circumradius is 1).
    # 
    # So we need: for each inner hexagon center (cx, cy), the entire hexagon must be inside the
    # outer hexagon. The most restrictive condition is that for each face of the outer hexagon,
    # the maximum projection of the inner hexagon onto that face's normal must be <= apothem of outer.
    # 
    # For a face with normal n = (cos(theta), sin(theta)), the inner hexagon centered at c projects
    # to [c·n - 1, c·n + 1] (since circumradius is 1, max projection offset is 1).
    # We need c·n + 1 <= S * sqrt(3)/2 for all 6 faces.
    # 
    # So: S >= (max over all centers c of max over 6 normals n of (c·n)) + 1) * 2/sqrt(3)
    
    normals = []
    for i in range(6):
        theta = math.pi / 3.0 * i + math.pi / 6.0  # 30, 90, 150, 210, 270, 330 degrees
        normals.append((math.cos(theta), math.sin(theta)))
    
    def required_side_length(cartesian_centers):
        """Compute the minimum side length of outer hexagon (angle 0) to contain all unit hexagons."""
        max_proj = 0.0
        for (cx, cy) in cartesian_centers:
            for (nx, ny) in normals:
                proj = cx * nx + cy * ny
                if proj > max_proj:
                    max_proj = proj
        return (max_proj + 1.0) * 2.0 / sqrt3
    
    # Try all combinations of N cells from all_cells
    # If all_cells is large, this could be expensive. Let's check size.
    if radius == 1:
        # 7 cells, need 11 - not enough, so radius must be >= 2
        pass
    
    # For radius 2, we have 19 cells. C(19,11) = 75582 which is manageable.
    best_side = float('inf')
    best_subset = None
    
    from itertools import combinations
    
    for combo in combinations(range(len(all_cells)), N):
        centers = [axial_to_cartesian(all_cells[i][0], all_cells[i][1]) for i in combo]
        side = required_side_length(centers)
        if side < best_side:
            best_side = side
            best_subset = combo
    
    # Build the inner list
    inner = []
    for idx in best_subset:
        q, r = all_cells[idx]
        x, y = axial_to_cartesian(q, r)
        inner.append([x, y, 0.0])
    
    return inner, (0.0, 0.0), best_side, 0.0
# EVOLVE-BLOCK-END
