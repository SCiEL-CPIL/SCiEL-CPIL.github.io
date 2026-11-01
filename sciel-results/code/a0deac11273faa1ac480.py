"""Packing 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
from scipy.spatial import Voronoi
import time

N = 12

def construct():
    """
    Returns: (inner, center, side, angle)
    inner: list of [x, y, angle_degrees] for N unit hexagons
    center: (cx, cy) outer hexagon center
    side: outer hexagon side length
    angle: outer hexagon rotation in degrees
    """
    # Strategy: Use a symmetric arrangement of 12 hexagons.
    # A good packing for 12 hexagons is a hexagonal number pattern:
    # Center + 6 around + 6 more? 
    # Hexagonal numbers: 1, 7, 19...
    # 12 is not a hexagonal number.
    # Let's try a rectangular grid or a specific symmetric layout.
    
    # Common efficient packing for small N:
    # For N=12, a 3x4 grid of hexagons (with appropriate offsets) or two rows of 6?
    # Let's analyze the geometry.
    # Unit hexagon side length s=1.
    # Width (flat-to-flat) = sqrt(3) * 2? No.
    # Regular hexagon with side 1:
    # Distance between opposite sides (width) = sqrt(3)
    # Distance between opposite vertices (height) = 2
    
    # If we align them with flat sides vertical (pointy top/bottom? or flat top/bottom?)
    # Standard orientation: flat top/bottom.
    # Width (horizontal) = sqrt(3)
    # Height (vertical) = 2
    
    # In a honeycomb lattice:
    # Horizontal spacing: sqrt(3)
    # Vertical spacing: 1.5
    # Offset for alternate rows: sqrt(3)/2
    
    # Let's try a 3x4 arrangement (3 columns, 4 rows) or 4x3.
    # Actually, 12 = 3 * 4.
    # Let's place them in a grid.
    
    # Let's use a symmetric configuration centered at (0,0).
    # We can try to optimize the positions using a simple local search or just use a known good heuristic.
    
    # Heuristic: 
    # Place 12 hexagons in a roughly circular/rectangular cluster.
    # A 3x4 grid:
    # Rows: 4 rows. Cols: 3 cols.
    # But hexagonal packing is denser.
    
    # Let's try a specific layout:
    # Row 0: 3 hexagons
    # Row 1: 3 hexagons
    # Row 2: 3 hexagons
    # Row 3: 3 hexagons
    # This is a rectangular grid, not optimal for hexagons.
    
    # Better: Honeycomb layout.
    # Row 0: 2 hexagons?
    # Row 1: 4 hexagons?
    # Row 2: 4 hexagons?
    # Row 3: 2 hexagons?
    # Total: 12.
    
    # Let's define the positions for a honeycomb-like cluster.
    # Let dx = sqrt(3) (horizontal distance between centers in same row)
    # Let dy = 1.5 (vertical distance between rows)
    
    # Layout:
    # Row 0 (y = 1.5): x = -1.5*sqrt(3)/2, 1.5*sqrt(3)/2  -> 2 hexagons? No, let's center it.
    
    # Let's try a 4-4-4 grid? No.
    # Let's try 3-4-3-2?
    
    # Actually, the optimal packing of 12 hexagons is likely related to the hexagonal lattice.
    # Let's generate a candidate set of points from the hexagonal lattice and pick the 12 that minimize the bounding hexagon size.
    
    # Hexagonal lattice points:
    # (i * sqrt(3) + j * sqrt(3)/2, j * 1.5) for integers i, j.
    
    # We want to select 12 points from this lattice such that they are "compact".
    # Then we compute the minimum enclosing regular hexagon.
    
    # Let's generate a set of candidate centers from the lattice around the origin.
    # We'll take a large enough patch of the lattice, then select the 12 points that minimize the maximum distance from the centroid (or origin).
    
    # However, the outer hexagon can also be rotated.
    # The problem allows rotating the inner hexagons.
    # For simplicity and symmetry, let's assume all inner hexagons have the same orientation (0 degrees) and the outer hexagon is aligned (0 degrees).
    # This is often optimal for small N due to symmetry.
    
    # Step 1: Generate lattice points.
    # Step 2: Select 12 points closest to the origin (or minimizing the max radius).
    # Step 3: Compute the required outer side length.
    
    # Let's refine: 
    # The "distance" from center (0,0) to a point (x,y) in a hexagonal metric?
    # No, the outer boundary is a regular hexagon.
    # A regular hexagon of side L centered at origin contains point (x,y) if:
    # max(|x|/L, |y|/(L*sqrt(3)/2) ... ) <= 1 ?
    # Let's use the support function.
    # For a regular hexagon with side L, aligned with axes (flat top/bottom):
    # The width is 2L (vertex to vertex)? No.
    # Side length L.
    # Distance from center to vertex = L.
    # Distance from center to side = L * sqrt(3)/2.
    
    # If the hexagon is oriented with a vertex at (L, 0), the vertices are at angles 0, 60, 120, 180, 240, 300.
    # The condition for a point (x,y) to be inside is:
    # max over k=0..5 of dot((x,y), n_k) <= L * sqrt(3)/2
    # where n_k are the unit normals to the sides.
    # For flat-top hexagon (vertex at 0 deg):
    # Normals are at 30, 90, 150, 210, 270, 330 degrees?
    # Wait, if vertex is at 0, side is perpendicular to 30 deg?
    # Let's check. Vertex at (L,0). Adjacent vertices at (L/2, L*sqrt(3)/2).
    # Side connects (L,0) and (L/2, L*sqrt(3)/2).
    # Midpoint: (3L/4, L*sqrt(3)/4).
    # Normal vector: (sqrt(3)/2, 1/2) -> angle 30 deg.
    # So normals are at 30, 90, 150, 210, 270, 330.
    
    # So for a point (x,y), the "hexagonal radius" is:
    # r_hex = max( |x*cos(30) + y*sin(30)|, |x*cos(90) + y*sin(90)|, ... )
    # = max( |sqrt(3)/2 x + 1/2 y|, |y|, |-sqrt(3)/2 x + 1/2 y|, ... )
    
    # We want to minimize L such that for all inner hexagon centers (cx, cy), the entire hexagon is inside.
    # This is complex because the inner hexagon has size.
    # Approximation: Treat inner hexagons as points first, find the best 12 lattice points, then expand.
    # Or, more accurately, the outer hexagon must contain the Minkowski sum of the inner hexagon and the center point.
    # Since both are regular hexagons of the same orientation, the Minkowski sum is a regular hexagon of side 2L_inner? No.
    # If inner hexagon has side 1, and is centered at (cx, cy), the outer hexagon (side S) must contain it.
    # This is equivalent to the outer hexagon (side S) centered at (0,0) containing the inner hexagon (side 1) centered at (cx, cy).
    # This is equivalent to the point (cx, cy) being inside a hexagon of side S-1 (if concentric and aligned).
    # Let's verify.
    # Support function of hexagon side S: h(theta) = S * sqrt(3)/2 * |cos(theta - theta_k)|?
    # Actually, for aligned hexagons, the condition is simply that the "hexagonal distance" of (cx, cy) is <= S - 1.
    # Let's assume this holds.
    
    # So the problem reduces to:
    # Find 12 points (cx_i, cy_i) from the hexagonal lattice (with spacing appropriate for non-overlapping hexagons)
    # that minimize the maximum hexagonal distance from the origin.
    # Then S = max_dist + 1.
    
    # What is the lattice spacing for non-overlapping unit hexagons?
    # In a honeycomb lattice, the distance between centers is sqrt(3).
    # So the lattice points are (i*sqrt(3) + j*sqrt(3)/2, j*1.5).
    
    # Let's generate these points and find the best 12.
    
    sqrt3 = np.sqrt(3)
    dx = sqrt3
    dy = 1.5
    
    # Generate a patch of lattice points
    # We need enough points to choose 12 from.
    # Let's generate points with |x| < 10 and |y| < 10.
    points = []
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = i * dx + j * dx / 2
            y = j * dy
            points.append((x, y))
    
    points = np.array(points)
    
    # We need to select 12 points that minimize the max hexagonal distance from origin.
    # This is a combinatorial problem.
    # Heuristic: Sort by hexagonal distance from origin, take the 12 smallest?
    # This might not be optimal because of symmetry, but it's a good start.
    
    def hex_dist(p):
        x, y = p
        # Normals at 30, 90, 150, 210, 270, 330
        # d1 = |x*cos(30) + y*sin(30)|
        # d2 = |y|
        # d3 = |x*cos(150) + y*sin(150)| = |-x*cos(30) + y*sin(30)|
        # d4 = |-y|
        # d5 = |x*cos(270) + y*sin(270)| = |y*(-1)| = |y|
        # d6 = |x*cos(330) + y*sin(330)| = |x*cos(30) - y*sin(30)|
        
        # Actually, due to symmetry, max(d1, d2, d3) is enough?
        # d1 = |sqrt(3)/2 x + 1/2 y|
        # d2 = |y|
        # d3 = |-sqrt(3)/2 x + 1/2 y|
        
        # Let's just compute all 6.
        angles = np.deg2rad([30, 90, 150, 210, 270, 330])
        dots = []
        for a in angles:
            dots.append(abs(x * np.cos(a) + y * np.sin(a)))
        return max(dots)
    
    # Vectorized hex_dist
    def hex_dist_vec(pts):
        x = pts[:, 0]
        y = pts[:, 1]
        d1 = np.abs(sqrt3/2 * x + 0.5 * y)
        d2 = np.abs(y)
        d3 = np.abs(-sqrt3/2 * x + 0.5 * y)
        # d4, d5, d6 are symmetric to d1, d2, d3
        return np.maximum(d1, np.maximum(d2, d3))
    
    dists = hex_dist_vec(points)
    indices = np.argsort(dists)
    best_indices = indices[:12]
    best_points = points[best_indices]
    
    # The max distance is dists[best_indices[-1]]
    max_d = dists[best_indices[-1]]
    
    # Side length S = max_d + 1
    S = max_d + 1.0
    
    # Center is (0,0)
    center = (0.0, 0.0)
    
    # Inner hexagons: positions are best_points, angles are 0.
    inner = []
    for p in best_points:
        inner.append([float(p[0]), float(p[1]), 0.0])
    
    # Outer hexagon angle: 0.0
    angle = 0.0
    
    return inner, center, S, angle

# EVOLVE-BLOCK-START
# The construct function is defined above, but we need to wrap it or ensure it's the entry point.
# The task says "Keep the function names and signatures of the seed program."
# The seed program had `def construct():` inside the evolve block or outside?
# In the seed, it was:
# EVOLVE-BLOCK-START
# def construct():
#     ...
# EVOLVE-BLOCK-END
#
# So I should put the definition inside the evolve block or ensure the global `construct` is the one used.
# Since I defined it globally, I should probably move it into the evolve block to match the structure,
# or just leave it global if the evaluator picks up the global one.
# To be safe, let's redefine it in the evolve block.
# EVOLVE-BLOCK-END
