"""Rewritten program: exhaustive search over 11-cell subsets of a size-3 hexagonal grid
to minimize the enclosing regular hexagon side length."""

import time
import math
from itertools import combinations
from math import sqrt, cos, sin

N = 11
TIME_LIMIT = 120.0
# Use a safety margin
DEADLINE = time.time() + TIME_LIMIT - 5.0


def hex_to_xy(q, r, d=sqrt(3.0)):
    """Convert axial coords (q, r) with spacing d to Cartesian (x, y)."""
    x = d * (q + r / 2.0)
    y = d * (r * 0.8660254037844386)  # d * sqrt(3)/2
    return x, y


def xy_to_hex(q, r):
    """Return Cartesian coords for axial (q, r) with unit spacing (d=sqrt(3))."""
    return hex_to_xy(q, r, sqrt(3.0))


def generate_cells(size):
    """Generate all axial coords (q, r) of a hexagon of given size (radius)."""
    cells = []
    for q in range(-size, size + 1):
        for r in range(-size, size + 1):
            if abs(q + r) <= size:
                cells.append((q, r))
    return cells


def project_widths(coords):
    """
    Given list of (x, y) Cartesian points, compute the 3 directional widths
    for a regular hexagon (flat-top orientation).
    
    For a regular hexagon with flat-top orientation and side L:
    - Width in x-direction: L * sqrt(3)
    - Width in 60-degree direction: L * sqrt(3)
    - Width in 120-degree direction: L * sqrt(3)
    
    Actually, for a flat-top regular hexagon centered at origin with side L:
    The vertices are at angles 0, 60, 120, 180, 240, 300 degrees from center,
    at distance L from center.
    
    The width (max - min) along any of the 3 principal axes (0°, 60°, 120°) 
    through the center equals 2L.
    
    Wait, let me reconsider. For a regular hexagon with side length L:
    - The distance from center to any vertex is L.
    - The distance from center to any edge midpoint is L * sqrt(3)/2.
    - The width across opposite vertices is 2L.
    - The width across opposite edges is 2 * L * sqrt(3)/2 = L * sqrt(3).
    
    For a flat-top hexagon (vertex at angle 0):
    - Along x-axis (through vertices): width = 2L
    - Along 60° axis (through vertices): width = 2L
    - Along 30° axis (through edges): width = L * sqrt(3)
    
    Hmm, I need to be more careful. Let me think about this differently.
    
    A regular hexagon with side L, oriented with a flat top (so the top and bottom
    edges are horizontal), has:
    - Width (horizontal, left to right) = 2 * L * cos(30°) = L * sqrt(3)
    - Height (vertical, bottom to top) = 2L
    
    Wait, no. Let me just define it properly.
    
    A regular hexagon with side L, flat-top orientation (one pair of edges horizontal):
    The vertices relative to center at distance R = L from center, at angles:
    30°, 90°, 150°, 210°, 270°, 330° (for flat-top, the top edge is horizontal)
    
    Actually, for flat-top:
    Top-left: (-L/2, L*sqrt(3)/2)
    Top-right: (L/2, L*sqrt(3)/2)
    Right: (L, 0)
    Bottom-right: (L/2, -L*sqrt(3)/2)
    Bottom-left: (-L/2, -L*sqrt(3)/2)
    Left: (-L, 0)
    
    So width (x-range) = 2L, height (y-range) = L*sqrt(3).
    
    For a point set to fit inside such a hexagon centered at (cx, cy):
    - The hexagon in x-direction spans from cx - L to cx + L
    - In the 60° direction... 
    
    Let me use the standard approach: project points onto 3 axes at 60° intervals,
    find the max projection range on each axis. The minimum enclosing hexagon side
    length is max(range_x, range_y1, range_y2) / 2, where the ranges are computed
    along the 3 principal directions.
    
    Actually, for a regular hexagon with side L and flat-top orientation:
    - Project onto x-axis: range must be <= 2L
    - Project onto axis at 60°: range must be <= 2L  
    - Project onto axis at 120°: range must be <= 2L
    
    Wait, that's not right either. Let me just compute it numerically.
    
    For a flat-top regular hexagon with side L centered at origin:
    The 6 vertices are at:
    (L, 0), (L/2, L*sqrt(3)/2), (-L/2, L*sqrt(3)/2), (-L, 0), (-L/2, -L*sqrt(3)/2), (L/2, -L*sqrt(3)/2)
    
    So the x-range is [-L, L], width = 2L.
    The y-range is [-L*sqrt(3)/2, L*sqrt(3)/2], width = L*sqrt(3).
    
    For a point (px, py) to be inside this hexagon, it must satisfy:
    |px| <= L and |py| <= L*sqrt(3)/2 and |px| + |py|/sqrt(3) <= L ... 
    Hmm, let me just use the projection method.
    
    The 3 normal directions to the edges are at 90°, 30°, -30° (or 0°, 60°, 120° if we
    consider the directions through opposite vertices).
    
    I think the cleanest approach: the minimum L such that all points fit in a hexagon
    of side L is determined by the maximum of the 3 directional "widths" divided by
    the appropriate factor.
    
    For flat-top hexagon side L:
    - Width in x-direction: 2L
    - Width in direction 60° from x-axis: 2L (through opposite vertices)
    - Width in direction 30° from x-axis: L*sqrt(3) (through opposite edges)
    
    Wait, I'm overcomplicating this. Let me just say:
    
    For a regular hexagon with side L, the maximum distance from center to any point
    on the boundary in any direction is L (at vertices) or L*sqrt(3)/2 (at edge midpoints).
    
    The key insight: a point (x,y) is inside a flat-top regular hexagon of side L centered
    at origin iff:
      |x| <= L
      |y| <= L * sqrt(3)/2  
      |x| + |y| / sqrt(3) <= L  (for the diagonal constraints)
    
    Actually, the simplest correct formulation: project onto the 3 axes at 0°, 60°, 120°.
    For a flat-top hexagon of side L, the max projection on each of these axes is L.
    So the minimum L is max over all 3 axes of (max_proj - min_proj) / 2.
    
    Let me verify: For the hexagon vertices listed above, projecting onto x-axis:
    max = L, min = -L, range = 2L, so (max-min)/2 = L. Correct.
    
    Projecting onto 60° axis (direction (cos60, sin60) = (0.5, sqrt(3)/2)):
    Dot products with vertices:
    (L,0) -> L*0.5 = 0.5L
    (L/2, L*sqrt(3)/2) -> L/2*0.5 + L*sqrt(3)/2*sqrt(3)/2 = L/4 + 3L/4 = L
    (-L/2, L*sqrt(3)/2) -> -L/4 + 3L/4 = 0.5L
    (-L, 0) -> -0.5L
    (-L/2, -L*sqrt(3)/2) -> -L/4 - 3L/4 = -L
    (L/2, -L*sqrt(3)/2) -> L/4 - 3L/4 = -0.5L
    max = L, min = -L, range = 2L, (max-min)/2 = L. Correct.
    
    So the algorithm is: for 3 axes at 0°, 60°, 120°, compute the range of projections,
    and L = max(range_i) / 2.
    """
    points = coords
    angles = [0, 60, 120]
    max_half_range = 0
    for a_deg in angles:
        a = math.radians(a_deg)
        dx = math.cos(a)
        dy = math.sin(a)
        projections = [x * dx + y * dy for (x, y) in points]
        r = max(projections) - min(projections)
        max_half_range = max(max_half_range, r / 2.0)
    return max_half_range


def construct():
    start_time = time.time()
    
    # Generate all cells of a size-3 hexagon (19 cells)
    cells = generate_cells(3)
    n_cells = len(cells)
    
    # Precompute Cartesian coordinates
    cart_coords = [hex_to_xy(q, r, sqrt(3.0)) for (q, r) in cells]
    
    best_L = float('inf')
    best_subset = None
    best_center = (0.0, 0.0)
    
    # Try all combinations of 11 cells from 19
    # C(19, 11) = 75582, very fast
    count = 0
    for subset in combinations(range(n_cells), N):
        if time.time() > DEADLINE:
            break
        count += 1
        pts = [cart_coords[i] for i in subset]
        
        # Compute center of mass
        cx = sum(p[0] for p in pts) / N
        cy = sum(p[1] for p in pts) / N
        
        # Translate to center
        centered_pts = [(x - cx, y - cy) for (x, y) in pts]
        
        L = project_widths(centered_pts)
        
        if L < best_L:
            best_L = L
            best_subset = subset
            best_center = (cx, cy)
    
    # If no valid subset found (shouldn't happen), fall back to seed
    if best_subset is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Build inner list
    inner = []
    for i in best_subset:
        x, y = cart_coords[i]
        inner.append([x, y, 0.0])
    
    return inner, best_center, best_L, 0.0
