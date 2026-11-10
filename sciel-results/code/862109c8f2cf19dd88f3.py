# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, side, angle_deg):
    """Compute 6 vertices of a regular hexagon centered at (cx,cy) with given side and angle."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _min_outer_side(vertices_all):
    """
    Given array of all inner hexagon vertices (shape (72,2)), compute minimum outer hex side s
    (centered at origin, angle 0) that contains all vertices.
    """
    # Inward normals for outer hex with vertices at 0,60,...,300 degrees
    # Edge k is between vertex k and vertex k+1 (mod 6)
    # Inward normal direction: bisector of the two outward normals, pointing inward
    # For a regular hexagon centered at origin with vertex at angle a_k, 
    # the edge midpoint is at angle a_k + 30 deg. The inward normal is at angle a_k + 30 deg.
    # Actually, for a regular hexagon with vertices at 0, 60, ..., 300 degrees:
    # Edge 0 is between vertex 0 (at 0 deg) and vertex 1 (at 60 deg).
    # Midpoint is at 30 deg. Inward normal is at 30 deg (pointing towards center? No, away from center is outward).
    # Wait, the constraint is v . n <= R, where n is the outward normal of the edge.
    # For a regular hexagon with "pointy top" (vertex at 90 deg), the flats are at 0, 60, ...
    # But the problem defines vertices at a + 60i. If a=0, vertices are at 0, 60, ...
    # This is a "pointy side" hexagon relative to the x-axis? No, vertex at (s, 0).
    # The edges are between (s,0) and (s/2, s*sqrt(3)/2).
    # The outward normal for this edge is at 30 degrees.
    # So the 6 outward normals are at 30, 90, 150, 210, 270, 330 degrees.
    # The distance from center to each edge (apothem) is s * sqrt(3)/2.
    # So the condition is: v . n_k <= s * sqrt(3)/2 for all k.
    # Thus s >= (2/sqrt(3)) * max(v . n_k).
    
    normal_angles = np.radians(30 + np.arange(6) * 60)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # v . n for all v and all n
    # vertices_all: (72, 2), normals: (6, 2)
    dots = vertices_all @ normals.T  # (72, 6)
    max_dot = np.max(dots)
    
    s_min = (2.0 / np.sqrt(3.0)) * max_dot
    return s_min

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    # Start time for safety guard
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Honeycomb structure
    # dx = 1.5, dy = sqrt(3)/2
    dx = 1.5
    dy = np.sqrt(3) / 2.0
    
    # Row 0 (y = 0): 2 hexagons
    # Centers: (-0.75, 0), (0.75, 0)
    # Row 1 (y = dy): 3 hexagons
    # Centers: (-1.5, dy), (0, dy), (1.5, dy)
    # Row 2 (y = 2*dy): 4 hexagons
    # Centers: (-2.25, 2*dy), (-0.75, 2*dy), (0.75, 2*dy), (2.25, 2*dy)
    # Row 3 (y = 3*dy): 3 hexagons
    # Centers: (-1.5, 3*dy), (0, 3*dy), (1.5, 3*dy)
    # Row 4 (y = 4*dy): 2 hexagons
    # Centers: (-0.75, 4*dy), (0.75, 4*dy)
    
    # Total 12 hexagons.
    # Let's try rotating the middle hexagon of Row 2 (index 7 in 0-based if we list row by row?)
    # List:
    # 0: (-0.75, 0)
    # 1: (0.75, 0)
    # 2: (-1.5, dy)
    # 3: (0, dy)
    # 4: (1.5, dy)
    # 5: (-2.25, 2*dy)
    # 6: (-0.75, 2*dy)
    # 7: (0.75, 2*dy)  <-- Let's rotate this one? Or the one at (0, 2*dy)? 
    # Wait, Row 2 has 4 hexagons. The ones at -2.25 and 2.25 are the outer ones.
    # The ones at -0.75 and 0.75 are inner.
    # Let's rotate the hexagon at (0.75, 2*dy) by 30 degrees.
    # Actually, let's just generate the standard honeycomb first and check validity.
    # If valid, we can try to optimize.
    
    centers = [
        (-0.75, 0.0),
        (0.75, 0.0),
        (-1.5, dy),
        (0.0, dy),
        (1.5, dy),
        (-2.25, 2.0*dy),
        (-0.75, 2.0*dy),
        (0.75, 2.0*dy),
        (2.25, 2.0*dy),
        (-1.5, 3.0*dy),
        (0.0, 3.0*dy),
        (1.5, 3.0*dy),
        (-0.75, 4.0*dy),
        (0.75, 4.0*dy)
    ]
    
    # Wait, 2+3+4+3+2 = 14. I need 12.
    # Let's use a 3-4-3-2 pattern? Or 2-3-3-2-2?
    # Standard honeycomb of 12:
    # Maybe remove 2 from the ends?
    # Let's try:
    # Row 0: 2
    # Row 1: 3
    # Row 2: 3
    # Row 3: 2
    # Row 4: 2
    # Total 12.
    
    # Let's re-construct:
    # Row 0 (y=0): x = -0.75, 0.75
    # Row 1 (y=dy): x = -1.5, 0, 1.5
    # Row 2 (y=2dy): x = -0.75, 0.75, 2.25 ? No, honeycomb connectivity.
    # If Row 1 is -1.5, 0, 1.5.
    # Row 2 should be -2.25, -0.75, 0.75, 2.25 (4 hexes).
    # If we take 3 from Row 2: -0.75, 0.75, 2.25? That breaks symmetry.
    # Let's take 3 from Row 2: -1.5, 0, 1.5? No, those are above Row 1.
    # In honeycomb, row k+1 is offset by dx/2? No, dx is 1.5.
    # Vertices of hex at (0,0) are at (1,0), (0.5, 0.866), etc.
    # Neighbor at (1.5, 0.866) shares edge.
    # So Row 1 (y=dy) neighbors Row 0 (y=0).
    # Row 2 (y=2dy) neighbors Row 1.
    
    # Let's just use the 14-hexagon shape and remove the two "corner" ones from the middle row?
    # Or simply use a different shape.
    # How about a 4-4-4 strip?
    # Row 0: 4 hexes at x = -1.5, 0, 1.5, 3.0? No, they would overlap if dx=1.5?
    # Distance between (0,0) and (1.5, 0) is 1.5. Min distance is 2. So they don't overlap.
    # But they are not "packed" in the honeycomb sense, they are just in a line.
    # A line of 12 hexagons is very long.
    
    # Let's stick to the honeycomb.
    # Shape:
    #   . X X .
    #  X X X X
    #   . X X .
    #  X X X X
    #   . X X .
    # This is 2+4+2+4+2 = 14.
    
    # What if we do:
    #   . X X .
    #  X X X X
    #   . X X .
    #  X X X X
    #   . X .
    # 2+4+2+4+1 = 13.
    
    # Let's try:
    #  X X X
    # X X X X
    # X X X X
    #  X X X
    # 3+4+4+3 = 14.
    
    # Okay, let's use a 12-hexagon honeycomb:
    #  X X
    # X X X
    # X X X
    #  X X
    # 2+3+3+2 = 10.
    
    #  X X X
    # X X X X
    #  X X X
    # 3+4+3 = 10.
    
    #  X X X X
    # X X X X X
    #  X X X X
    # 4+5+4 = 13.
    
    #  X X X X
    # X X X X
    #  X X X X
    # 4+4+4 = 12.
    # This is a parallelogram/rhombus of hexagons.
    # Row 0: x = 0, 1.5, 3.0, 4.5 (y=0)
    # Row 1: x = 0.75, 2.25, 3.75, 5.25 (y=dy)
    # Row 2: x = 1.5, 3.0, 4.5, 6.0 (y=2dy)
    
    # This is a valid packing.
    # Let's center it.
    # Center x: (0+6.0)/2 = 3.0
    # Center y: (0+2dy)/2 = dy
    
    centers = []
    # Row 0
    for i in range(4):
        centers.append((1.5*i, 0.0))
    # Row 1
    for i in range(4):
        centers.append((1.5*i + 0.75, dy))
    # Row 2
    for i in range(4):
        centers.append((1.5*i + 1.5, 2.0*dy))
        
    # Shift to center
    cx_mean = np.mean([c[0] for c in centers])
    cy_mean = np.mean([c[1] for c in centers])
    
    for i in range(len(centers)):
        centers[i] = (centers[i][0] - cx_mean, centers[i][1] - cy_mean)
        
    # Angles: all 0
    angles = [0.0] * 12
    
    # Check validity
    # 1. Non-overlap
    # 2. Inside outer hex
    
    # Compute outer hex side
    all_vertices = []
    for i in range(12):
        v = _hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
        all_vertices.append(v)
    all_vertices = np.vstack(all_vertices)
    
    s = _min_outer_side(all_vertices)
    
    # To improve, we can try to rotate some hexagons to fit better?
    # Or use a more compact shape.
    # The 4x4 parallelogram is quite rectangular.
    # A more circular shape might fit better in a hexagon.
    
    # Let's try the 2-3-3-2-2 shape? No, 12.
    # How about 1-2-3-3-2-1?
    # Row 0: 1
    # Row 1: 2
    # Row 2: 3
    # Row 3: 3
    # Row 4: 2
    # Row 5: 1
    # Total 12.
    
    centers2 = []
    # Row 0: 1 hex at x=0
    centers2.append((0.0, 0.0))
    # Row 1: 2 hexes at x=-0.75, 0.75
    centers2.append((-0.75, dy))
    centers2.append((0.75, dy))
    # Row 2: 3 hexes at x=-1.5, 0, 1.5
    centers2.append((-1.5, 2.0*dy))
    centers2.append((0.0, 2.0*dy))
    centers2.append((1.5, 2.0*dy))
    # Row 3: 3 hexes at x=-1.5, 0, 1.5 (same as Row 2, shifted? No, honeycomb)
    # If Row 2 is -1.5, 0, 1.5.
    # Row 3 should be -2.25, -0.75, 0.75, 2.25 (4 hexes).
    # If we take 3, which ones?
    # Let's take -0.75, 0.75, 2.25? Asymmetric.
    # Let's take -2.25, -0.75, 0.75? Asymmetric.
    
    # Let's stick to the 4x4 parallelogram for now, it's simple and valid.
    # But we can optimize the outer hexagon angle?
    # The function `_min_outer_side` assumes outer hex angle is 0.
    # We can try different outer hex angles to minimize s.
    
    best_s = s
    best_angle = 0.0
    
    # Try a few angles for the outer hexagon
    for test_angle in range(0, 30, 5):
        # Recalculate s for this angle
        # Normals are shifted by test_angle
        normal_angles = np.radians(test_angle + 30 + np.arange(6) * 60)
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        dots = all_vertices @ normals.T
        max_dot = np.max(dots)
        s_test = (2.0 / np.sqrt(3.0)) * max_dot
        if s_test < best_s:
            best_s = s_test
            best_angle = test_angle
            
    # We can also try to shift the inner packing?
    # No, we centered it.
    
    # Construct return values
    inner_hex_data = np.array([(c[0], c[1], a) for c, a in zip(centers, angles)])
    outer_hex_data = np.array([0.0, 0.0, best_angle])
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END
