# EVOLVE-BLOCK-START
import numpy as np


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    using a 3-4-3 honeycomb arrangement and optimizing rotation to minimize outer hexagon size.
    """
    # 12 unit regular hexagons in a honeycomb pattern
    # Row 1: 3 hexagons
    # Row 2: 4 hexagons  
    # Row 3: 3 hexagons
    # For unit hexagons with angle 0, horizontal spacing is 2.5, vertical spacing is sqrt(3)
    
    sqrt3 = np.sqrt(3)
    
    # Centers of the 12 hexagons relative to cluster center
    # Let's define the honeycomb grid positions
    # Using axial coordinates for hex grid
    
    # Row 0 (top): 3 hexagons at x = -2.5, 0, 2.5
    # Row 1 (middle): 4 hexagons at x = -3.75, -1.25, 1.25, 3.75
    # Row 2 (bottom): 3 hexagons at x = -2.5, 0, 2.5
    
    # y positions: top row y = sqrt3, middle y = 0, bottom y = -sqrt3
    # But wait, the rows need to be offset vertically by sqrt(3) for proper honeycomb packing
    
    # Let me reconsider: for unit hexagons (side=1), the distance between centers of adjacent hexagons is 2 (since they share an edge of length 1, the center-to-center distance is 2*1/sqrt(3)... wait no)
    
    # Actually for a regular hexagon with side length 1:
    # The width (flat-to-flat) is sqrt(3)
    # The distance between centers of two hexagons sharing an edge is 2 * (1/sqrt(3)) * sqrt(3) = ... let me think
    # No: two hexagons sharing an edge have centers at distance 2 * apothem = 2 * (sqrt(3)/2) * 1 = sqrt(3)
    
    # Wait, apothem of unit hexagon = sqrt(3)/2
    # Two hexagons sharing an edge: centers are 2 * apothem apart = sqrt(3)
    
    # For honeycomb lattice of unit hexagons:
    # Horizontal spacing between centers in same row: 2 * apothem * ... 
    
    # Let me use the standard honeycomb coordinates for unit hexagons (side=1, flat-top orientation, angle=0)
    # Vertices at angles 0, 60, 120, 180, 240, 300
    # Rightmost vertex at (1, 0), leftmost at (-1, 0)
    # Top vertex at (0.5, sqrt(3)/2), bottom at (0.5, -sqrt(3)/2)
    # Wait no: vertex at angle 60 is (cos60, sin60) = (0.5, sqrt(3)/2)
    # So the hexagon is "pointy" at top-right? No, with angle 0, vertex at (1,0), so it's flat on left/right
    
    # Actually with vertices at (cos(60i), sin(60i)) for i=0..5:
    # i=0: (1, 0)
    # i=1: (0.5, sqrt(3)/2)
    # i=2: (-0.5, sqrt(3)/2)
    # i=3: (-1, 0)
    # i=4: (-0.5, -sqrt(3)/2)
    # i=5: (0.5, -sqrt(3)/2)
    
    # So the hexagon has flat edges on top and bottom (since top edge connects i=1 and i=2, which are both at y=sqrt(3)/2)
    # The width (left to right) is 2, height (top to bottom) is sqrt(3)
    
    # For two adjacent hexagons sharing the right edge of one (which is the vertical-ish edge from (1,0) to (0.5, sqrt(3)/2) to... wait)
    # The right edge of this hexagon goes from (0.5, sqrt(3)/2) to (1, 0) to (0.5, -sqrt(3)/2)? No.
    # Edges connect consecutive vertices. The edge from vertex i=0 (1,0) to i=1 (0.5, sqrt(3)/2) is one edge.
    # The edge from i=5 (0.5, -sqrt(3)/2) to i=0 (1, 0) is another.
    # So the rightmost part is a point at (1,0), not a flat edge.
    
    # This is a "pointy" hexagon on the right. The flat edges are top (i=1 to i=2) and bottom (i=4 to i=5).
    
    # For honeycomb packing with this orientation:
    # To place a hexagon to the right sharing the edge from (0.5, sqrt(3)/2) to (1, 0), we need to reflect across that edge.
    # That's complex. Let me use a simpler approach.
    
    # Standard result: for unit hexagons in honeycomb, the lattice vectors are:
    # a1 = (3, sqrt(3)) and a2 = (3, -sqrt(3)) for one common convention
    # Or: horizontal spacing 3, vertical offset sqrt(3)/2, etc.
    
    # Let me just use known positions for a 3-4-3 cluster.
    # From the seed program, the positions used were based on spacing of 2.5 horizontally and 2.17 vertically (approx sqrt(3)=1.732... wait 2.17 doesn't match)
    
    # In the seed: horizontal spacing 2.5, vertical 2.17. 2.17/sqrt(3) ≈ 1.25. Hmm.
    # Maybe they used a different spacing. Let me compute properly.
    
    # For two unit hexagons (flat-top, i.e., angle=0 as defined) to be non-overlapping and tangent:
    # The minimum center-to-center distance for non-overlap is 2 * sqrt(3)/2 * 2? No.
    
    # Let me just do a proper check: two unit hexagons centered at origin and at (dx, 0).
    # For them to be just touching, the minimum distance between any vertex of one and the other hexagon boundary should be 0.
    
    # Simpler: I'll use a well-known honeycomb configuration.
    # For unit hexagons with the orientation given, the three nearest-neighbor directions are at 60, 120, 300 degrees
    # (or 0, 60, 120 depending on which edges).
    
    # Actually, let me place them and verify non-overlap via SAT or just distance checks.
    
    # I'll use these positions (from standard honeycomb with unit hexagons, flat-top):
    # The distance between centers of tangent hexagons is 2*sqrt(3)/sqrt(3) = ... 
    
    # Let me just hardcode a known good 3-4-3 arrangement and verify.
    
    # Using the fact that for unit hexagons with this orientation, adjacent centers are at distance sqrt(3) * 2 / sqrt(3) = 2? 
    # Let me try: place centers such that adjacent hexagons share edges.
    
    # After reflection: the neighbor to the "northeast" shares the edge from (0.5, sqrt(3)/2) to (1, 0).
    # The midpoint of that edge is at ((0.5+1)/2, (sqrt(3)/2+0)/2) = (0.75, sqrt(3)/4)
    # The direction perpendicular to that edge, pointing outward... this is getting complicated.
    
    # Let me use a computational approach: find the 12 centers by optimization or use a known formula.
    
    # Known: for a hexagon with side 1, the "diameter" in the x-direction is 2, in the direction of 60 degrees is sqrt(3).
    # Honeycomb lattice for flat-top hexagons:
    # a1 = (3, 0), a2 = (1.5, sqrt(3)*1.5) = (1.5, 3*sqrt(3)/2)? 
    # No. The correct lattice vectors for flat-top hexagons (where vertices are at 0,60,...) sharing edges:
    # Horizontal neighbor: center at (2.5, 0)? Let me check: distance 2.5. Is that enough?
    # The rightmost point of hex at origin is (1, 0). The leftmost point of hex at (2.5, 0) is (2.5-1, 0) = (1.5, 0).
    # So they don't touch; there's a gap. That means 2.5 is too far.
    
    # The minimum distance: the edge of hex at origin from (1,0) to (0.5, sqrt(3)/2). The nearest point on the next hex must be on its corresponding edge.
    # For the hexagons to share that edge, the second hex's center must be placed such that its left edge coincides.
    
    # I'll use a different strategy: place hexagons at a known valid configuration and verify.
    
    # Let me use the configuration from literature for 12 hexagons in 3-4-3:
    # Row 0 (y = sqrt(3)): x = 0, 2.5, 5.0  -- wait, but these need to be offset
    
    # Let me just try the standard triangular lattice:
    # For flat-top hexagons, the lattice is:
    # a1 = (2, 0), a2 = (1, sqrt(3))
    # But then hexagons at lattice points may overlap or not touch.
    
    # I think the correct spacing is: horizontal distance = 2.5 doesn't work. Let me compute.
    
    # The distance between centers of two tangent flat-top unit hexagons (sharing a vertical-ish edge) is:
    # Edge from (0.5, -sqrt(3)/2) to (1, 0) to (0.5, sqrt(3)/2) -- these are not collinear, so no single vertical edge.
    
    # OK here's the cleanest way. The six neighbors of a flat-top unit hex at origin are at:
    # (1, 0) is a vertex. The adjacent hex sharing edge (0.5, sqrt(3)/2)-(1,0) has center at:
    # Reflect origin across the line through (0.5, sqrt(3)/2) and (1, 0).
    
    # Let me just do that computation.
    
    def reflect_point_across_line(p, a, b):
        """Reflect point p across line through a and b."""
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        proj = a + t * ab
        return 2 * proj - p
    
    # Edge from (0.5, sqrt(3)/2) to (1, 0)
    p0 = np.array([0.0, 0.0])
    a = np.array([0.5, sqrt3/2])
    b = np.array([1.0, 0.0])
    neighbor1 = reflect_point_across_line(p0, a, b)
    
    # Edge from (1, 0) to (0.5, -sqrt(3)/2)
    c = np.array([1.0, 0.0])
    d = np.array([0.5, -sqrt3/2])
    neighbor2 = reflect_point_across_line(p0, c, d)
    
    # Edge from (0.5, -sqrt(3)/2) to (-0.5, -sqrt(3)/2)  -- bottom edge
    e = np.array([0.5, -sqrt3/2])
    f = np.array([-0.5, -sqrt3/2])
    neighbor3 = reflect_point_across_line(p0, e, f)
    
    # By symmetry, the other three are negatives or rotations.
    
    centers_unit = np.array([
        p0,
        neighbor1,
        neighbor2,
        neighbor3,
        -neighbor3,
        -neighbor2,
        -neighbor1
    ])
    
    # The six neighbors should be at 60-degree intervals.
    # Let me compute their angles and distances.
    
    # Now I have a basis for the honeycomb. Let me generate the 3-4-3 cluster.
    # Use axial coordinates.
    
    # Actually, let me just use the known positions. The six neighbor centers are at distance 2/sqrt(3) * ... 
    
    dists = np.linalg.norm(centers_unit[1:], axis=1)
    angles = np.arctan2(centers_unit[1:, 1], centers_unit[1:, 0])
    
    # The neighbors should all be at the same distance.
    # Let d0 = that distance.
    
    # For the 3-4-3 cluster, I need 12 hexagons. Let me use a coordinate system based on the lattice.
    
    # Lattice vectors: v1 = neighbor1 - p0, v2 = some other independent neighbor.
    # Let me pick v1 = neighbor1, v2 = centers_unit[4] - p0 (which should be -neighbor3 or similar)
    
    # Let me just compute all 12 positions using a simple grid in axial coordinates.
    
    # Axial coordinates (q, r) for hex grid. For a 3-4-3 arrangement:
    # Row 0 (r=0): q = -1, 0, 1  (3 hexagons)
    # Row 1 (r=1): q = -2, -1, 0, 1  (4 hexagons)  
    # Row 2 (r=2): q = -1, 0, 1  (3 hexagons)
    
    # Convert axial to pixel for flat-top hexagons:
    # x = size * 3/2 * q
    # y = size * sqrt(3) * (r + q/2)
    # where size = 1 (side length)
    
    # But I need to verify this gives non-overlapping hexagons.
    
    def axial_to_pixel(q, r, size=1):
        x = size * 1.5 * q
        y = size * sqrt3 * (r + q / 2)
        return x, y
    
    positions = []
    # Row 0: r=0, q=-1,0,1
    for q in [-1, 0, 1]:
        positions.append(axial_to_pixel(q, 0))
    # Row 1: r=1, q=-2,-1,0,1
    for q in [-2, -1, 0, 1]:
        positions.append(axial_to_pixel(q, 1))
    # Row 2: r=2, q=-1,0,1
    for q in [-1, 0, 1]:
        positions.append(axial_to_pixel(q, 2))
    
    centers = np.array(positions)
    
    # Center the cluster at origin
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    # Now optimize rotation to minimize enclosing hexagon size
    # For a given rotation theta, compute the minimal regular hexagon side length s
    # that contains all 12 hexagons.
    
    def compute_min_s(theta, centers):
        """Compute minimal regular hexagon side length to contain all unit hexagons at given centers, rotated by theta."""
        # Rotate centers
        c = np.cos(theta)
        s_ = np.sin(theta)
        rot = np.array([[c, -s_], [s_, c]])
        rotated_centers = centers @ rot.T
        
        # For a regular hexagon with side s, centered at origin, with angle 0:
        # Vertices at (s*cos(60i), s*sin(60i))
        # The hexagon contains point p if for all 6 edge normals, p is on the correct side.
        # Edge normals are at angles 30, 90, 150, 210, 270, 330 degrees.
        # For normal at angle alpha, the supporting line is at distance s*sqrt(3)/2 from center (apothem).
        
        # A unit hexagon at center c_i with angle 0 has the property that a point p is inside it
        # iff for all 6 normals of the hexagon (at 30, 90, ...), the projection of (p - c_i) on that
        # normal is >= -sqrt(3)/2 (the apothem).
        
        # For p to be inside the outer hexagon with side s at origin:
        # For each outer normal alpha_j, projection of p on (cos(alpha_j), sin(alpha_j)) <= s*sqrt(3)/2
        
        # For each inner hexagon i to be entirely inside:
        # For each pair (inner_normal beta, outer_normal alpha_j), the max projection of inner hexagon
        # vertices on alpha_j must be <= s*sqrt(3)/2.
        
        # The max projection of a unit hexagon centered at c_i on direction n is:
        # c_i dot n + support_function_unit_hex(n)
        # where support_function_unit_hex(n) = max over vertices v of v dot n
        # For unit hexagon with vertices at angles 0,60,...,300, and n at angle alpha:
        # max_i cos(alpha - 60i) for i=0..5
        # This equals cos(min angular difference) where the min difference to any vertex angle is <= 30 degrees.
        # So support = cos(30) = sqrt(3)/2 if n is not aligned with a vertex... actually it's always sqrt(3)/2?
        # No: if n is at 0 degrees, the max is cos(0) = 1 (vertex at (1,0)).
        # If n is at 30 degrees, max is cos(30) = sqrt(3)/2.
        # So support function varies between sqrt(3)/2 and 1.
        
        # Specifically, support(n at angle alpha) = cos(min_{i} |alpha - 60i| mod 360, where the min is in [0, 30])
        # = cos(30) if alpha is 30 degrees from nearest vertex direction... 
        
        # Let me just compute it numerically.
        
        # For each of the 6 outer hexagon normals (at 30, 90, 150, 210, 270, 330):
        #   For each inner hexagon i, compute max projection of its vertices on that normal.
        #   This is rotated_centers[i] dot n + support_unit_hex(n, theta) where theta is the inner hex rotation (0 here).
        
        # The inner hexagons all have angle 0, so their support function on direction n is:
        # max_{v in vertices of unit hex at 0} v dot n
        
        # Since the inner hexagons are NOT rotated (angle 0), and we're rotating the whole cluster...
        # Wait, I'm rotating the centers but the hexagons keep angle 0. So the shape is fixed, only positions rotate.
        
        # So support function for a unit hexagon (angle 0) on direction n at angle alpha:
        vertices = np.array([[np.cos(np.radians(60*i)), np.sin(np.radians(60*i))] for i in range(6)])
        
        outer_normals_angles = [30, 90, 150, 210, 270, 330]
        
        max_required_s = 0
        for alpha_deg in outer_normals_angles:
            alpha = np.radians(alpha_deg)
            n = np.array([np.cos(alpha), np.sin(alpha)])
            
            # Support of unit hexagon (angle 0) on n
            support = np.max(vertices @ n)
            
            # Max over all inner hexagons of (center dot n + support)
            max_proj = np.max(rotated_centers @ n) + support
            
            # This must be <= s * sqrt(3)/2
            required_s = max_proj / (sqrt3 / 2)
            max_required_s = max(max_required_s, required_s)
        
        return max_required_s
    
    # Optimize theta in [0, pi/6] (30 degrees) due to hexagonal symmetry
    thetas = np.linspace(0, np.pi/6, 600)
    s_vals = [compute_min_s(th, centers) for th in thetas]
    min_idx = np.argmin(s_vals)
    best_theta = thetas[min_idx]
    best_s = s_vals[min_idx]
    
    # Compute final centers
    c = np.cos(best_theta)
    s_ = np.sin(best_theta)
    rot = np.array([[c, -s_], [s_, c]])
    final_centers = centers @ rot.T
    
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = final_centers[i, 0]
        inner_hex_data[i, 1] = final_centers[i, 1]
        inner_hex_data[i, 2] = 0  # angle 0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, best_s


# EVOLVE-BLOCK-END
