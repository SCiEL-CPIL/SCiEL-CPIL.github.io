# EVOLVE-BLOCK-START
import numpy as np
import time

def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon.
    Returns (inner, outer, s) where:
      inner: (12, 3) array of (x, y, angle_deg) for each inner hexagon
      outer: (3,) array (x, y, angle_deg) for the outer hexagon
      s: side length of the outer hexagon
    """
    start_time = time.time()
    deadline = 620.0  # Safety margin before 630s limit

    sqrt3 = np.sqrt(3.0)

    # Optimal configuration: 12 hexagons in a hexagonal cluster pattern
    # 1 center + 6 in first ring + 5 in second ring (or similar)
    # For unit hexagons with flat-top orientation (angle=0):
    #   Width = 2, Height = sqrt(3)
    #   Horizontal neighbor distance = 3 (vertex to vertex, but with offset for hex lattice)
    #   For hexagonal lattice of flat-top hexagons:
    #     v1 = (3, 0) for horizontal
    #     v2 = (1.5, sqrt(3)) for diagonal
    
    # Actually, for non-overlapping unit hexagons (side=1):
    # Two hexagons overlap if their center distance is less than the minimum
    # distance between edges. For flat-top hexagons:
    # - Horizontally adjacent (sharing vertical edge): center distance = 2 * apothem = sqrt(3)
    #   Wait no. Apothem = sqrt(3)/2. Two hexagons sharing an edge have center distance = 2*apothem = sqrt(3).
    #   But they share an edge, so they're touching. For non-overlapping, we need distance >= sqrt(3) for edge-sharing.
    #   
    # Let me reconsider. For flat-top unit hexagons:
    # The hexagon has vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
    # 
    # Two flat-top hexagons can be adjacent in 6 directions. The center-to-center distance
    # for edge-sharing adjacency is 2 * apothem = sqrt(3) ≈ 1.732.
    # 
    # But we need them to NOT overlap (can touch). So minimum center distance depends on direction.
    # In the direction perpendicular to an edge (normal direction), min distance = 2*apothem = sqrt(3).
    # In the direction along an edge, the constraint is different.
    #
    # Actually, the correct way: two convex polygons don't overlap iff there exists a separating axis.
    # For two regular hexagons of the same orientation, the minimum center distance for non-overlap
    # in any direction is the width in that direction.
    #
    # The width of a unit flat-top hexagon in direction θ is:
    #   w(θ) = 2 * max over vertices of |v · n| where n is unit vector in direction θ
    #   Actually, width = max(v·n) - min(v·n) for vertices v.
    #
    # For flat-top hexagon with vertices at angles 0,60,120,180,240,300 (unit radius):
    #   In x-direction: max x = 1, min x = -1, width = 2
    #   In y-direction: max y = sqrt(3)/2, min y = -sqrt(3)/2, width = sqrt(3)
    #   In direction 30°: n = (cos30, sin30) = (sqrt(3)/2, 1/2)
    #     v·n for vertices:
    #       (1,0)·n = sqrt(3)/2
    #       (0.5, sqrt(3)/2)·n = 0.5*sqrt(3)/2 + sqrt(3)/2*1/2 = sqrt(3)/4 + sqrt(3)/4 = sqrt(3)/2
    #       (-0.5, sqrt(3)/2)·n = -sqrt(3)/4 + sqrt(3)/4 = 0
    #       (-1,0)·n = -sqrt(3)/2
    #       (-0.5,-sqrt(3)/2)·n = -sqrt(3)/4 - sqrt(3)/4 = -sqrt(3)/2
    #       (0.5,-sqrt(3)/2)·n = sqrt(3)/4 - sqrt(3)/4 = 0
    #     max = sqrt(3)/2, min = -sqrt(3)/2, width = sqrt(3)
    #
    # So the minimum separation in direction θ is w(θ)/2 from center to edge.
    # For two hexagons with centers c1, c2, they don't overlap iff for all axes n:
    #   |(c2-c1)·n| >= w(n)  ... no, that's not right either.
    #
    # SAT: Two convex polygons are separated iff there exists an axis n such that
    # the projections of the two polygons onto n don't overlap.
    # For two identical regular hexagons with the same orientation and centers c1, c2:
    # The projection of hexagon i onto axis n is [c_i·n - r_i(n), c_i·n + r_i(n)] where r_i(n) is the support function.
    # They are separated iff for some n: |c1·n - c2·n| > 2*r(n)
    # i.e., |(c2-c1)·n| > 2*r(n) for some n.
    #
    # The support function r(n) of a unit flat-top hexagon in direction n = (cos θ, sin θ) is:
    #   r(θ) = max_k (v_k · n) where v_k are the 6 vertices.
    #
    # For the hexagonal lattice, the tightest constraints are along the 6 edge normals.
    # For flat-top hexagons, edge normals are at angles 30°, 90°, 150°, 210°, 270°, 330°.
    # The support in these directions:
    #   At 90° (y-direction): r = sqrt(3)/2
    #   At 30°: r = sqrt(3)/2 (computed above)
    #   By symmetry, all 6 edge normals give r = sqrt(3)/2.
    #
    # But what about other directions? At 0° (x-direction): r = 1.
    # At 15°: n = (cos15, sin15)
    #   (1,0)·n = cos15 ≈ 0.966
    #   (0.5, sqrt(3)/2)·n = 0.5*cos15 + sqrt(3)/2*sin15 = 0.5*0.966 + 0.866*0.259 = 0.483 + 0.224 = 0.707
    #   So r(15°) = cos15 ≈ 0.966
    #
    # The maximum of r(θ) is 1 (at θ=0°, 60°, etc.) and minimum is sqrt(3)/2 (at θ=30°, 90°, etc.)
    #
    # For two hexagons to NOT overlap, we need:
    #   For all θ: |(c2-c1)·n(θ)| <= 2*r(θ)  ... no wait.
    #   They DON'T overlap iff there EXISTS some θ such that |(c2-c1)·n(θ)| > 2*r(θ).
    #   They DO overlap iff for ALL θ: |(c2-c1)·n(θ)| <= 2*r(θ).
    #
    # So the minimum distance for non-overlap in direction d = c2-c1 is:
    #   d_min = min over directions of the threshold where separation occurs.
    #
    # This is getting complex. Let me just use a known good configuration.
    
    # Known optimal: hexagonal close packing of 12 hexagons.
    # Pattern: rows of 4, 4, 4 with hexagonal offset, OR a cluster.
    # 
    # Let me try the cluster: 1 center, 6 around it, 5 more.
    # For flat-top hexagons, the 6 neighbors are at distance sqrt(3) in directions 0°, 60°, 120°, 180°, 240°, 300°.
    # Wait, that's not right. The 6 neighbors in a hexagonal tiling are at angles 0°, 60°, ... but at what distance?
    # In a hexagonal tiling of unit hexagons, adjacent centers are at distance 2*apothem = sqrt(3).
    # The 6 neighbors are at angles 0°, 60°, 120°, 180°, 240°, 300° from the center, at distance sqrt(3).
    
    # Let's use this hexagonal lattice:
    # Center at (0,0)
    # Ring 1: 6 hexagons at distance sqrt(3) at angles 0, 60, 120, 180, 240, 300
    # Ring 2: 6 hexagons at distance 2*sqrt(3) at angles 0, 60, 120, 180, 240, 300
    # But that's 13 total. We need 12.
    # 
    # Alternative: 1 + 6 + 5 = 12. Use 5 from ring 2.
    # Or use a different arrangement.
    
    # Let me try: rows of 4, 4, 4 in hexagonal pattern.
    # For flat-top hexagons in a hexagonal lattice:
    # Row j: centers at x = i*3 + (j%2)*1.5, y = j*sqrt(3) for i=0,1,2,3
    # This gives 12 hexagons in a 4x3 grid with hexagonal offset.
    
    # Let me verify non-overlap for this configuration.
    # Two hexagons in the same row: distance = 3 in x. 
    # Width in x-direction is 2, so gap = 3-2 = 1 > 0. OK.
    # Two hexagons in adjacent rows: 
    #   Row 0: (0,0), (3,0), (6,0), (9,0)
    #   Row 1: (1.5, sqrt(3)), (4.5, sqrt(3)), (7.5, sqrt(3)), (10.5, sqrt(3))
    #   Distance from (0,0) to (1.5, sqrt(3)): sqrt(1.5^2 + 3) = sqrt(2.25+3) = sqrt(5.25) ≈ 2.29
    #   Distance from (0,0) to (-1.5, sqrt(3)) doesn't exist in this setup.
    #   What about (0,0) and (1.5, sqrt(3))? Need to check SAT.
    #   d = (1.5, sqrt(3)), |d| = sqrt(5.25) ≈ 2.291
    #   Direction of d: angle = atan2(sqrt(3), 1.5) = atan2(1.732, 1.5) ≈ 49.1°
    #   Support in this direction: need r(49.1°)
    #   n = (cos49.1°, sin49.1°) ≈ (0.654, 0.756)
    #   Vertices: (1,0), (0.5,0.866), (-0.5,0.866), (-1,0), (-0.5,-0.866), (0.5,-0.866)
    #   (1,0)·n = 0.654
    #   (0.5,0.866)·n = 0.327 + 0.655 = 0.982
    #   (-0.5,0.866)·n = -0.327 + 0.655 = 0.328
    #   (-1,0)·n = -0.654
    #   (-0.5,-0.866)·n = -0.327 - 0.655 = -0.982
    #   (0.5,-0.866)·n = 0.327 - 0.655 = -0.328
    #   r = 0.982
    #   2*r = 1.964
    #   |d| = 2.291 > 1.964. So they are separated. Good.
    
    # Let me check the tightest case: (0,0) and (1.5, sqrt(3))
    # We need to verify for ALL axes, not just the direction of d.
    # The critical axes are the edge normals of both hexagons.
    # For flat-top hexagons, edge normals are at 30°, 90°, 150°, 210°, 270°, 330°.
    # 
    # Axis at 90° (y-direction): 
    #   Hex1 projects to [-sqrt(3)/2, sqrt(3)/2] centered at 0: [-0.866, 0.866]
    #   Hex2 projects to [-sqrt(3)/2+sqrt(3), sqrt(3)/2+sqrt(3)] = [0.866, 2.598]
    #   Gap = 0.866 - 0.866 = 0. They touch! 
    #   Wait, that means they're just touching, which is OK (non-overlapping).
    #   But the problem says "two inner hexagons not separated: some edge normal of either one must give projection intervals more than 1e-6 apart"
    #   So touching is NOT allowed? Let me re-read.
    #   "two inner hexagons not separated: some edge normal of either one (separating axis test) must give projection intervals more than 1e-6 apart"
    #   This means: for the hexagons to be validly separated, there must exist an axis where the projections are more than 1e-6 apart.
    #   If they just touch, the gap is 0, which is NOT more than 1e-6. So they would be INVALID.
    #   We need strict separation with gap > 1e-6.
    
    # So I need to add a small epsilon to the spacing.
    # Let me scale the lattice by a factor S > 1 to ensure separation.
    
    S = 1.0 + 2e-6  # Small scaling to ensure separation
    
    # Hexagonal lattice centers (before centering):
    centers = []
    for j in range(3):
        for i in range(4):
            x = (i * 3.0 + (j % 2) * 1.5) * S
            y = j * sqrt3 * S
            centers.append((x, y))
    
    # Center the arrangement
    xs = np.array([c[0] for c in centers])
    ys = np.array([c[1] for c in centers])
    cx_mean = (xs.min() + xs.max()) / 2.0
    cy_mean = (ys.min() + ys.max()) / 2.0
    
    centers_centered = [(x - cx_mean, y - cy_mean) for x, y in centers]
    
    # All hexagons have angle 0 (flat-top)
    angles = [0.0] * 12
    
    # Compute minimal containing regular hexagon (flat-top, centered at origin)
    # For a point (vx, vy) to be inside a flat-top regular hexagon with side s centered at origin:
    # The 6 constraints are:
    #   s >= vx + vy/sqrt(3)
    #   s >= 2*vy/sqrt(3)  
    #   s >= vy/sqrt(3) - vx
    #   s >= -vy/sqrt(3) - vx
    #   s >= -2*vy/sqrt(3)
    #   s >= vx - vy/sqrt(3)
    
    required_s = 0.0
    for idx, (cx, cy) in enumerate(centers_centered):
        for k in range(6):
            angle_rad = np.deg2rad(k * 60)
            vx = cx + np.cos(angle_rad)
            vy = cy + np.sin(angle_rad)
            
            c1 = vx + vy / sqrt3
            c2 = 2 * vy / sqrt3
            c3 = vy / sqrt3 - vx
            c4 = -vy / sqrt3 - vx
            c5 = -2 * vy / sqrt3
            c6 = vx - vy / sqrt3
            
            s_min_point = max(c1, c2, c3, c4, c5, c6)
            required_s = max(required_s, s_min_point)
    
    # Add margin for numerical safety
    s = required_s + 1e-9
    s = max(s, np.sqrt(12) + 1e-9)
    
    # Build output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = centers_centered[i][0]
        inner_hex_data[i, 1] = centers_centered[i][1]
        inner_hex_data[i, 2] = angles[i]
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s

# EVOLVE-BLOCK-END
