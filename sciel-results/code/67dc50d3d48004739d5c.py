# EVOLVE-BLOCK-START
import numpy as np


def _hex_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon (center cx,cy, circumradius s, rotation angle_deg)."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length to maximize 1/side.

    Uses a hexagonal cluster arrangement: 1 center + 6 first-ring + 4 second-ring hexagons,
    with the minimal enclosing regular hexagon computed via the support function.
    """
    # Pointy-top unit hexagons (angle 90 deg means a vertex points up)
    # Hex lattice: for pointy-top, neighbors are at 60-degree increments, distance = 2 * (sqrt(3)/2) * 1? 
    # Actually for pointy-top unit circumradius hexagons, edge length = 1, 
    # horizontal spacing between centers in same row = sqrt(3), vertical = 1.5
    # But for a hexagonal cluster, let's use axial coordinates.
    
    # Use flat-top orientation for cleaner hexagonal symmetry.
    # Flat-top: angle 0 means flat edge on top. Vertex at angle 0 is to the right.
    # For flat-top unit hexagons, the lattice neighbors are at:
    # (1.5, sqrt(3)/2), (0, sqrt(3)), etc. Distance between adjacent centers = sqrt(3) for edge-sharing?
    # No: for unit edge length hexagons, adjacent center distance = 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732
    # Wait, that's for edge-to-edge. Let me reconsider.
    
    # Unit regular hexagon: side length = 1, circumradius = 1, apothem = sqrt(3)/2.
    # Two edge-sharing hexagons: center distance = 2 * apothem = sqrt(3).
    
    # Hexagonal cluster of 11: center at (0,0), 6 at distance sqrt(3) at 60-degree increments,
    # 4 more at distance 2*sqrt(3) in 4 of the 6 directions (to make 11 = 1+6+4).
    
    # Let's use flat-top hexagons (angle 0).
    # Axial to pixel for flat-top: x = 1.5 * q, y = sqrt(3)/2 * q + sqrt(3) * r
    # For hex cluster radius 2 in axial coords (q,r):
    # Center: (0,0)
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1) -> 6 hexes
    # Ring 2: (2,0), (2,-1), (1,-2), (-1,2), (-2,1), (-2,0) -> but we only need 4 more to get 11.
    # 1+6 = 7, need 4 more from ring 2.
    
    # Actually, the standard "hexagon of hexagons" with side length 2 has 1+6+12=19.
    # For 11, we need a different shape. Let me use: center + 6 ring1 + 4 from ring2.
    
    # Let me just place them explicitly in a compact arrangement.
    # Use pointy-top for the cluster, which gives a nice hexagonal shape.
    
    # For pointy-top unit hexagons (vertex up), the lattice is:
    # Row spacing: 1.5 (vertical), column spacing: sqrt(3)
    # Let me define centers using axial coordinates converted to pixel.
    
    sqrt3 = np.sqrt(3)
    
    # Axial coordinates (q, r) for the 11 hexagons forming a compact cluster:
    # This is a hexagon with "radius" 1 in terms of the 6-neighbor ring, plus 4 extra.
    # Actually, let me use the arrangement that minimizes the bounding hexagon.
    
    # Optimal arrangement for 11 unit hexagons in a hexagonal container:
    # It's known that the minimal enclosing hexagon for 11 unit hexagons has side ~3.
    # Let me construct: center at origin, 6 around it at 60-degree spacing (distance sqrt(3)),
    # and 4 more at the "corners" of the ring.
    
    # Centers for 11 hexagons (pointy-top, but we'll use angle for the hexagon orientation)
    # Let's use flat-top hexagons with angle 0.
    
    # Flat-top hex lattice: adjacent centers at distance sqrt(3) in 6 directions at 0, 60, 120, 180, 240, 300 degrees.
    d = sqrt3  # distance between adjacent hexagon centers
    
    # 11 centers:
    centers = []
    # Center
    centers.append([0.0, 0.0])
    # 6 first-ring neighbors
    for i in range(6):
        angle = np.radians(60 * i)
        centers.append([d * np.cos(angle), d * np.sin(angle)])
    # 4 second-ring: at angles 0, 120, 240, and... let's pick 4 that are compact.
    # Second ring at distance 2*d in directions 0, 60, 120, 180, 240, 300.
    # Pick 4: let's use 0, 120, 240, and one more. Actually for symmetry, use 4 that form a pattern.
    # To minimize bounding hexagon, pick 4 from the 6 second-ring positions.
    # Let's try: second ring at angles 60, 180, 300 (3) + one more? No, need 4.
    # Let me just use all 6 second-ring? That's 1+6+6=13, too many.
    # 11 = 1 + 6 + 4. So 4 of the 6 second-ring positions.
    
    # For minimal bounding hexagon, pick the 4 second-ring positions that are most "inward" 
    # relative to the 6-fold symmetry. Actually, any 4 will break symmetry.
    # Let me try a different arrangement: not a perfect hex cluster.
    
    # Alternative: use a 3-4-3-1 arrangement or similar.
    # Let me just compute the bounding hexagon for the 1+6+4 arrangement and see.
    
    # Second ring at distance 2*d, pick angles 0, 120, 240, and 60 (or some 4)
    # Let me try angles 0, 120, 240, 300 (4 positions)
    second_ring_angles = [0, 120, 240, 300]
    for angle_deg in second_ring_angles:
        angle = np.radians(angle_deg)
        centers.append([2 * d * np.cos(angle), 2 * d * np.sin(angle)])
    
    centers = np.array(centers)
    
    # Now compute the minimal enclosing regular hexagon (centered at origin, aligned with flat-top).
    # For a flat-top hexagon centered at origin with circumradius R, the vertices are at:
    # (R, 0), (R/2, R*sqrt(3)/2), (-R/2, R*sqrt(3)/2), (-R, 0), (-R/2, -R*sqrt(3)/2), (R/2, -R*sqrt(3)/2)
    # The side length = R (for a regular hexagon, side = circumradius).
    # Wait, for a regular hexagon, side length = circumradius. Yes.
    
    # The hexagon contains a point (x,y) if it's inside all 6 half-planes defined by the edges.
    # For flat-top hex with circumradius R, the 6 edge normals are at angles 30, 90, 150, 210, 270, 330 degrees.
    # The apothem = R * cos(30) = R * sqrt(3)/2.
    # A point (x,y) is inside if |dot((x,y), normal_i)| <= apothem for each of the 6 normals.
    
    # So the minimal R is: max over all points of (|dot(point, normal_i)| / apothem_ratio) 
    # where apothem = R * sqrt(3)/2, so R >= |dot(point, normal_i)| * 2/sqrt(3) for each i.
    
    normals = []
    for i in range(6):
        angle = np.radians(30 + 60 * i)
        normals.append([np.cos(angle), np.sin(angle)])
    normals = np.array(normals)
    
    # For each point and each normal, compute the required R
    # R_i >= |dot(p, n_i)| / (sqrt(3)/2) = |dot(p, n_i)| * 2/sqrt(3)
    
    max_required_R = 0.0
    for c in centers:
        dots = np.abs(centers @ normals.T)  # This is wrong, let me fix
        break
    
    max_required_R = 0.0
    for c in centers:
        dots = np.abs(c @ normals.T)  # shape (6,)
        required_R = dots * 2.0 / sqrt3
        max_required_R = max(max_required_R, np.max(required_R))
    
    # But we also need to account for the hexagon's own extent (circumradius 1).
    # Each inner hexagon has vertices at distance 1 from its center.
    # So the effective "point" to check is the center plus the hexagon's vertices.
    # Actually, we need to check all vertices of all inner hexagons.
    
    # Let me recompute using all vertices of all inner hexagons.
    all_vertices = []
    for c in centers:
        # Flat-top unit hexagon vertices (circumradius 1, angle 0)
        for i in range(6):
            angle = np.radians(60 * i)
            all_vertices.append([c[0] + np.cos(angle), c[1] + np.sin(angle)])
    all_vertices = np.array(all_vertices)
    
    max_required_R = 0.0
    for v in all_vertices:
        dots = np.abs(v @ normals.T)
        required_R = dots * 2.0 / sqrt3
        max_required_R = max(max_required_R, np.max(required_R))
    
    outer_side = max_required_R  # side length = circumradius for regular hexagon
    
    # Build inner hex data: (x, y, angle_deg)
    inner_hex_data = np.zeros((11, 3))
    inner_hex_data[:, 0] = centers[:, 0]
    inner_hex_data[:, 1] = centers[:, 1]
    inner_hex_data[:, 2] = 0.0  # flat-top, angle 0
    
    # Outer hex data: centered at origin, flat-top, angle 0
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, outer_side


# EVOLVE-BLOCK-END
