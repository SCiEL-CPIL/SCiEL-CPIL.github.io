"""Packing 11 unit hexagons in a regular hexagon - hexagonal lattice cluster."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle_rad + math.pi * i / 3
        x = cx + side * math.cos(a)
        y = cy + side * math.sin(a)
        verts.append((x, y))
    return verts

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    angle_rad = math.radians(angle_deg)
    # Rotate point into hexagon frame
    cos_a = math.cos(-angle_rad)
    sin_a = math.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    # Regular hexagon with side s, flat-top orientation:
    # |x| <= s, |y| <= s*sqrt(3)/2, |x|/s + |y|/(s*sqrt(3)/2) <= 1 (for the 6 edges)
    # Actually for a hexagon with vertices at angles 0,60,120,180,240,300:
    # The hexagon is defined by 6 half-planes. For flat-top (vertex at angle 0):
    # |x*cos(60*k) + y*sin(60*k)| <= s*cos(30) = s*sqrt(3)/2 for k=0..5
    # Wait, let me use the standard: for hex with vertex at angle 0, side s:
    # The 6 edge normals are at angles 30, 90, 150, 210, 270, 330
    # Each edge is at distance s*sqrt(3)/2 from center
    apothem = side * math.sqrt(3) / 2
    for k in range(6):
        a = math.radians(30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True

def _hexagons_disjoint(v1, v2, angle1, angle2, tol=1e-6):
    """Check if two unit hexagons are disjoint (no overlap)."""
    # Check if any vertex of hex1 is inside hex2, and vice versa
    verts1 = _hex_vertices(v1[0], v1[1], 1.0, angle1)
    verts2 = _hex_vertices(v2[0], v2[1], 1.0, angle2)
    for vx, vy in verts1:
        if _point_in_hexagon(vx, vy, v2[0], v2[1], 1.0, angle2):
            return False
    for vx, vy in verts2:
        if _point_in_hexagon(vx, vy, v1[0], v1[1], 1.0, angle1):
            return False
    return True

def _all_inside(inner, outer_center, outer_side, outer_angle, tol=1e-6):
    """Check all inner hexagons are inside outer hexagon."""
    for ix, iy, iangle in inner:
        verts = _hex_vertices(ix, iy, 1.0, iangle)
        for vx, vy in verts:
            if not _point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                return False
    return True

def _check_valid(inner, outer_center, outer_side, outer_angle):
    """Full validity check."""
    # Check disjointness
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner[i], inner[j], inner[i][2], inner[j][2]):
                return False
    # Check all inside outer
    if not _all_inside(inner, outer_center, outer_side, outer_angle):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Hexagonal lattice arrangement for 11 hexagons
    # Lattice coordinates: center at (0,0), first ring at 6 positions, second ring partial
    # For unit hexagons in hexagonal lattice, spacing is 2 in lattice directions
    # Lattice vectors: e0=(2,0), e1=(1,sqrt(3)), e2=(-1,sqrt(3))
    
    sqrt3 = math.sqrt(3)
    
    # Generate hexagonal cluster of 11: center + 6 + 4
    # Using axial coordinates (q, r) where position = q*e0 + r*e1
    # e0 = (2, 0), e1 = (1, sqrt(3))
    # Positions in axial coords for hex cluster radius 1:
    # (0,0) - center
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) - first ring (6)
    # (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1) - second ring (12)
    # We need 11 total: center(1) + first(6) + 4 from second ring
    
    # For maximum compactness, pick 4 from second ring that minimize bounding box
    # By symmetry, pick (2,0), (-1,2), (-2,1), (0,-2) or similar symmetric subset
    
    # Actually, let's use the full hexagonal number pattern:
    # 1 + 6 + 12 = 19 for radius 2. We only need 11.
    # Best: 1 + 6 + 4. Pick 4 from the 12 in second ring.
    # For a hexagonal bounding shape, pick every other one or a symmetric set.
    
    # Let me compute positions for a symmetric 11-hex cluster
    # Axial to cartesian: x = 2*q + r, y = sqrt(3)*r
    # Wait, for hexagonal tiling of unit hexagons:
    # If hexagons have flat sides of length 1 (width=2, height=sqrt(3)),
    # and are arranged in a hexagonal lattice, the centers are at:
    # (2*q + r, sqrt(3)*r) for axial coords (q, r)
    
    # Let me verify: adjacent hexagons should be 2 apart center-to-center
    # (0,0) and (1,0): dist = 2. OK
    # (0,0) and (0,1): dist = sqrt(1+3) = 2. OK
    # (0,0) and (1,-1): dist = sqrt(4+3) = sqrt(7) ≈ 2.65. Not adjacent.
    # Hmm, let me reconsider.
    
    # For regular hexagons of side 1, the distance between centers of adjacent hexagons
    # in a hexagonal tiling is 2 (flat-to-flat distance = sqrt(3), but center-to-center = 2*apothem = 2*sqrt(3)/2... no)
    
    # Actually: regular hexagon side s=1, apothem = sqrt(3)/2, so flat-to-flat = sqrt(3).
    # In a hexagonal tiling, adjacent hexagons share an edge, so center-to-center = 2*apothem = sqrt(3)?
    # No wait. If two hexagons share an edge, the distance between centers is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3).
    
    # Hmm, but that would make the lattice spacing sqrt(3), not 2.
    # Let me reconsider. A regular hexagon with side 1 has:
    # - Width (flat to flat) = sqrt(3)
    # - Height (vertex to vertex) = 2
    # When tiled, adjacent hexagons are separated by center-to-center = sqrt(3) in the 6 lattice directions.
    
    # Lattice vectors for hexagonal tiling of unit hexagons:
    # v0 = (sqrt(3), 0)
    # v1 = (sqrt(3)/2, 3/2)
    # v2 = (-sqrt(3)/2, 3/2)
    # |v0| = sqrt(3), |v1| = sqrt(3/4 + 9/4) = sqrt(3). Good.
    
    # So positions: (q*v0 + r*v1) = (sqrt(3)*q + sqrt(3)/2*r, 3/2*r)
    
    # Let me redo with correct lattice:
    # v0 = (sqrt(3), 0)
    # v1 = (sqrt(3)/2, 3/2)
    
    # Axial (q,r) -> cartesian: x = sqrt(3)*q + sqrt(3)/2*r, y = 3/2*r
    
    # Hex cluster of radius 1 (11 hexagons would be 1+6+4, but full radius 1 is 1+6=7, radius 2 is 1+6+12=19)
    # For 11, use center + first ring (6) + 4 from second ring.
    
    # Second ring positions in axial coords (q,r) with max(|q|,|r|,|q+r|) = 2:
    # (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)... 
    # Wait, let me list all with hex distance 2 from origin:
    # (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)
    # That's 12. Pick 4 symmetrically.
    
    # For a hexagonal bounding shape, pick (2,0), (-2,2), (-2,0), (0,-2) - but that's not symmetric.
    # Better: (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2) - that's 6, giving 13 total.
    # For 11: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0) - 6 from second ring, total 13.
    # Or 4: (2,0), (0,2), (-2,0), (0,-2) - total 11. This gives a cross/diamond shape.
    
    # Let me just compute the bounding hexagon for various configurations and pick the best.
    
    # Configuration: center + 6 first ring + 4 second ring at (2,0), (0,2), (-2,0), (0,-2)
    axial_positions = [(0,0)]  # center
    # First ring (hex distance 1):
    first_ring = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    axial_positions.extend(first_ring)
    # Second ring, pick 4:
    second_ring_4 = [(2,0), (0,2), (-2,0), (0,-2)]
    axial_positions.extend(second_ring_4)
    
    assert len(axial_positions) == 11
    
    # Convert to cartesian
    def axial_to_cart(q, r):
        x = sqrt3 * q + sqrt3 / 2 * r
        y = 3 / 2 * r
        return x, y
    
    inner = []
    for q, r in axial_positions:
        x, y = axial_to_cart(q, r)
        inner.append([x, y, 0.0])
    
    # Now find the minimum outer hexagon side length
    # The outer hexagon should be concentric with the same orientation (angle=0)
    # For a hexagon with vertex at angle 0, the 6 edge normals are at 30, 90, 150, 210, 270, 330
    # A point (x,y) is inside if for all 6 normals n_k: x*cos(n_k) + y*sin(n_k) <= s*sqrt(3)/2
    
    # Find minimum s such that all inner hexagon vertices are inside
    max_required = 0.0
    for ix, iy, iangle in inner:
        verts = _hex_vertices(ix, iy, 1.0, iangle)
        for vx, vy in verts:
            for k in range(6):
                a = math.radians(30 + 60 * k)
                nx = math.cos(a)
                ny = math.sin(a)
                dist = vx * nx + vy * ny
                if dist > max_required:
                    max_required = dist
    
    # s * sqrt(3)/2 >= max_required => s >= max_required * 2/sqrt(3)
    outer_side = max_required * 2 / sqrt3
    
    # Verify validity
    if not _check_valid(inner, (0.0, 0.0), outer_side, 0.0):
        # Add small margin
        outer_side *= 1.001
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return inner, (0.0, 0.0), outer_side, 0.0
    
    return inner, (0.0, 0.0), outer_side, 0.0
# EVOLVE-BLOCK-END
