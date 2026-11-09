"""Improved packing of 11 unit hexagons inside a regular hexagon."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 100  # safety margin

    sqrt3 = math.sqrt(3)
    inradius = sqrt3 / 2.0

    # Generate all candidate positions in a grid-like arrangement
    # Using a hexagonal lattice for pointy-top hexagons
    # Adjacent hexagons at distance sqrt(3) are tangent

    # For 11 hexagons, use a compact arrangement
    # Hexagonal numbers: 1, 7, 19, 37, ...
    # 11 is between 7 and 19, so use a modified arrangement

    # Strategy: place 1 central hexagon, 6 in first ring, 4 in second ring
    # This gives 11 hexagons in a roughly circular pattern

    # Pointy-top hexagon lattice vectors:
    # Horizontal spacing: 1.5
    # Vertical spacing: sqrt(3)
    # Offset rows shifted by 0.75

    # Let's use a different approach: place hexagons at hexagonal lattice points
    # For pointy-top hexagons with side 1:
    # Neighbors are at:
    # (±1.5, ±sqrt(3)/2) and (0, ±sqrt(3))

    # Central hexagon at origin
    positions = [(0.0, 0.0)]

    # First ring: 6 hexagons at distance sqrt(3)
    for i in range(6):
        angle = i * math.pi / 3.0
        x = sqrt3 * math.cos(angle)
        y = sqrt3 * math.sin(angle)
        positions.append((x, y))

    # Second ring: add 4 more hexagons to reach 11 total
    # The first ring has 6, so we need 4 more
    # Place them at alternating positions in the second ring
    # Second ring hexagons are at distance 2*sqrt(3) from center
    # But we only need 4 of them

    # Actually, let's try a different arrangement
    # Place them in a 3x4 grid pattern or similar

    # Let me try: rows of hexagons
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 1 hexagon (or adjust)
    # Total: 11

    # For pointy-top hexagons:
    # Horizontal spacing within row: 3.0 (side + gap for tangent)
    # Wait, adjacent tangent hexagons have center distance = sqrt(3) * 2 / sqrt(3) ... 
    # Actually for pointy-top, horizontal neighbors are at distance 3.0? No...

    # Let me reconsider. For regular hexagons with side length 1:
    # The distance between centers of two tangent hexagons depends on orientation
    # If both are pointy-top, horizontal neighbors: distance = 2 * inradius = sqrt(3)
    # No wait, inradius = sqrt(3)/2, so distance between centers of tangent 
    # horizontally-aligned pointy-top hexagons = 2 * inradius = sqrt(3)

    # Actually I need to be more careful. Let me use a known compact arrangement.

    # Simple approach: linear arrangement won't be optimal, but let's try
    # a compact 2D arrangement

    # Use a hexagonal packing pattern
    # For N=11, use positions that minimize the enclosing hexagon

    # Let's define positions on a triangular lattice scaled appropriately
    # Hexagons of side 1 have inradius sqrt(3)/2
    # For non-overlapping, center distance >= 2*inradius = sqrt(3)

    # Place on a hexagonal lattice with spacing sqrt(3)
    positions = []
    
    # Define basis vectors for hexagonal lattice
    a1 = (sqrt3, 0.0)
    a2 = (sqrt3/2, 1.5)  # This needs to give distance sqrt(3) from neighbors
    
    # Actually, let me just use explicit coordinates for a compact 11-hex arrangement
    # Central + 6 around + 4 more in a cross pattern

    # Position 0: center
    positions = [(0.0, 0.0)]
    
    # Positions 1-6: first ring (hexagonal)
    for i in range(6):
        theta = i * math.pi / 3.0
        positions.append((sqrt3 * math.cos(theta), sqrt3 * math.sin(theta)))
    
    # Positions 7-10: extend outward in 4 directions
    # Use directions that fill gaps
    for i in range(4):
        theta = i * math.pi / 2.0 + math.pi / 4.0  # diagonal directions
        r = 2 * sqrt3
        positions.append((r * math.cos(theta), r * math.sin(theta)))

    inner = [[x, y, 0.0] for x, y in positions[:N]]

    # Compute minimal enclosing regular hexagon side length
    # For a regular hexagon with side s, the inradius is s*sqrt(3)/2
    # The support function in direction theta is s*sqrt(3)/2 for the inradius
    # But for the circumradius it's s
    # Actually, for a regular hexagon centered at origin with one vertex at angle 0:
    # The vertices are at distance s from center at angles 0, 60, 120, 180, 240, 300
    # The inradius (distance to edge midpoint) is s*sqrt(3)/2
    # The support function h(theta) = max_i v_i · u(theta) where v_i are vertices
    # For containment, we need: for all inner hex vertices, they lie inside outer hex
    # Equivalently, for each of the 6 outward normals of the outer hex, 
    # the projection of all inner hex vertices must be <= outer inradius

    # Inner hexagon vertices for a hexagon at (cx, cy) with rotation angle_deg:
    # Vertices at distance 1 from center at angles angle_deg + k*60 for k=0..5
    # Plus offset by (cx, cy)

    def get_hex_vertices(cx, cy, angle_deg):
        verts = []
        for k in range(6):
            theta = math.radians(angle_deg + k * 60)
            vx = cx + math.cos(theta)
            vy = cy + math.sin(theta)
            verts.append((vx, vy))
        return verts

    # For a regular hexagon with side s centered at (ox, oy) with rotation phi,
    # the 6 outward normal directions are at angles phi + k*60 + 30 for k=0..5
    # The inradius is s*sqrt(3)/2
    # A point (x,y) is inside if for all k: (x-ox)*cos(nk) + (y-oy)*sin(nk) <= s*sqrt(3)/2

    # We want to find minimum s such that all inner hex vertices are inside
    # This is equivalent to: s >= max over all inner vertices v and all 6 normal directions n:
    #   (v_x - ox)*n_x + (v_y - oy)*n_y) / (sqrt(3)/2)
    # assuming center at (ox, oy) = (0,0) for simplicity

    # Collect all vertices of all inner hexagons
    all_vertices = []
    for x, y, ang in inner:
        verts = get_hex_vertices(x, y, ang)
        all_vertices.extend(verts)

    # For outer hexagon centered at origin with rotation 0,
    # normals are at angles 30, 90, 150, 210, 270, 330 degrees
    max_proj = 0.0
    for k in range(6):
        theta = math.radians(30 + k * 60)
        nx, ny = math.cos(theta), math.sin(theta)
        for vx, vy in all_vertices:
            proj = vx * nx + vy * ny
            if proj > max_proj:
                max_proj = proj

    side = max_proj / inradius
    side = math.ceil(side * 10000) / 10000  # round up slightly

    return inner, (0.0, 0.0), side, 0.0
# EVOLVE-BLOCK-END
