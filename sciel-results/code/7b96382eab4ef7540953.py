import math

N = 11

def construct():
    # Place 11 unit hexagons on a hexagonal lattice (axial coordinates).
    # Axial -> cartesian for pointy-top unit hexagons:
    #   x = sqrt(3) * (q + r/2)
    #   y = 1.5 * r
    # The 11 positions form a compact cluster:
    #   center (0,0)
    #   ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    #   ring 2 (4 corners): (2,-1), (1,2), (-2,1), (-1,-2)
    # These 4 are chosen to keep the cluster as small as possible while reaching ring 2.
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 2), (-2, 1), (-1, -2),
    ]
    
    inner = []
    sqrt3 = math.sqrt(3)
    for q, r in axial_coords:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
    
    # Compute the minimal bounding regular hexagon.
    # The vertices of the placed hexagons have max radius 3.0 (the ring-2 corners).
    # For a regular hexagon of side S centered at origin, the apothem (inradius) is S*sqrt(3)/2.
    # The farthest points of the unit hexagons from origin are at distance 3.0 (their vertices).
    # Actually, let's compute the max distance from origin to any vertex of any inner hexagon.
    # A unit hexagon centered at (cx, cy) has vertices at distance 1.0 from its center.
    # So the max distance from origin to a vertex is max(|center| + 1.0).
    # The ring-2 hexagons are at distance 3.0 from origin (e.g., (2,-1): x=sqrt(3)*(2-0.5)=1.5*sqrt(3), y=1.5*(-1)=-1.5; dist=sqrt(6.75+2.25)=sqrt(9)=3).
    # So max vertex distance = 3.0 + 1.0 = 4.0.
    # For a regular hexagon of side S, the circumradius (center to vertex) is S.
    # So we need S >= 4.0.
    # But wait, the hexagons are rotated by 0 degrees. Their vertices are at 0, 60, 120, 180, 240, 300 degrees from their centers.
    # The outer hexagon can also be rotated. Let's just use S=4.0 with angle 0 for simplicity, or compute tighter.
    # Actually, the bounding hexagon doesn't need to be axis-aligned. Let's just use S=4.0.
    
    center = (0.0, 0.0)
    side = 4.0
    angle = 0.0
    
    return inner, center, side, angle
