import math

def construct():
    # Arrangement: 11 unit hexagons in a dense hexagonal lattice (flat-top)
    # Centers are placed on a triangular lattice with spacing slightly larger than sqrt(3)
    # to ensure strict disjointness (no touching) due to evaluator strictness.
    
    sqrt3 = math.sqrt(3.0)
    epsilon = 0.01
    D = sqrt3 + epsilon # Spacing between centers
    
    # 11 centers forming a compact cluster
    # Axial coordinates (q, r)
    axial_coords = [
        (0, 0),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (-1, -1),
        (2, 0),
        (-2, 0),
        (0, 2),
        (0, -2)
    ]
    
    inner = []
    centers = []
    for q, r in axial_coords:
        x = D * (q + r / 2.0)
        y = D * (sqrt3 / 2.0) * r
        centers.append((x, y))
        # Inner hexagons are flat-top (angle 0)
        inner.append([x, y, 0.0])
        
    # Calculate the required outer side length
    # The outer hexagon must contain all vertices of the inner hexagons.
    # Inner hexagon vertices relative to center (flat-top, side 1):
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    
    max_r_sq = 0.0
    h_half = sqrt3 / 2.0
    for cx, cy in centers:
        # Check all 6 vertices
        # 1
        dx = cx + 1.0
        dy = cy
        r_sq = dx*dx + dy*dy
        if r_sq > max_r_sq: max_r_sq = r_sq
        
        # 2
        dx = cx + 0.5
        dy = cy + h_half
        r_sq = dx*dx + dy*dy
        if r_sq > max_r_sq: max_r_sq = r_sq
        
        # 3
        dx = cx - 0.5
        dy = cy + h_half
        r_sq = dx*dx + dy*dy
        if r_sq > max_r_sq: max_r_sq = r_sq
        
        # 4
        dx = cx - 1.0
        dy = cy
        r_sq = dx*dx + dy*dy
        if r_sq > max_r_sq: max_r_sq = r_sq
        
        # 5
        dx = cx - 0.5
        dy = cy - h_half
        r_sq = dx*dx + dy*dy
        if r_sq > max_r_sq: max_r_sq = r_sq
        
        # 6
        dx = cx + 0.5
        dy = cy - h_half
        r_sq = dx*dx + dy*dy
        if r_sq > max_r_sq: max_r_sq = r_sq

    # The circumradius of the outer hexagon must be at least max_r
    # Add a small margin to ensure "inside" strictly
    side = math.sqrt(max_r_sq) + 0.01
    
    center = (0.0, 0.0)
    outer_angle = 0.0
    
    return inner, center, side, outer_angle
