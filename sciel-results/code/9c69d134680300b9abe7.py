"""Packed hexagons in a hexagon - hexagonal cluster approach"""
import numpy as np

def construct():
    n = 11  # number of inner hexagons
    
    # Generate hexagonal cluster positions
    # For a hexagonal cluster of radius r, we have 1 + 3r(r+1) hexagons
    # We'll try different radii and pick the best fit for n
    
    # First, find the minimal r such that 1+3r(r+1) >= n
    r = 0
    while 1 + 3*r*(r+1) < n:
        r += 1
    
    # Generate all positions in hexagonal grid with radius <= r
    # Using axial coordinates (q, s) where q+s+t=0, t=-q-s
    positions = []
    for q in range(-r, r+1):
        for s in range(max(-r, -q-r), min(r, -q+r)+1):
            positions.append((q, s))
    
    # We have more positions than needed, take the first n (centered)
    # Sort by distance from center to get a nice cluster
    def hex_dist(q, s):
        t = -q - s
        return (abs(q) + abs(s) + abs(t)) / 2
    
    positions.sort(key=lambda p: hex_dist(p[0], p[1]))
    positions = positions[:n]
    
    # Convert axial coords to Cartesian for flat-top hexagons with side 1
    # Center-to-center distance = sqrt(3) for adjacent hexagons
    # For flat-top: x = sqrt(3) * (q + s/2), y = 1.5 * s
    # Actually, let me use pointy-top which might be more natural
    # For pointy-top hexagons with side 1:
    # x = sqrt(3) * (q + s/2), y = 1.5 * s
    # Center to center distance is sqrt(3)
    
    # Let's use flat-top orientation
    # For flat-top: 
    # x = 1.5 * q, y = sqrt(3) * (s + q/2)
    # This gives center-to-center distance of sqrt(3)
    
    inner = []
    for q, s in positions:
        x = 1.5 * q
        y = np.sqrt(3) * (s + q / 2.0)
        angle = 0.0  # flat-top orientation
        inner.append([x, y, angle])
    
    # Compute bounding regular hexagon
    # The outer hexagon should be centered at origin, flat-top, and contain all inner hexagons
    # Each inner hexagon has vertices at distance 1 from its center
    # So we need the outer hexagon to contain all points at distance <= 1 from each inner center
    
    # For a flat-top regular hexagon with side L centered at origin:
    # The constraint for a point (px, py) to be inside is:
    # |px| <= L (for the left/right flats)
    # |py| <= L * sqrt(3) / 2 (for top/bottom... wait no)
    
    # Actually for flat-top hexagon with side L:
    # Vertices at (L,0), (L/2, L*sqrt(3)/2), (-L/2, L*sqrt(3)/2), (-L,0), (-L/2,-L*sqrt(3)/2), (L/2,-L*sqrt(3)/2)
    # The hexagon is the intersection of 6 half-planes
    # For flat-top: 
    #   px <= L
    #   px >= -L  
    #   sqrt(3)*px + 2*py <= 2*L*sqrt(3)/2 = L*sqrt(3)  ... let me derive properly
    
    # The 6 sides of a flat-top hexagon with circumradius R (= L for unit side):
    # Side 1 (right): x = R
    # Side 2 (upper right): x/2 + y*sqrt(3)/2 = R  =>  x + y*sqrt(3) = 2R
    # Side 3 (upper left): -x/2 + y*sqrt(3)/2 = R  =>  -x + y*sqrt(3) = 2R
    # Side 4 (left): x = -R
    # Side 5 (lower left): -x/2 - y*sqrt(3)/2 = R  =>  -x - y*sqrt(3) = 2R
    # Side 6 (lower right): x/2 - y*sqrt(3)/2 = R  =>  x - y*sqrt(3) = 2R
    
    # So for a point to be inside, all 6 inequalities must hold:
    # x <= R
    # -x <= R  => x >= -R
    # x + y*sqrt(3) <= 2R
    # -x + y*sqrt(3) <= 2R
    # -x - y*sqrt(3) <= 2R  => x + y*sqrt(3) >= -2R
    # x - y*sqrt(3) <= 2R  => -x + y*sqrt(3) >= -2R
    
    # Combining:
    # |x| <= R
    # |x + y*sqrt(3)| <= 2R
    # |-x + y*sqrt(3)| <= 2R  => |x - y*sqrt(3)| <= 2R
    
    # Wait, let me re-derive. For flat-top hex with side s (circumradius = s):
    # The 6 supporting lines are at distance s*sqrt(3)/2 (inradius) from center
    # Normal directions at angles 0, 60, 120, 180, 240, 300 degrees
    
    # Inradius = s * sqrt(3)/2
    # For normal at angle theta, the constraint is: x*cos(theta) + y*sin(theta) <= inradius
    
    # So:
    # theta=0: x <= s*sqrt(3)/2
    # theta=60: x/2 + y*sqrt(3)/2 <= s*sqrt(3)/2  => x + y*sqrt(3) <= s*sqrt(3)
    # theta=120: -x/2 + y*sqrt(3)/2 <= s*sqrt(3)/2  => -x + y*sqrt(3) <= s*sqrt(3)
    # theta=180: -x <= s*sqrt(3)/2  => x >= -s*sqrt(3)/2
    # theta=240: -x/2 - y*sqrt(3)/2 <= s*sqrt(3)/2  => -x - y*sqrt(3) <= s*sqrt(3)  => x + y*sqrt(3) >= -s*sqrt(3)
    # theta=300: x/2 - y*sqrt(3)/2 <= s*sqrt(3)/2  => x - y*sqrt(3) <= s*sqrt(3)  => -x + y*sqrt(3) >= -s*sqrt(3)
    
    # So the constraints are:
    # |x| <= s*sqrt(3)/2
    # |x + y*sqrt(3)| <= s*sqrt(3)
    # |x - y*sqrt(3)| <= s*sqrt(3)
    
    # Each inner hexagon also has extent 1 from its center (circumradius = 1)
    # So for each inner hex center (cx, cy), we need:
    # |cx| + 1 <= s*sqrt(3)/2  (approximately, but this is conservative)
    # More precisely, we need to check the maximum of the 6 linear forms over the inner hex
    
    # For a linear form ax + by, the max over a hexagon centered at (cx,cy) with circumradius 1 is:
    # a*cx + b*cy + max over vertices of (a*dx + b*dy) where (dx,dy) are vertex offsets
    # The max of a linear form over a regular hexagon of circumradius 1 is sqrt(a^2 + b^2) * (something)
    # Actually, for a regular hexagon, the support function in direction (a,b) is:
    # sqrt(3)/2 * sqrt(a^2+b^2) if we normalize... let me just compute numerically
    
    # Simpler: for each inner hex, compute its 6 vertices, then for each of the 6 outer constraints,
    # find the max value of the linear form over all vertices of all inner hexes
    
    sqrt3 = np.sqrt(3)
    
    # For each inner hex center, generate its 6 vertices (flat-top, circumradius 1)
    # Vertices at angles 0, 60, 120, 180, 240, 300 from center
    all_vertices = []
    for x, y, ang in inner:
        for k in range(6):
            theta = np.radians(60 * k)
            vx = x + np.cos(theta)
            vy = y + np.sin(theta)
            all_vertices.append((vx, vy))
    
    # Now find minimal s such that all vertices satisfy:
    # |vx| <= s*sqrt(3)/2
    # |vx + vy*sqrt(3)| <= s*sqrt(3)
    # |vx - vy*sqrt(3)| <= s*sqrt(3)
    
    max_val = 0.0
    for vx, vy in all_vertices:
        val1 = abs(vx) / (sqrt3 / 2)
        val2 = abs(vx + vy * sqrt3) / sqrt3
        val3 = abs(vx - vy * sqrt3) / sqrt3
        max_val = max(max_val, val1, val2, val3)
    
    side = max_val
    
    # Center the outer hexagon at origin
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, side, angle
