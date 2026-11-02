"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Place N unit regular hexagons inside a regular hexagon, minimizing outer side length.
    
    Strategy: Use a hexagonal lattice arrangement. Unit hexagons on a hexagonal grid
    with spacing sqrt(3) (for pointy-top or flat-top orientation) can be packed efficiently.
    
    For flat-top hexagons (angle 0), the centers on a hexagonal lattice are:
    - Horizontal spacing: 3/2 * side = 1.5 (for unit hexagons, side=1)
    - Vertical spacing: sqrt(3) * side = sqrt(3)
    - Odd rows offset by 0.75
    
    Wait, let me reconsider. For regular hexagons of side length 1:
    - The distance between centers of adjacent hexagons in a hexagonal packing is sqrt(3)
      (for flat-top orientation, the "radius" to edge midpoint is sqrt(3)/2, so center-to-center
       is 2 * sqrt(3)/2 = sqrt(3))
    
    Actually, for regular hexagons with side s=1:
    - Circumradius (center to vertex) = 1
    - Inradius (center to edge midpoint) = sqrt(3)/2
    - For flat-top hexagons, adjacent centers in the same row are 3/2 apart horizontally? 
      No, that's not right either.
    
    Let me think more carefully. A regular hexagon with side 1 has:
    - Width (flat-top, distance between parallel sides) = sqrt(3)
    - Height (flat-top, distance between opposite vertices) = 2
    
    For hexagonal packing of flat-top hexagons:
    - Centers in the same row: separated by 3/2 * 1 = 1.5? No...
    
    Actually, for two flat-top unit hexagons side by side sharing an edge:
    The shared edge is vertical. The centers are separated horizontally by 2 * inradius = 2 * sqrt(3)/2 = sqrt(3).
    
    So for flat-top hexagons:
    - Horizontal spacing between adjacent centers in same row: sqrt(3)
    - Vertical spacing between rows: 3/2 * 1 = 1.5? No...
    
    Let me use coordinates. Flat-top hexagon vertices (centered at origin):
    (1, 0), (1/2, sqrt(3)/2), (-1/2, sqrt(3)/2), (-1, 0), (-1/2, -sqrt(3)/2), (1/2, -sqrt(3)/2)
    
    Two such hexagons sharing the right edge of the left one (edge from (1/2, sqrt(3)/2) to (1/2, -sqrt(3)/2)):
    The right hexagon's center is at (1 + 1/2, 0) = (3/2, 0)? No, the shared edge is at x=1/2 for the left hex... 
    
    Wait, the right edge of the left hexagon goes from (1/2, sqrt(3)/2) to (1/2, -sqrt(3)/2). For the right hexagon to share this edge, its left edge must be at the same location. The left edge of a flat-top hexagon centered at (cx, cy) goes from (cx - 1/2, cy + sqrt(3)/2) to (cx - 1/2, cy - sqrt(3)/2). So cx - 1/2 = 1/2, meaning cx = 1. So the right hexagon center is at (1, 0).
    
    So horizontal spacing is 1? That can't be right for non-overlapping.
    
    Let me re-examine. The vertices of a flat-top unit hexagon centered at origin:
    Angle 0: (1, 0)
    Angle 60: (cos60, sin60) = (1/2, sqrt(3)/2)
    Angle 120: (-1/2, sqrt(3)/2)
    Angle 180: (-1, 0)
    Angle 240: (-1/2, -sqrt(3)/2)
    Angle 300: (1/2, -sqrt(3)/2)
    
    The rightmost point is at (1, 0). The edges on the right side are:
    From (1,0) to (1/2, sqrt(3)/2) and from (1,0) to (1/2, -sqrt(3)/2).
    
    For two hexagons to be adjacent horizontally (sharing a vertical-ish edge), they would share an edge. But flat-top hexagons don't have vertical edges. The edges are at angles 30, 90, 150, 210, 270, 330 degrees from the center... 
    
    I think I'm overcomplicating this. Let me just use a known packing.
    
    For pointy-top hexagons (rotated 30 degrees from flat-top):
    Vertices at angles 30, 90, 150, 210, 270, 330.
    
    Actually, let's just use a simple grid and optimize. For N=11, a 3x4 grid or similar might work.
    
    Let me try a hexagonal arrangement with 3 rows:
    Row 0: 4 hexagons
    Row 1: 4 hexagons  
    Row 2: 3 hexagons
    Total: 11
    
    For pointy-top hexagons with side 1:
    - The "width" (horizontal extent) of one hexagon is sqrt(3)
    - The "height" (vertical extent) of one hexagon is 2
    - Horizontal spacing between centers in same row: sqrt(3)
    - Vertical spacing between rows: 3/2
    - Odd rows offset by sqrt(3)/2
    
    Let me place them and compute the bounding hexagon.
    """
    # Use pointy-top hexagons (angle = 30 degrees for the rotation parameter)
    # Actually, the angle parameter is the rotation of the hexagon. Let's use 0 for flat-top.
    
    # For a hexagonal lattice of unit hexagons (flat-top, angle=0):
    # Centers form a triangular lattice with spacing sqrt(3) in one direction
    # and 3/2 in the other, with offsets.
    
    # Let's use axial coordinates for hex grid:
    # For flat-top hexagons:
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    
    # We'll place 11 hexagons in a compact hexagonal shape:
    # Ring 0: 1 hexagon (center)
    # Ring 1: 6 hexagons
    # Ring 2: 4 hexagons (to make 11 total: 1+6+4=11)
    
    # Axial coordinates (q, r) for a hex grid:
    # Center: (0,0)
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Ring 2: (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)
    # That's 12 in ring 2. We only need 4 more.
    
    # Let's pick a subset. For symmetry, let's take:
    # (2,0), (2,-2), (-2,0), (-2,2) - these are at the "corners"
    
    hex_coords = [
        (0, 0),  # center
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),  # ring 1
        (2, 0), (2, -2), (-2, 0), (-2, 2)  # 4 from ring 2
    ]
    
    # Convert axial to Cartesian for flat-top hexagons:
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    
    sqrt3 = np.sqrt(3)
    centers = []
    for q, r in hex_coords:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    # All hexagons have angle 0 (flat-top)
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Now compute the minimum enclosing regular hexagon.
    # The outer hexagon has center (cx, cy), side length S, and rotation angle theta.
    # A point (px, py) is inside a regular hexagon with center (cx,cy), side S, rotation theta
    # if the distance from (px,py) to each of the 6 edges is <= inradius = S*sqrt(3)/2.
    
    # The 6 edge normals for a flat-top hexagon rotated by theta:
    # For a flat-top hexagon, the edge normals point at angles 0, 60, 120, 180, 240, 300 degrees.
    # With rotation theta, they point at theta, theta+60, ..., theta+300.
    
    # For a given center and side length, the hexagon contains a point if:
    # For each edge normal direction phi:
    #   (px - cx) * cos(phi) + (py - cy) * sin(phi) <= S * sqrt(3) / 2
    
    # We need to find the minimum S such that all points are inside.
    # For a fixed center and rotation, the minimum S is:
    #   S = max over all points and all 6 edge normals of:
    #       ((px - cx) * cos(phi) + (py - cy) * sin(phi)) / (sqrt(3) / 2)
    #   where phi = theta + k * 60 degrees for k=0..5
    
    # We also need to optimize over center (cx, cy) and rotation theta.
    # This is a 3D optimization problem. Let's use scipy.optimize.minimize.
    
    from scipy.optimize import minimize
    
    def hexagon_side_length(cx, cy, theta):
        """Compute minimum side length of hexagon with center (cx,cy), rotation theta to contain all points."""
        inradius_factor = np.sqrt(3) / 2.0
        max_val = 0.0
        for px, py in centers:
            for k in range(6):
                phi = theta + k * np.pi / 3.0
                val = (px - cx) * np.cos(phi) + (py - cy) * np.sin(phi)
                if val > max_val:
                    max_val = val
        return max_val / inradius_factor
    
    # Initial guess: center at (0,0), rotation 0
    x0 = [0.0, 0.0, 0.0]
    
    def objective(x):
        cx, cy, theta = x
        return hexagon_side_length(cx, cy, theta)
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-8})
    
    cx, cy, theta = result.x
    side = hexagon_side_length(cx, cy, theta)
    
    # Convert theta to degrees
    theta_deg = np.degrees(theta)
    
    return inner, (cx, cy), side, theta_deg
# EVOLVE-BLOCK-END
