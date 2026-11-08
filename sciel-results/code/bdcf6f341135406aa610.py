"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

def _hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    angle = np.radians(angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    angles = np.radians(np.arange(0, 360, 60))
    # Rotate
    x = side * np.cos(angles)
    y = side * np.sin(angles)
    # Apply rotation
    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a
    return np.column_stack((x_rot + cx, y_rot + cy))

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px, py) is inside hexagon centered at (cx, cy)."""
    # Transform to local coordinates
    angle = np.radians(angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    dx = px - cx
    dy = py - cy
    # Inverse rotation
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    # Check if in hexagon: for a regular hexagon with flat top (vertices at 0,60,...),
    # the boundaries are at |x| <= side and |y| <= side*sqrt(3)/2, but with slanted sides
    # Actually for vertices at 0, 60, 120, ...:
    # The hexagon can be described as:
    # |y| <= side * sqrt(3) / 2
    # |x| <= side - |y| / sqrt(3)  ... no that's not right
    # Let me think again. For a hexagon with vertices at (s,0), (s/2, s*sqrt(3)/2), etc.
    # The six sides are:
    # 1. x = s (right)
    # 2. from (s,0) to (s/2, s*sqrt(3)/2): line x + y/sqrt(3) = s
    # 3. from (s/2, s*sqrt(3)/2) to (-s/2, s*sqrt(3)/2): y = s*sqrt(3)/2
    # 4. from (-s/2, s*sqrt(3)/2) to (-s, 0): -x + y/sqrt(3) = s
    # 5. x = -s (left)
    # 6. from (-s,0) to (-s/2, -s*sqrt(3)/2): -x - y/sqrt(3) = s
    # So constraints:
    # |x| <= s
    # |y| <= s*sqrt(3)/2
    # |x| + |y|/sqrt(3) <= s
    s = side
    sqrt3 = np.sqrt(3.0)
    if abs(lx) > s:
        return False
    if abs(ly) > s * sqrt3 / 2:
        return False
    if abs(lx) + abs(ly) / sqrt3 > s:
        return False
    return True

def _hexagon_overlap_distance(hex1, hex2):
    """
    Estimate minimum separation between two hexagons.
    hex1, hex2: (cx, cy, angle_deg)
    Returns a value that is > 0 if separated, <= 0 if overlapping.
    Uses center distance minus max extent as a rough metric.
    """
    cx1, cy1, a1 = hex1
    cx2, cy2, a2 = hex2
    dx = cx2 - cx1
    dy = cy2 - cy1
    dist = np.sqrt(dx**2 + dy**2)
    # For two unit hexagons, minimum distance between centers to not overlap is 2*sqrt(3) ≈ 3.464
    # This is when they touch along a side.
    # Actually, the inradius of a unit hexagon is sqrt(3)/2, so two hexagons touching side-to-side
    # have center distance = 2 * sqrt(3)/2 = sqrt(3) ≈ 1.732? No.
    # The apothem (inradius) is s*sqrt(3)/2. Two hexagons touching along a full side have
    # center distance = 2 * apothem = s*sqrt(3) = sqrt(3) for s=1.
    # But if they touch at a vertex, the distance can be less? No, the minimum center distance
    # for non-overlapping is actually the distance between parallel sides = 2*apothem = sqrt(3).
    # Wait, let me reconsider. The distance between opposite sides is 2 * s * sqrt(3)/2 = s*sqrt(3).
    # So two hexagons can be placed with centers s*sqrt(3) apart and just touching.
    # But they could also be rotated relative to each other, allowing closer packing in some directions.
    # For a conservative estimate, use center distance - s*sqrt(3) as a proxy.
    return dist - np.sqrt(3.0)

def _min_distance_to_outer_boundary(hexagon, outer_center, outer_side, outer_angle):
    """
    Estimate minimum distance from hexagon to outer boundary.
    hexagon: (cx, cy, angle_deg)
    Returns value > 0 if hexagon is well inside, <= 0 if touching or outside.
    """
    cx, cy, a = hexagon
    ox, oy = outer_center
    dx = cx - ox
    dy = cy - oy
    dist_center = np.sqrt(dx**2 + dy**2)
    # The outer hexagon has inradius = outer_side * sqrt(3)/2
    # The inner hexagon has circumradius = outer_side... no, side=1, so circumradius=1
    # For the inner hexagon to be inside, we need dist_center + 1 <= outer_side * sqrt(3)/2? 
    # Not exactly, because of rotation. But as a rough estimate:
    # The maximum extent of the inner hexagon from its center is 1 (circumradius).
    # The minimum distance from outer center to boundary is outer_side * sqrt(3)/2 (inradius).
    # So we need dist_center + 1 <= outer_side * sqrt(3)/2
    # Thus the "slack" is outer_side * sqrt(3)/2 - dist_center - 1
    return outer_side * np.sqrt(3.0) / 2.0 - dist_center - 1.0

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N unit hexagons inside a regular hexagon, minimizing the outer side length.
    Uses optimization to find good positions and rotations.
    """
    # Initial guess: place hexagons in a grid pattern
    # For N hexagons, use a hexagonal arrangement
    sqrt3 = np.sqrt(3.0)
    
    # Generate initial positions in a hexagonal grid
    # Number of rows needed: approximately ceil(N / 3) or so
    # Let's use a simple spiral or grid
    positions = []
    # Place in a compact hexagonal pattern
    # Row 0: 1 hex
    # Row 1: 2 hexes
    # Row 2: 3 hexes
    # etc.
    row = 0
    placed = 0
    while placed < N:
        for col in range(row + 1):
            if placed >= N:
                break
            # Hexagonal grid spacing
            x = col * 2.0 - row  # offset for hex grid
            y = row * sqrt3 / 2.0 * 2  # vertical spacing
            # Actually for hex grid with flat-top hexagons:
            # Horizontal spacing: 1.5 * side = 1.5
            # Vertical spacing: sqrt(3) * side = sqrt(3)
            # Let me use a simpler approach: just place them in a line or grid
            positions.append([x * 1.5, y * 0.866, 0.0])
            placed += 1
        row += 1
    
    # Simpler initial guess: place in a straight line
    positions = [[2.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    
    # State vector: for each hexagon, (x, y, angle_deg)
    # Plus outer parameters: (ox, oy, outer_side, outer_angle)
    # Total variables: 3*N + 4
    # But we can fix outer center at origin and outer angle at 0 to reduce symmetry
    # So variables: 3*N + 1 (outer_side)
    
    n_vars = 3 * N + 1
    
    def objective(params):
        """Minimize outer_side, with penalties for violations."""
        outer_side = params[-1]
        if outer_side < 0.5:
            return 100.0 + outer_side
        
        # Extract hexagon positions
        hexes = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            angle = params[3*i+2]
            hexes.append((x, y, angle))
        
        # Penalty for overlaps between hexagons
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                sep = _hexagon_overlap_distance(hexes[i], hexes[j])
                if sep < 0:
                    penalty += sep**2
        
        # Penalty for hexagons outside outer boundary
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        for i in range(N):
            d = _min_distance_to_outer_boundary(hexes[i], outer_center, outer_side, outer_angle)
            if d < 0:
                penalty += d**2
        
        # Also penalize if outer_side is too small
        return outer_side + 10.0 * penalty
    
    # Initial parameters
    x0 = np.zeros(n_vars)
    for i in range(N):
        x0[3*i] = positions[i][0]
        x0[3*i+1] = positions[i][1]
        x0[3*i+2] = positions[i][2]
    x0[-1] = 5.0  # initial outer side guess
    
    # Optimize
    result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-8})
    
    best_params = result.x
    outer_side = best_params[-1]
    
    # Build inner hexagons
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        angle = best_params[3*i+2]
        inner.append([x, y, angle])
    
    return inner, (0.0, 0.0), outer_side, 0.0
# EVOLVE-BLOCK-END
