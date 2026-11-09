"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Place N unit regular hexagons inside a regular hexagon, minimizing outer side length.
    
    Approach: Model as circle packing. Replace each unit hexagon with its circumcircle (radius 1).
    The outer hexagon of side L has a circumradius R_circ = L (distance from center to vertex).
    We need to pack N circles of radius 1 inside a circle of radius R_circ + 1 (so that the hexagons
    fit inside the outer hexagon, accounting for the fact that the circumcircle of a unit hexagon
    has radius 1, and the outer hexagon's inradius is L*sqrt(3)/2, but we use circumcircle approximation).
    
    Actually, let's be more careful. A unit regular hexagon has:
    - Side length 1
    - Circumradius (center to vertex) = 1
    - Inradius (center to edge midpoint) = sqrt(3)/2 ≈ 0.866
    
    The outer hexagon of side L has:
    - Circumradius = L
    - Inradius = L * sqrt(3)/2
    
    For a unit hexagon to fit inside the outer hexagon, the distance from the outer hexagon's center
    to the unit hexagon's center plus the unit hexagon's maximum extent in any direction must be <= 
    the outer hexagon's minimum extent in that direction.
    
    Using the circle packing approximation: we treat each unit hexagon as a circle of radius 1 (its circumradius).
    The outer hexagon can be approximated by its circumscribed circle of radius L. So we need to pack N 
    circles of radius 1 inside a circle of radius L + 1? No, that's not right either.
    
    Let me think again. If we place a unit hexagon (circumradius 1) at the center of the outer hexagon, 
    the vertices of the unit hexagon are at distance 1 from the center. For these to be inside the outer 
    hexagon of side L, we need 1 <= L * sqrt(3)/2 (the inradius), so L >= 2/sqrt(3) ≈ 1.155.
    
    But if we use the circumcircle approximation, we say the unit hexagon occupies a circle of radius 1, 
    and the outer hexagon contains a circle of radius L (its circumradius). So we need to pack N circles 
    of radius 1 inside a circle of radius L. This gives L >= 1 for N=1, which is conservative (worse than 
    the true minimum of ~1.155), but it's a valid upper bound approach.
    
    Actually, for the packing problem, we want to minimize L such that N circles of radius 1 fit inside 
    a circle of radius L. This is the standard circle packing problem.
    
    For N=11, the optimal packing of 11 circles of radius 1 in a circle:
    - One circle in the center
    - 6 circles around it at distance 2 from center (hexagonal arrangement)
    - 4 more circles... 
    
    The standard result for packing 11 unit circles: the optimal container radius is about 3.304...
    But let's use a heuristic to find a good arrangement.
    
    For simplicity and robustness, I'll use a known good configuration for 11 circles and compute 
    the required outer hexagon side length.
    
    Configuration for 11 circles:
    - 1 at center (0,0)
    - 6 at distance 2 from center, at angles 0, 60, 120, 180, 240, 300 degrees
    - 4 more: these can be placed further out. A common arrangement is to put them at the "gaps" 
      of the hexagonal ring, at a larger radius.
    
    Let me compute the maximum distance from center for this arrangement. The 6 inner ring circles 
    are at radius 2. The 4 outer circles need to be placed such that they don't overlap with the 
    inner ring or each other.
    
    Actually, for a simpler approach, let's just use a grid/hexagonal lattice and find the minimal 
    bounding circle, then convert to hexagon side length.
    
    For N=11, a good arrangement is:
    - Row 0 (y=0): x = -2, 0, 2 (3 circles)
    - Row 1 (y=sqrt(3)): x = -3, -1, 1, 3 (4 circles)  
    - Row 2 (y=2*sqrt(3)): x = -2, 0, 2 (3 circles)
    - Row 3 (y=3*sqrt(3)): x = 0 (1 circle)
    
    Wait, that's 11 circles. Let me check distances.
    
    Actually, let me just use a simple force-directed simulation to pack 11 unit circles and find 
    the minimal bounding radius.
    """
    # Use a known good packing for 11 circles
    # Hexagonal packing: 1 in center, 6 in first ring, 4 in second ring (partial)
    
    centers = []
    # Center circle
    centers.append((0.0, 0.0))
    
    # First ring: 6 circles at distance 2 from center
    for i in range(6):
        angle = i * np.pi / 3
        centers.append((2.0 * np.cos(angle), 2.0 * np.sin(angle)))
    
    # Second ring: 4 circles. Place them at angles 30, 90, 150, 210, 270, 330 (offset by 30 from first ring)
    # But we only need 4. Let's place them at the positions that maximize symmetry.
    # A common arrangement: place 4 circles at the "outer" gaps.
    # Distance from center for second ring: sqrt(3) * 2 ≈ 3.464? No.
    # In hexagonal packing, the second ring is at distance 4 from center? No.
    # 
    # Let me think: in a hexagonal lattice with spacing 2, the second ring circles are at distance 
    # sqrt(2^2 + (2*sqrt(3))^2) = sqrt(4+12) = 4? No, that's not right.
    #
    # Actually, in a hexagonal close packing of unit circles (spacing 2), the neighbors of the center 
    # are at distance 2. The next neighbors (second ring) are at distance 4 in the axial directions, 
    # or at distance sqrt(4 + 12) = 4 in the "diagonal" directions? 
    #
    # Let me just use a simple approach: place the remaining 4 circles at a larger radius, say 4, 
    # at angles 30, 150, 270, 390(=30). Wait, that's not symmetric.
    #
    # For simplicity, let's use a different arrangement. Let me use a rectangular grid-like arrangement.
    
    # Reset and use a different strategy: use scipy's optimization to minimize the bounding radius
    from scipy.optimize import minimize
    
    # Initial guess: hexagonal arrangement
    n = N
    x0 = np.zeros(n * 2)
    # Place first ring
    for i in range(6):
        angle = i * np.pi / 3
        x0[i*2] = 2.0 * np.cos(angle)
        x0[i*2+1] = 2.0 * np.sin(angle)
    # Place remaining 4 at larger radius
    for j in range(4):
        idx = 7 + j
        angle = (j + 0.5) * np.pi / 3  # Offset by 30 degrees
        x0[idx*2] = 4.0 * np.cos(angle)
        x0[idx*2+1] = 4.0 * np.sin(angle)
    
    def objective(params):
        pts = params.reshape(-1, 2)
        # Compute max distance from center
        max_dist = np.max(np.sqrt(np.sum(pts**2, axis=1)))
        # Penalty for overlapping circles
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt(np.sum((pts[i] - pts[j])**2))
                if d < 2.0:
                    penalty += (2.0 - d)**2
        return max_dist + 100.0 * penalty
    
    result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
    
    pts = result.x.reshape(-1, 2)
    
    # Compute the required outer hexagon side length
    # The maximum distance from center to any circle center is:
    max_center_dist = np.max(np.sqrt(np.sum(pts**2, axis=1)))
    
    # The outer hexagon must contain all circles of radius 1.
    # The circumradius of the outer hexagon must be at least max_center_dist + 1.
    # But we also need to account for the hexagon shape. Using the circumcircle approximation:
    # Outer hexagon side L = circumradius = max_center_dist + 1
    
    L = max_center_dist + 1.0
    
    # However, we can potentially do better by rotating the outer hexagon. 
    # For now, use the simple approximation.
    
    # Create the inner hexagon positions
    inner = []
    for i in range(n):
        x, y = pts[i]
        angle = 0.0  # Default rotation
        inner.append([float(x), float(y), float(angle)])
    
    # Center of outer hexagon
    center = (0.0, 0.0)
    
    return inner, center, float(L), 0.0
# EVOLVE-BLOCK-END
