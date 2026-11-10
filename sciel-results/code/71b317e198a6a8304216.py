# EVOLVE-BLOCK-START
import numpy as np
import time

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses a 'flower' configuration: 1 center, 6 in first ring, 5 in second ring.
    Hexagons are rotated 30 degrees (Pointy Top) to allow efficient packing.
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    # Time guard
    start_time = time.time()
    deadline = 600.0
    
    # Scaling factor to ensure gaps > 1e-6
    # Minimum distance for touching is sqrt(3) ~ 1.732
    # Gap needed is 1e-6. 
    # Relative expansion: 1e-6 / 1.732 ~ 5.7e-7.
    # Use 1.0e-6 to be safe (gap ~ 1.7e-6).
    S = 1.0 + 1.0e-6
    
    centers = []
    
    # 1. Center
    centers.append([0.0, 0.0])
    
    # 2. Ring 1 (6 hexes)
    # Distance sqrt(3) * S
    r1 = np.sqrt(3) * S
    for i in range(6):
        angle = np.deg2rad(60 * i)
        x = r1 * np.cos(angle)
        y = r1 * np.sin(angle)
        centers.append([x, y])
        
    # 3. Ring 2 (5 hexes)
    # Distance 3 * S (derived from rhombus geometry)
    # Angles 30, 90, 150, 210, 270
    r2 = 3.0 * S
    angles_r2 = [30, 90, 150, 210, 270]
    for ang_deg in angles_r2:
        angle = np.deg2rad(ang_deg)
        x = r2 * np.cos(angle)
        y = r2 * np.sin(angle)
        centers.append([x, y])
        
    inner_hex_data = np.array(centers)
    # All angles are 30 degrees (Pointy Top)
    inner_hex_data = np.column_stack((inner_hex_data, np.full(12, 30.0)))
    
    # Outer hexagon
    # Side length calculation
    # The furthest vertices from the origin are at distance 4*S (Ring 2 center dist 3*S + hex radius 1*S).
    # For a pointy-top outer hexagon (angle 30), the vertices are at distance s.
    # To contain the pointy vertices of the inner hexagons which align with the outer vertices,
    # we need s >= 4*S.
    s = 4.0 * S
    
    # Verify s >= sqrt(12)
    if s < np.sqrt(12):
        s = np.sqrt(12) + 1e-9
        
    outer_hex_data = np.array([0.0, 0.0, 30.0])
    
    return inner_hex_data, outer_hex_data, s

# EVOLVE-BLOCK-END
