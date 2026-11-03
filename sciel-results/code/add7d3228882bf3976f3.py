import numpy as np
import time

def construct():
    # Deadline guard: ensure we finish well within 150s
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin
    
    # Strategy: Concentric ring packing with 11 hexagons
    # Ring 0: 1 hexagon at center
    # Ring 1: 6 hexagons at distance sqrt(3) (touching neighbors)
    # Ring 2: 4 hexagons at distance 3 (filling gaps in ring 1)
    # All inner hexagons rotated 30 degrees (flat-top orientation)
    # Outer hexagon also rotated 30 degrees
    
    # Use a small epsilon to ensure strict disjointness (interiors don't overlap)
    # The minimum distance between centers of touching unit hexagons is sqrt(3)
    # We add eps to ensure a tiny gap
    eps = 1e-5
    
    centers = []
    
    # Ring 0: center
    centers.append((0.0, 0.0))
    
    # Ring 1: 6 hexagons at distance sqrt(3) + eps, angles 0, 60, 120, 180, 240, 300
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3) + eps
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
        
    # Ring 2: 4 hexagons at distance 3 + 2*eps, angles 30, 90, 150, 270
    # These fill the "gaps" between ring 1 hexagons
    # Distance from ring 1 hex at angle 0 to ring 2 hex at angle 30:
    # Ring 1 at (sqrt(3), 0), Ring 2 at (3*cos30, 3*sin30) = (3*sqrt(3)/2, 3/2)
    # dx = 3*sqrt(3)/2 - sqrt(3) = sqrt(3)/2
    # dy = 3/2 - 0 = 3/2
    # dist = sqrt(3/4 + 9/4) = sqrt(12/4) = sqrt(3)
    # So they are exactly touching at distance sqrt(3). Good.
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + 2 * eps
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
        
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
        
    # Calculate required outer side length
    # Inner hexagons have angle 30 degrees (flat-top)
    # Vertices of a unit hexagon with angle 30 deg (relative to x-axis):
    # The "radius" (center to vertex) is 1.0
    # The "apothem" (center to side midpoint) is sqrt(3)/2
    
    # For angle 30 deg, vertices are at angles 30, 90, 150, 210, 270, 330
    # Max x offset from center: cos(30) = sqrt(3)/2 (at 30 and 330 deg)
    # Max y offset from center: sin(90) = 1.0 (at 90 deg)
    
    # Find the maximum extent of the cluster
    max_x = 0.0
    max_y = 0.0
    
    for x, y in centers:
        # Max x of this hexagon: x + sqrt(3)/2
        # Max y of this hexagon: y + 1.0
        max_x = max(max_x, x + np.sqrt(3)/2)
        max_y = max(max_y, y + 1.0)
        
        # Also check negative extents for symmetry (though center is at origin)
        max_x = max(max_x, -x + np.sqrt(3)/2)
        max_y = max(max_y, -y + 1.0)
    
    # Outer hexagon has angle 30 degrees (flat-top)
    # Width = sqrt(3) * S (distance between parallel vertical sides)
    # Height = 2 * S (distance between parallel horizontal sides)
    
    # Constraints:
    # sqrt(3) * S >= 2 * max_x
    # 2 * S >= 2 * max_y  =>  S >= max_y
    
    s_x = 2 * max_x / np.sqrt(3)
    s_y = max_y
    
    side = max(s_x, s_y)
    
    # Add a tiny margin to ensure containment
    side += 0.001
    
    return inner, (0.0, 0.0), side, 30.0
