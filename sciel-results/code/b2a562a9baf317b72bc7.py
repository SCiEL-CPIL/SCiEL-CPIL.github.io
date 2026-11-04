import numpy as np
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    # Generate 12 hex positions in a hexagonal lattice
    # Using flat-top hexagons, adjacent centers are sqrt(3) apart
    positions = []
    
    # Center
    positions.append((0.0, 0.0))
    
    # Ring 1: 6 hexagons at distance sqrt(3)
    for i in range(6):
        angle = np.radians(60 * i)
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        positions.append((x, y))
    
    # Ring 2: need 5 more from the 12 positions at distance 2*sqrt(3)
    # The ring 2 positions are at angles 30 + 60*i degrees
    ring2 = []
    for i in range(12):
        angle = np.radians(30 + 60 * i)
        x = 2 * np.sqrt(3) * np.cos(angle)
        y = 2 * np.sqrt(3) * np.sin(angle)
        ring2.append((x, y))
    
    # Take 5 from ring 2 - choose ones that are adjacent to ring 1 for compactness
    # All ring 2 positions are adjacent to ring 1, so take first 5
    positions.extend(ring2[:5])
    
    # We now have 12 positions
    assert len(positions) == 12
    
    # Compute minimal enclosing regular hexagon
    def min_hexagon_side(points, center, rotation):
        max_proj = 0.0
        for k in range(6):
            angle = rotation + np.radians(60 * k)
            nx = np.cos(angle)
            ny = np.sin(angle)
            for (px, py) in points:
                proj = (px - center[0]) * nx + (py - center[1]) * ny
                max_proj = max(max_proj, abs(proj))
        return 2.0 * max_proj / np.sqrt(3)
    
    # Try various rotations and center offsets
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_rotation = 0.0
    
    # Try center at origin and small offsets
    center_offsets = [(0.0, 0.0)]
    # Try small offsets in case asymmetric cluster benefits from shift
    for dx in [-0.5, -0.25, 0.0, 0.25, 0.5]:
        for dy in [-0.5, -0.25, 0.0, 0.25, 0.5]:
            center_offsets.append((dx, dy))
    
    # Try various rotations
    rotation_candidates = [np.radians(i * 3) for i in range(60)]
    
    for cx, cy in center_offsets:
        for rot in rotation_candidates:
            side = min_hexagon_side(positions, (cx, cy), rot)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_rotation = rot
    
    # Inner hexagons are all axis-aligned
    inner = [[pos[0], pos[1], 0.0] for pos in positions]
    
    return inner, best_center, best_side, np.degrees(best_rotation)
