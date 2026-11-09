import numpy as np
import time

N = 12

def construct():
    # Hardcoded solution: 1 center, 6 ring 1, 5 ring 2
    # This fits in a hexagon of side 5.
    # We can potentially improve by optimizing, but hardcoded is safe from timeout.
    
    inner = []
    
    # Center
    inner.append([0.0, 0.0, 0.0])
    
    # Ring 1: 6 hexagons at distance 2
    for k in range(6):
        ang = k * 60.0
        rad = np.deg2rad(ang)
        x = 2.0 * np.cos(rad)
        y = 2.0 * np.sin(rad)
        inner.append([x, y, 0.0])
        
    # Ring 2: 5 hexagons at distance 4
    # Pick 5 from the 12 available lattice positions
    # Angles: 0, 60, 120, 180, 240
    for k in range(5):
        ang = k * 72.0 # Not lattice aligned, but let's check validity
        # Actually, let's use lattice aligned angles to ensure no overlap with ring 1
        # Lattice angles for ring 2 are 0, 30, 60, 90...
        # Let's use 0, 60, 120, 180, 240
        pass
    
    # Re-do Ring 2 with valid lattice angles
    ring2_angles = [0.0, 60.0, 120.0, 180.0, 240.0]
    for ang in ring2_angles:
        rad = np.deg2rad(ang)
        x = 4.0 * np.cos(rad)
        y = 4.0 * np.sin(rad)
        inner.append([x, y, 0.0])
        
    # Outer hexagon
    center = [0.0, 0.0]
    side = 5.0
    angle = 0.0
    
    return inner, center, side, angle
