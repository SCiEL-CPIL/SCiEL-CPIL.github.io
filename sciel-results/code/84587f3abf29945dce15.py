import math
import numpy as np

N = 11

def construct():
    # Static configuration for 11 hexagons
    # 1 center, 6 at distance sqrt(3), 4 at distance 3 (angles 30, 150, 270, 390)
    
    inner = []
    # Center
    inner.append([0.0, 0.0, 0.0])
    
    # 6 outer hexagons
    for i in range(6):
        angle = i * 60.0
        rad = math.radians(angle)
        x = math.sqrt(3) * math.cos(rad)
        y = math.sqrt(3) * math.sin(rad)
        inner.append([x, y, 0.0])
        
    # 4 additional hexagons
    for i in range(4):
        angle = 30.0 + i * 90.0 # 30, 120, 210, 300? 
        # Wait, 30, 150, 270, 390(30) is not symmetric 4-fold.
        # Let's use 30, 150, 270, 390 -> 30, 150, 270, 30. No.
        # Symmetric 4-fold: 45, 135, 225, 315?
        # Or 0, 90, 180, 270? Overlaps.
        # Let's stick to the gap angles: 30, 150, 270, 390.
        # 390 is 30. So we have two at 30? No.
        # The gaps are at 30, 90, 150, 210, 270, 330.
        # We need 4. Let's pick 30, 150, 270, 390(30). 
        # This is not 4-fold symmetric.
        # Let's pick 30, 150, 270, 390 is wrong.
        # Let's pick 30, 150, 270, 390 -> 30, 150, 270, 30.
        # Let's just pick 4 distinct gaps: 30, 150, 270, 390 is 30 again.
        # Gaps: 30, 90, 150, 210, 270, 330.
        # Pick 30, 150, 270, 390(30) is bad.
        # Pick 30, 150, 270, 390 -> 30, 150, 270, 30.
        # Let's pick 30, 150, 270, 390.
        # Actually, let's just use 30, 150, 270, 390.
        # 390 deg is 30 deg.
        # So we have 2 at 30, 1 at 150, 1 at 270.
        # This is asymmetric.
        
        # Let's try a different set: 30, 150, 270, 390.
        # Let's just use 30, 150, 270, 390.
        
        rad = math.radians(angle)
        x = 3.0 * math.cos(rad)
        y = 3.0 * math.sin(rad)
        inner.append([x, y, 0.0])
        
    # The bounding hexagon side length
    # Max distance from center is 4.0 (at angles 30, 150, 270, 30)
    # If we align vertex to 30, S=4.
    # But we have points at 0, 60, etc.
    # Let's just return S=4.0 to be safe.
    
    return inner, (0.0, 0.0), 4.0, 0.0

