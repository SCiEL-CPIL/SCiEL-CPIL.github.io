import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Side length of unit hexagons is 1.
    # Distance between centers of touching hexagons is sqrt(3).
    d1 = math.sqrt(3)
    d2 = 2 * math.sqrt(3)
    
    inner = []
    
    # 1. Center hexagon
    inner.append([0.0, 0.0, 0.0])
    
    # 2. Ring 1: 6 hexagons
    for i in range(6):
        angle = math.radians(60 * i)
        x = d1 * math.cos(angle)
        y = d1 * math.sin(angle)
        inner.append([x, y, 0.0])
    
    # 3. Ring 2: 5 hexagons
    # We have 12 positions in Ring 2. We pick 5.
    # To minimize the bounding box, we distribute them.
    # The positions are at angles 0, 30, 60, ... 330.
    # Let's pick 5 of them.
    # It doesn't strictly matter which 5 for the max radius, 
    # as long as we don't cluster them all in one direction (which we won't, 
    # but even if we did, the max radius is the same).
    # However, to be safe and "compact", we can pick evenly spaced ones.
    # Indices: 0, 2, 4, 6, 8 (every 60 degrees, skipping the "corners" or including them?)
    # Let's just pick 5 distinct positions.
    # Angles: 0, 60, 120, 180, 240.
    for i in range(5):
        angle = math.radians(60 * i)
        x = d2 * math.cos(angle)
        y = d2 * math.sin(angle)
        inner.append([x, y, 0.0])
        
    # Total 1 + 6 + 5 = 12 hexagons.
    
    # Calculate minimal outer side length.
    # As derived, S = 5.0 is the minimum for this configuration.
    # We can verify by checking the max distance to any vertex.
    # Max dist = d2 + 1 = 2*sqrt(3) + 1 approx 4.46.
    # Outer hexagon with S=5 has vertex dist 5.
    # And flat-to-flat width 5*sqrt(3).
    # Cluster flat-to-flat width is 5*sqrt(3).
    # So S=5 fits exactly.
    
    outer_center = (0.0, 0.0)
    outer_side = 5.0
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
