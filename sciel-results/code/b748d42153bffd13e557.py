import numpy as np
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Triangular lattice points
    # Basis vectors
    u = np.array([1.0, 0.0])
    v = np.array([0.5, math.sqrt(3)/2])
    
    # Generate points in shell 0, 1, 2
    points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            p = i * u + j * v
            # Check distance squared
            dist_sq = i**2 + i*j + j**2
            if dist_sq <= 4:
                points.append(p)
                
    # We have 19 points. We need 12.
    # To minimize the bounding box, we should pick points that are "closest" to the center.
    # However, all points with dist_sq <= 1 are 7 points.
    # We need 5 more.
    # The points with dist_sq = 3 are at distance sqrt(3) ~ 1.732.
    # The points with dist_sq = 4 are at distance 2.
    # If we pick 5 points with dist_sq = 3, they are at distance 1.732.
    # The container must have inradius >= 1.732.
    # Inradius of hex with side S is S*sqrt(3)/2.
    # S*sqrt(3)/2 >= 1.732 => S >= 2.
    # If S=2, circumradius is 2.
    # Points at dist_sq=3 are inside.
    # So we can pick any 5 from the 6 points with dist_sq=3.
    # Let's pick the first 5.
    
    # Sort points by distance
    points_sorted = sorted(points, key=lambda p: np.dot(p, p))
    
    # Take the first 12
    selected = points_sorted[:12]
    
    # The minimal bounding hexagon side L
    # Max distance from origin is sqrt(3) for the selected points?
    # Wait, if we pick 5 from dist_sq=3, max dist is sqrt(3).
    # But we must also consider the shape.
    # The points at dist_sq=3 are at angles 30, 90, 150, 210, 270, 330.
    # If we pick 5, we miss one.
    # The max projection will be determined by the points we picked.
    # If we pick 5 out of 6, the max projection is still sqrt(3).
    # So inradius must be >= sqrt(3).
    # S*sqrt(3)/2 >= sqrt(3) => S >= 2.
    # So L = S + 1 = 3.
    
    # Let's verify if we can pick 12 points with max dist < sqrt(3)?
    # No, we only have 7 points with dist < sqrt(3).
    # So we must pick at least 5 points with dist >= sqrt(3).
    # So max dist >= sqrt(3).
    # So L >= 3.
    
    # So L=3 is optimal for this configuration.
    
    inner = []
    for p in selected:
        inner.append([p[0], p[1], 0.0])
        
    return inner, (0.0, 0.0), 3.0, 0.0
# EVOLVE-BLOCK-END
