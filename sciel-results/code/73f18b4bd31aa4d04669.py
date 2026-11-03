"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Place N unit hexagons in a straight line sharing edges
    # For flat-top hexagons, edge-sharing neighbors are spaced sqrt(3) apart horizontally
    # Use a small gap factor to ensure strict disjointness
    gap = 1.001
    spacing = math.sqrt(3) * gap
    
    inner = []
    for k in range(N):
        x = spacing * (k - (N - 1) / 2.0)
        y = 0.0
        angle = 0.0  # flat-top orientation
        inner.append([x, y, angle])
    
    center = (0.0, 0.0)
    
    # The minimal enclosing regular hexagon for a linear chain of N unit hexagons
    # The chain has length sqrt(3)*(N-1) + 2 in the x-direction (center to center span + one hexagon width)
    # and height 2 in the y-direction
    # For a regular hexagon to enclose this, we need to find the minimal side length
    # 
    # The bounding box of the chain:
    # x-range: from -sqrt(3)*(N-1)/2 - sqrt(3)/2 to sqrt(3)*(N-1)/2 + sqrt(3)/2 = +/- sqrt(3)*N/2
    # y-range: +/- 1 (height of hexagon is 2, so half-height is 1)
    # 
    # A regular hexagon with side s, flat-top, has:
    # - horizontal extent: s (from -s/2 to s/2 at the flat sides... no)
    # Let me compute vertices of outer hexagon with side s, center at origin, flat-top:
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees, radius s
    # x-extent: s (from -s to s? no, from -s*cos(0)= -s to s*cos(0)=s... wait)
    # Vertex at angle 0: (s, 0), angle 60: (s/2, s*sqrt(3)/2), etc.
    # x-range: [-s, s], y-range: [-s*sqrt(3)/2, s*sqrt(3)/2]
    #
    # We need s >= sqrt(3)*N/2 and s*sqrt(3)/2 >= 1
    # s >= max(sqrt(3)*N/2, 2/sqrt(3))
    # For N=11: sqrt(3)*11/2 = 9.526, 2/sqrt(3) = 1.155
    # So s >= 9.526
    #
    # But we also need to check that the hexagons fit inside, not just the bounding box.
    # The chain is along the x-axis, so the critical constraint is the x-extent.
    # With the gap factor, x-extent = sqrt(3)*gap*N/2
    # So s = sqrt(3)*gap*N/2 + epsilon
    
    side = math.sqrt(3) * gap * N / 2.0 + 0.01
    angle = 0.0
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Return best found (which is this static solution)
        pass
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
