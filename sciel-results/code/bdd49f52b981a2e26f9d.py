"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    """
    Constructs 30 distinct integer coordinate pairs in [-30, 30]^2.
    The configuration is designed to maximize the multiplicity of a specific triangle area.
    
    Strategy:
    Place 15 points on the line y=0 at x=0..14.
    Place 15 points on the line y=1 at x=0..14.
    
    This creates:
    - 14 adjacent pairs on y=0. Each forms a triangle of doubled area 1 with any of the 15 points on y=1. (14 * 15 = 210)
    - 14 adjacent pairs on y=1. Each forms a triangle of doubled area 1 with any of the 15 points on y=0. (14 * 15 = 210)
    Total triangles with doubled area 1: 420.
    
    Other areas will have lower multiplicities.
    """
    points = []
    # 15 points on y=0
    for x in range(15):
        points.append((x, 0))
    # 15 points on y=1
    for x in range(15):
        points.append((x, 1))
        
    return points
# EVOLVE-BLOCK-END
