"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 600

# EVOLVE-BLOCK-START
def construct():
    # Create a triangular pulse spanning the entire domain [-1/4, 1/4]
    # x ranges from -1/4 to 1/4
    # We want a triangle that is 0 at the ends and 1 at the center.
    
    # Indices 0 to N-1
    # Center is at N/2
    centers = np.arange(N)
    center = N / 2.0
    
    # Distance from center in index units
    dist = np.abs(centers - center)
    
    # Max distance is N/2
    # Height should be 1 - dist / (N/2)
    # At dist=0, h=1. At dist=N/2, h=0.
    
    heights = 1.0 - (dist / (N / 2.0))
    
    # Ensure non-negative (numerical safety)
    heights = np.maximum(heights, 0.0)
    
    return heights.tolist()
# EVOLVE-BLOCK-END
