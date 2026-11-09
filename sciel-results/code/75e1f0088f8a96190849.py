"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    # Use scaled Bessel J_{1/2} zero pattern: z_i = c * i^2
    # The Bessel zeros are at i^2 * pi^2, but scaling down may reduce B
    # Try c = 1 (i.e., z_i = i^2) which is ~10x smaller than Bessel zeros
    z = [float(i ** 2) for i in range(1, 13)]
    
    # They must be finite, > 0 and pairwise distinct
    # i^2 for i=1..12 are all positive, finite, and distinct
    
    return z
# EVOLVE-BLOCK-END
