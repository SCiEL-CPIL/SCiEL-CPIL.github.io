"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    n = 16
    # The star K_{1,15} has 15 edges and is edge-critical:
    # - Diameter is exactly 2 (any two leaves are distance 2 via center)
    # - Removing any edge disconnects the corresponding leaf
    # No graph with more edges can be edge-critical for diameter 2,
    # because adding any edge to the star makes it non-critical.
    return [[0, v] for v in range(1, n)]
# EVOLVE-BLOCK-END
