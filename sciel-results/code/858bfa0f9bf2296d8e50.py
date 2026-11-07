"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    n = 16
    # Use K_8,8 which has 64 edges, diameter 2, and is triangle-free.
    # In a triangle-free graph with diameter 2, every edge is critical 
    # because removing it increases the distance between its endpoints to >= 3.
    
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
