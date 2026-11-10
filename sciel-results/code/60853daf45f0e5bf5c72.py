"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    """Return a non-2-colorable 5-uniform hypergraph with minimal edges.
    
    Uses the complete 5-uniform hypergraph on 9 vertices (126 edges) as a
    guaranteed non-2-colorable construction. Any 2-coloring of 9 vertices
    has at least ceil(9/2)=5 vertices of one color, forming a monochromatic
    edge.
    """
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n = 9
    edges = [list(e) for e in combinations(range(n), 5)]
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
