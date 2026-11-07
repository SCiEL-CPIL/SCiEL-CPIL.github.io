"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((e[0], e[1]))
        base_edges.add((e[1], e[0]))
    
    # Strategy: Use vertex 1 as a hub.
    # The base graph is a path 0-1-2-...-23.
    # We want to add edges such that the graph remains triangle-free and has diameter at most 4.
    # By connecting vertex 1 to all even vertices (0, 2, 4, ..., 22), we create a structure where:
    # - Even vertices are at distance 1 from 1 (or 0 if already connected).
    # - Odd vertices (except 1) are at distance 2 from 1 (via their even neighbor).
    # - Specifically, for any odd vertex v (v != 1), its neighbors are v-1 and v+1 (both even).
    #   Since v-1 and v+1 are connected to 1, dist(v, 1) = 2.
    # - For any two vertices u, v:
    #   - If both are even: dist(u, v) <= 2 (via 1).
    #   - If both are odd: dist(u, v) <= 4 (u -> even -> 1 -> even -> v).
    #   - If one is even (e) and one is odd (o):
    #     - If o is neighbor of e, dist=1.
    #     - Else, dist(o, 1) = 2, dist(e, 1) = 1, so dist(o, e) <= 3.
    #   - Max distance is 4 (between two odd vertices not adjacent to the same even).
    
    # Triangle-freeness:
    # - The original path has no triangles.
    # - Added edges are (1, e) for even e.
    # - Can (1, e) form a triangle?
    #   - Neighbors of 1 are 0, 2 (from path) and all even e (added).
    #   - Neighbors of e (even) are e-1, e+1 (odd) from path, and 1 (added).
    #   - For a triangle involving (1, e), we need a common neighbor of 1 and e.
    #   - Common neighbors must be in N(1) ∩ N(e).
    #   - N(1) = {0, 2, 4, ..., 22} (all evens).
    #   - N(e) = {e-1, e+1, 1}.
    #   - Intersection: {e-1, e+1, 1} ∩ {evens}.
    #   - e-1 and e+1 are odd, so not in N(1) (unless 1 is odd, which it is, but 1 is in N(1)? No, 1 is not its own neighbor).
    #   - Wait, N(1) contains evens. 1 is odd. So 1 is not in N(1).
    #   - So N(1) ∩ N(e) = {e-1, e+1} ∩ {evens} = empty set (since e-1, e+1 are odd).
    #   - Thus, no triangles are formed.
    
    added_edges = []
    for a in range(0, n, 2):
        if (1, a) not in base_edges:
            added_edges.append([1, a])
            
    return added_edges
# EVOLVE-BLOCK-END
