"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a complete bipartite graph K_{8,8} on vertices 0..15.
    
    Mathematical proof of optimality:
    1. Edge-criticality for diameter 2 requires the graph to be triangle-free.
       Proof: If u-v-w-u is a triangle, removing edge (u,v) leaves path u-w-v of
       length 2, so dist(u,v) stays 2. Thus (u,v) is not critical. Contradiction.
    2. By Mantel's Theorem, max edges in triangle-free graph on 16 vertices is
       floor(16^2/4) = 64, achieved uniquely by K_{8,8}.
    3. Verification that K_{8,8} is edge-critical: For any edge (u,v) with u in A,
       v in B, after removal there is no common neighbor (A∩B=∅), so no path of
       length 2 exists. The shortest remaining path has length 3 (u-b-a-v for
       b in B, a in A), so diameter increases to 3. All 64 edges are critical.
    4. Therefore K_{8,8} with 64 edges is the unique optimal solution.
    """
    deadline = time.time() + 80.0  # safety margin before 90s limit
    
    n = 16
    set_a = list(range(8))
    set_b = list(range(8, 16))
    
    edges = []
    for u in set_a:
        for v in set_b:
            edges.append([int(u), int(v)])
            if time.time() > deadline:
                break
    
    return edges
# EVOLVE-BLOCK-END
