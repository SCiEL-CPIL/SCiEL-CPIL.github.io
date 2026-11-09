"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START_TIME = None
_DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _add_edge_creates_q3(adj, u, v):
    """Fast check: does adding edge (u,v) create a Q3 subgraph?
    In Q3 with edge (0,1), the structure is:
      u->0, v->1
      a,b -> {2,4} (neighbors of 0, not 1)
      c,d -> {3,5} (neighbors of 1, not 0)
      e -> 6 (adjacent to 2,4,7)
      f -> 7 (adjacent to 3,5,6)
    Required edges: (u,a),(u,b),(v,c),(v,d),(a,c),(b,d),(a,e),(b,e),(c,f),(d,f),(e,f)
    """
    # Get neighbors of u and v (excluding each other, since we're checking after adding)
    nu = []
    nv = []
    for x in range(16):
        if x == u or x == v:
            continue
        if adj[u, x]:
            nu.append(x)
        if adj[v, x]:
            nv.append(x)
    
    if len(nu) < 2 or len(nv) < 2:
        return False
    
    # Try all ordered pairs (a,b) from nu and (c,d) from nv
    # a maps to Q3[2], b maps to Q3[4]
    # c maps to Q3[3], d maps to Q3[5]
    for a in nu:
        for b in nu:
            if a == b:
                continue
            # a must be adjacent to some c in nv (Q3[2]-Q3[3])
            # b must be adjacent to some d in nv (Q3[4]-Q3[5])
            valid_c = [c for c in nv if c != a and c != b and adj[a, c]]
            valid_d = [d for d in nv if d != a and d != b and adj[b, d]]
            if not valid_c or not valid_d:
                continue
            for c in valid_c:
                for d in valid_d:
                    if c == d:
                        continue
                    # Now find e adjacent to a and b, and f adjacent to c and d, with e-f edge
                    # e must be adjacent to a and b (and not u, v, c, d necessarily)
                    # f must be adjacent to c and d (and not u, v, a, b necessarily)
                    # e and f must be adjacent to each other
                    # All 8 vertices must be distinct: u, v, a, b, c, d, e, f
                    for e in range(16):
                        if e in (u, v, a, b, c, d):
                            continue
                        if not (adj[a, e] and adj[b, e]):
                            continue
                        # e is a candidate for Q3[6]
                        for f in range(16):
                            if f in (u, v, a, b, c, d, e):
                                continue
                            if not (adj[c, f] and adj[d, f]):
                                continue
                            if not adj[e, f]:
                                continue
                            # Found a Q3!
                            return True
    return False

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    import numpy as np
    
    # Start with K3,13: vertices 0,1,2 connect to 3..15
    adj = np.zeros((16, 16), dtype=bool)
    for a in range(3):
        for b in range(3, 16):
            adj[a, b] = True
            adj[b, a] = True
    
    best_adj = adj.copy()
    best_count = 39  # 3*13 = 39 edges
    
    # Candidate edges: all pairs not already present
    all_pairs = [(i, j) for i in range(16) for j in range(i+1, 16)]
    remaining = [p for p in all_pairs if not adj[p[0], p[1]]]
    
    random.seed(42)
    random.shuffle(remaining)
    
    for (u, v) in remaining:
        if _elapsed() > _DEADLINE:
            break
        
        # Check if adding (u,v) creates a Q3
        if _add_edge_creates_q3(adj, u, v):
            # Skip this edge
            continue
        else:
            # Add the edge
            adj[u, v] = True
            adj[v, u] = True
            best_count += 1
            best_adj = adj.copy()
    
    # Return the best graph found
    edges = []
    for i in range(16):
        for j in range(i+1, 16):
            if best_adj[i, j]:
                edges.append([i, j])
    
    return edges
# EVOLVE-BLOCK-END
