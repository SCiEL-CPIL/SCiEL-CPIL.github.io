import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin for safety
    
    n = 20
    # Precompute all pairs
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    best_edges = []
    
    # Optimization: Use bitmasks for adjacency
    # neighbors[u] is an integer bitmask
    
    while time.time() < deadline:
        random.shuffle(all_edges)
        
        # neighbors[u] is a bitmask of connected vertices
        neighbors = [0] * n
        current_edges = []
        
        for u, v in all_edges:
            # Check if adding (u, v) creates a 4-cycle
            # A 4-cycle u-w-x-v exists if:
            # There is w in N(u) \ {v} and x in N(v) \ {u, w} such that w~x
            
            # Get neighbors of u and v as bitmasks
            nu = neighbors[u]
            nv = neighbors[v]
            
            # We need to check if there is a path u-w-x-v
            # Iterate through potential intermediate w
            # w must be a neighbor of u, and w != v
            # x must be a neighbor of v, and x != u, x != w
            # and w must be connected to x
            
            creates_c4 = False
            
            # Mask out v from u's neighbors
            candidates_w = nu & ~(1 << v)
            
            # Iterate through set bits in candidates_w
            cw = candidates_w
            while cw:
                # Extract lowest set bit
                lsb = cw & -cw
                w = lsb.bit_length() - 1
                cw -= lsb
                
                # Now check if there is an x in N(v) such that x is connected to w
                # x must be in N(v)
                # x must not be u
                # x must not be w
                # x must be in N(w)
                
                # Mask out u and w from v's neighbors
                candidates_x = nv & ~(1 << u) & ~(1 << w)
                
                # Check intersection with N(w)
                if candidates_x & neighbors[w]:
                    creates_c4 = True
                    break
            
            if creates_c4:
                continue
            
            # Safe to add
            neighbors[u] |= (1 << v)
            neighbors[v] |= (1 << u)
            current_edges.append((u, v))
            
        if len(current_edges) > len(best_edges):
            best_edges = current_edges[:]
            
    # Assign colours
    # Since graph is C4-free, any colouring works.
    # We use colours 0..19.
    res = []
    for i, (u, v) in enumerate(best_edges):
        c = i % 20
        res.append([u, v, c])
        
    return res
