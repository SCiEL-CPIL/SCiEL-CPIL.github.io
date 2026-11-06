import time
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 115
    n = 16
    
    adj = [0] * n
    
    # Construct Clebsch-like graph
    # Keep d=1 edges for bit 3
    # Keep d=3 edges for bits 0,1,2 (i.e., same bit 3)
    for u in range(n):
        for v in range(u+1, n):
            diff = u ^ v
            # d=1, bit 3
            if diff == (1 << 3):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            # d=3, bits 0,1,2 (diff has 3 bits set, none is bit 3)
            # This means diff is 111 (7) or 110 (6) or 101 (5) or 011 (3)
            # But d=3 means popcount(diff) == 3.
            # And bit 3 must be 0 in diff.
            if diff.bit_count() == 3 and (diff & (1 << 3)) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
    # This should give 40 edges.
    # Check Q3
    def find_q3_edges(adj):
        # ... same as before ...
        pass
        
    # If Q3 found, remove edges
    # ...
    
    return edges
