import time
import random
import sys

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Strategy:
    # 1. Deterministic Greedy Construction:
    #    Iterate through all possible edges (u, v) in a fixed order.
    #    Add edge (u, v) if it does not create a C6.
    #    C6 creation condition: Adding (u, v) creates a C6 iff there exist
    #    vertices x, y such that u-x-y-v is a path of length 3 in G, and
    #    {u, x, y, v} are distinct.
    #    Equivalently: u and v share a common neighbor w, and w has a neighbor z
    #    such that z is connected to u or v? No.
    #    Let's use the standard check:
    #    A graph is C6-free iff for any two non-adjacent vertices u, v,
    #    the number of common neighbors is at most 1? No, that's C4-free.
    #    
    #    Correct C6 check for adding edge (u, v):
    #    It creates a C6 if there is a path of length 4 between u and v in the current graph.
    #    Path: u - a - b - c - v.
    #    All 5 vertices must be distinct.
    #    
    #    Efficient check for "is there a path of length 4 between u and v?":
    #    Let N(x) be the set of neighbors of x.
    #    We need to check if there exist a in N(u), b in N(a), c in N(b) such that c in N(v)
    #    and {u, a, b, c, v} are distinct.
    #    
    #    This is still O(n^3) or worse per check if done naively.
    #    With n=40, we can precompute something?
    #    Or just use bitsets for fast intersection.
    #    
    #    Let's use Python integers as bitsets for neighbor sets.
    #    N[u] is an integer where bit i is 1 if u is connected to i.
    
    N = [0] * n
    edges = []
    edge_set = set()
    
    # Helper to check if adding (u, v) is safe
    def is_safe_add(u, v):
        # Check if there is a path u-a-b-c-v with distinct vertices
        # a must be in N(u)
        # c must be in N(v)
        # b must be in N(a) and N(c)
        # And u, a, b, c, v distinct.
        
        # Iterate over common neighbors of a and c? No.
        # Iterate over a in N(u):
        #   Iterate over c in N(v):
        #     If a == c: skip (path length 2, creates C4, allowed)
        #     If a == v or c == u: skip (u, v not adjacent yet, so a!=v, c!=u unless loops? No loops)
        #     Check if N(a) & N(c) has any bit b such that b != u, b != v, b != a, b != c.
        #     Actually, b is in N(a) and N(c).
        #     We need b != u, b != v, b != a, b != c.
        #     Since b is in N(a), b != a.
        #     Since b is in N(c), b != c.
        #     So we just need to ensure b != u and b != v.
        
        nu = N[u]
        nv = N[v]
        
        # Candidates for 'a' are neighbors of u
        # Candidates for 'c' are neighbors of v
        # We want to know if there exists a, c such that (N[a] & N[c]) has a bit set 
        # that is not u and not v.
        
        # Optimization:
        # Precompute for each pair (a, c) the count of common neighbors excluding u, v?
        # No, u, v change.
        
        # Let's just iterate. n=40 is small.
        # Max 40 * 40 = 1600 iterations per check.
        # Each iteration is a bit operation.
        # 1600 * (1000 checks) = 1.6M ops. Very fast.
        
        for a in range(n):
            if not (nu >> a) & 1:
                continue
            # a is neighbor of u
            if a == v: # Should not happen if u,v not adjacent, but a can be v? No, N(u) doesn't contain u. If v is neighbor of u, we wouldn't be adding it.
                continue
            
            for c in range(n):
                if not (nv >> c) & 1:
                    continue
                # c is neighbor of v
                if c == u:
                    continue
                if a == c:
                    # Path u-a-c-v is length 3. Adding u-v creates C4. Allowed.
                    continue
                
                # Check if there is a common neighbor b of a and c
                # that is not u and not v.
                common = N[a] & N[c]
                # Mask out u and v
                mask = ~(1 << u) & ~(1 << v)
                if common & mask:
                    return False
        
        return True

    # Phase 1: Greedy Construction
    # Order edges by some heuristic? 
    # Random order might be better for diversity? 
    # Or sorted by some centrality?
    # Let's try random order for multiple restarts.
    
    best_edges = []
    best_count = 0
    
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        random.seed() # New random state for each restart? Or just continue.
        
        N = [0] * n
        edges = []
        edge_set = set()
        
        # Generate all possible edges
        all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if time.time() > deadline - 5:
                break
            if is_safe_add(u, v):
                # Add edge
                N[u] |= (1 << v)
                N[v] |= (1 << u)
                edges.append([u, v])
                edge_set.add((u, v))
                edge_set.add((v, u))
        
        count = len(edges)
        if count > best_count:
            best_count = count
            best_edges = list(edges)
            best_N = N[:]
            best_edge_set = set(edge_set)
            
        if best_count >= 105:
            return best_edges

    # Phase 2: Local Search from Best
    # Try to improve by swapping edges.
    # A swap: remove (x, y), add (u, v).
    # Must maintain C6-freeness.
    # This is hard to check incrementally.
    # Instead, let's do a "hill climbing" on the edge set.
    # If we can't add any more edges in Phase 1, we are at a local maximum for that order.
    # Different orders yield different local maxima.
    # The 20 restarts in Phase 1 explore different local maxima.
    
    # Can we do better?
    # Maybe a smarter greedy order?
    # E.g., prefer edges that connect vertices with low degree?
    # Or high degree?
    
    # Let's try one more strategy: 
    # Start with a known dense structure?
    # E.g., Random Regular Graph?
    # Or just rely on the 20 restarts of random greedy.
    
    # Given the time budget, 20 restarts of O(n^2 * n^2) checks might be slow?
    # n=40. n^2/2 = 800 edges.
    # Each check is O(n^2) = 1600 ops.
    # Total ops per restart: 800 * 1600 = 1.28M.
    # 20 restarts: 25.6M ops.
    # Python is slow. 10M ops/sec? ~2.5 seconds.
    # Should fit in 145s easily.
    
    return best_edges
