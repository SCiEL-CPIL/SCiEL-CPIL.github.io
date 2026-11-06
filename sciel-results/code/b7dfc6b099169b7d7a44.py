import random
import time
import itertools

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    best_edges = None
    best_count = 0
    
    # Run multiple restarts
    for seed in range(200):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        
        # Initialize random graph
        # p=0.8 is a good start for dense K3,3-free graphs
        p = 0.8 
        adj = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        # Helper to get neighbors as bitset
        def get_neighbors(v):
            mask = 0
            for u in range(n):
                if adj[v][u]:
                    mask |= (1 << u)
            return mask
        
        # Precompute neighbor bitsets
        neigh = [0]*n
        for v in range(n):
            neigh[v] = get_neighbors(v)
        
        def update_neigh(v):
            mask = 0
            for u in range(n):
                if adj[v][u]:
                    mask |= (1 << u)
            neigh[v] = mask

        def remove_edge(u, v):
            adj[u][v] = 0
            adj[v][u] = 0
            neigh[u] &= ~(1 << v)
            neigh[v] &= ~(1 << u)

        def add_edge(u, v):
            adj[u][v] = 1
            adj[v][u] = 1
            neigh[u] |= (1 << v)
            neigh[v] |= (1 << u)

        # Greedy removal of K3,3
        # We iterate until no K3,3 is found
        # To speed up, we can limit iterations or check time
        while time.time() < deadline:
            # Calculate badness of edges
            badness = [[0]*n for _ in range(n)]
            found_violation = False
            
            # Iterate all triples
            for a1, a2, a3 in itertools.combinations(range(n), 3):
                # Common neighbors of a1, a2, a3
                common = neigh[a1] & neigh[a2] & neigh[a3]
                count = common.bit_count()
                if count >= 3:
                    found_violation = True
                    # For each a in {a1, a2, a3} and c in common:
                    # Edge (a, c) is part of comb(count-1, 2) K3,3s for this triple
                    add_val = (count - 1) * (count - 2) // 2
                    for a in [a1, a2, a3]:
                        # Iterate bits in common
                        c_mask = common
                        while c_mask:
                            # Get lowest set bit
                            lsb = c_mask & -c_mask
                            c = lsb.bit_length() - 1
                            badness[a][c] += add_val
                            badness[c][a] += add_val
                            c_mask ^= lsb
            
            if not found_violation:
                break
                
            # Find edge with max badness
            max_bad = -1
            max_u, max_v = -1, -1
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j] and badness[i][j] > max_bad:
                        max_bad = badness[i][j]
                        max_u, max_v = i, j
                        
            if max_u == -1:
                break
                
            remove_edge(max_u, max_v)
            
        # Greedy addition of edges
        # We try to add every possible edge if it doesn't create a K3,3
        # We do this in multiple passes until no more edges can be added
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u][v]:
                        continue
                    # Check if adding (u, v) creates a K3,3
                    # Condition: Exists a1, a2 in N(v) and b1, b2 in N(u)
                    # such that ai != bj and ai ~ bj
                    # Note: u, v are not in N(v) or N(u) respectively in the standard sense?
                    # N(v) are neighbors of v. u is not a neighbor of v yet.
                    # So N(v) and N(u) are disjoint from {u, v}?
                    # No, N(v) contains neighbors of v. u is not in N(v).
                    # So candidates for A (part containing u) are N(v).
                    # Candidates for B (part containing v) are N(u).
                    # Wait, in K3,3, u is in Part A. So u connects to Part B.
                    # Part B contains v.
                    # So Part B is {v, b1, b2}. b1, b2 must be neighbors of u.
                    # Part A is {u, a1, a2}. a1, a2 must be neighbors of v.
                    # So a1, a2 in N(v). b1, b2 in N(u).
                    # And a_i must be connected to b_j.
                    
                    # Optimize: 
                    # If |N(v)| < 2 or |N(u)| < 2, skip.
                    # Get lists of neighbors
                    # Using bitsets is faster for intersection, but we need to check edges between sets.
                    
                    # Let's extract lists for speed in inner loop?
                    # n=20, lists are short.
                    Nv = [x for x in range(n) if adj[v][x]]
                    Nu = [x for x in range(n) if adj[u][x]]
                    
                    if len(Nv) < 2 or len(Nu) < 2:
                        continue
                        
                    valid = True
                    # Iterate pairs in Nv and Nu
                    # To speed up, we can precompute a matrix of connections between Nv and Nu?
                    # Or just check.
                    for a1 in Nv:
                        for a2 in Nv:
                            if a1 >= a2: continue
                            for b1 in Nu:
                                for b2 in Nu:
                                    if b1 >= b2: continue
                                    # Check distinctness
                                    if a1 == b1 or a1 == b2 or a2 == b1 or a2 == b2:
                                        continue
                                    # Check edges
                                    if adj[a1][b1] and adj[a1][b2] and adj[a2][b1] and adj[a2][b2]:
                                        valid = False
                                        break
                                if not valid: break
                            if not valid: break
                        if not valid: break
                        if not valid: break
                        if not valid: break
                    if valid:
                        add_edge(u, v)
                        changed = True
                        if time.time() > deadline:
                            break
                if time.time() > deadline:
                    break
        
        # Count edges
        count = 0
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    count += 1
                    edges.append([i, j])
        
        if count > best_count:
            best_count = count
            best_edges = edges

    return best_edges if best_edges else []
