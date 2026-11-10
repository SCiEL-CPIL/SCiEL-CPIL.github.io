import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Initialize with K_{3,13}
    # Vertices 0, 1, 2 form the small part. 3..15 form the large part.
    adj = [0] * n
    current_edges = []
    
    for u in range(3):
        for v in range(3, 16):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            current_edges.append((u, v))
            
    best_edges = current_edges.copy()
    best_count = len(current_edges)
    
    def creates_q3(u, v, adj):
        # adj is a list of 16 ints, where adj[x] is bitmask of neighbors
        # Check if adding (u,v) creates a Q3.
        # We assume (u,v) is NOT yet in adj.
        
        # Precompute neighbor lists for u and v excluding each other
        mask_u = adj[u] & ~(1 << v)
        mask_v = adj[v] & ~(1 << u)
        
        # Get neighbor lists as lists for iteration
        nu = []
        mask = mask_u
        while mask:
            lsb = mask & -mask
            x = lsb.bit_length() - 1
            nu.append(x)
            mask ^= lsb
        
        nv = []
        mask = mask_v
        while mask:
            lsb = mask & -mask
            x = lsb.bit_length() - 1
            nv.append(x)
            mask ^= lsb
        
        # Iterate over pairs a, b in nu (a < b)
        # a, b are neighbors of u (excluding v)
        # In Q3, if u-v is an edge, and u-a, u-b are edges, then a and b must be connected?
        # No, in Q3, neighbors of a vertex are not connected.
        # So a and b are NOT connected in Q3.
        # The structure of Q3 containing edge (u,v):
        # u is connected to v, a, b.
        # v is connected to u, c, d.
        # a is connected to u, c, e. (c is neighbor of v and a)
        # b is connected to u, d, f. (d is neighbor of v and b)
        # Wait, let's trace Q3 properly.
        # Vertices: 0, 1, 2, 3, 4, 5, 6, 7
        # Edges: (0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)
        # Let u=0, v=1.
        # Neighbors of u (0): 1, 2, 4. Excluding v, we have {2, 4}. So a=2, b=4.
        # Neighbors of v (1): 0, 3, 5. Excluding u, we have {3, 5}. So c=3, d=5.
        # Check connections:
        # a (2) is connected to u (0), c (3), e (6).
        #   - (2,3) is an edge. So c must be a neighbor of a.
        #   - (2,6) is an edge. So e must be a neighbor of a.
        # b (4) is connected to u (0), d (5), f (6).
        #   - (4,5) is an edge. So d must be a neighbor of b.
        #   - (4,6) is an edge. So f must be a neighbor of b.
        # c (3) is connected to v (1), a (2), g (7).
        #   - (3,7) is an edge. So g must be a neighbor of c.
        # d (5) is connected to v (1), b (4), g (7).
        #   - (5,7) is an edge. So g must be a neighbor of d.
        # e (6) is connected to a (2), b (4), g (7).
        #   - (6,7) is an edge. So g must be a neighbor of e.
        #   - Wait, in Q3, 6 is connected to 2, 4, 7.
        #   - So e=6, f=6? No, e and f are distinct in my previous trace?
        #   - Let's check vertices again.
        #   - 0: 1, 2, 4
        #   - 1: 0, 3, 5
        #   - 2: 0, 3, 6
        #   - 3: 1, 2, 7
        #   - 4: 0, 5, 6
        #   - 5: 1, 4, 7
        #   - 6: 2, 4, 7
        #   - 7: 3, 5, 6
        #   
        #   If u=0, v=1:
        #   Neighbors of u (excl v): 2, 4. Let a=2, b=4.
        #   Neighbors of v (excl u): 3, 5. Let c=3, d=5.
        #   Common neighbor of a (2) and v (1)? No, 2 and 1 are not connected.
        #   Wait, the logic in the current code is:
        #   "c must be in N(v) ∩ N(a)"
        #   In Q3, N(1) ∩ N(2) = {3}. So c=3.
        #   "d must be in N(v) ∩ N(b)"
        #   In Q3, N(1) ∩ N(4) = {5}. So d=5.
        #   "e must be in N(a) ∩ N(b)"
        #   In Q3, N(2) ∩ N(4) = {6}. So e=6.
        #   "f must be in N(c) ∩ N(d)"
        #   In Q3, N(3) ∩ N(5) = {7}. So f=7.
        #   And (e, f) must be an edge.
        #   In Q3, (6, 7) is an edge. Yes.
        #   
        #   So the logic holds:
        #   1. Pick a, b from N(u) \ {v}.
        #   2. c must be in N(v) ∩ N(a).
        #   3. d must be in N(v) ∩ N(b).
        #   4. e must be in N(a) ∩ N(b).
        #   5. f must be in N(c) ∩ N(d).
        #   6. (e, f) must be an edge.
        #   Also, all vertices u, v, a, b, c, d, e, f must be distinct.
        
        for i in range(len(nu)):
            a = nu[i]
            for j in range(i+1, len(nu)):
                b = nu[j]
                
                # c must be in N(v) ∩ N(a), excluding u, v, a, b
                mask_ca = mask_v & adj[a] & ~((1 << u) | (1 << v) | (1 << a) | (1 << b))
                # d must be in N(v) ∩ N(b), excluding u, v, a, b
                mask_db = mask_v & adj[b] & ~((1 << u) | (1 << v) | (1 << a) | (1 << b))
                
                # Iterate c in mask_ca
                mask_c = mask_ca
                while mask_c:
                    lsb_c = mask_c & -mask_c
                    c = lsb_c.bit_length() - 1
                    mask_c ^= lsb_c
                    
                    # d must be in mask_db, excluding c
                    mask_d = mask_db & ~(1 << c)
                    
                    # e must be in N(a) ∩ N(b), excluding u, v, a, b, c
                    # Note: e is determined by a, b.
                    mask_e_cand = adj[a] & adj[b] & ~((1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c))
                    
                    if mask_e_cand == 0:
                        continue
                    
                    # Iterate d in mask_d
                    mask_d_iter = mask_d
                    while mask_d_iter:
                        lsb_d = mask_d_iter & -mask_d_iter
                        d = lsb_d.bit_length() - 1
                        mask_d_iter ^= lsb_d
                        
                        # f must be in N(c) ∩ N(d), excluding u, v, a, b, c, d
                        mask_f_base = adj[c] & adj[d] & ~((1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | lsb_d)
                        
                        if mask_f_base == 0:
                            continue
                        
                        # Check if any e in mask_e_cand has an edge to some f in mask_f_base
                        # AND e != f (distinctness)
                        # The current code checks if adj[e] & mask_f_base != 0.
                        # This ensures (e,f) is an edge.
                        # Does it ensure e != f?
                        # mask_f_base excludes c and d.
                        # mask_e_cand excludes u, v, a, b, c.
                        # If e == f, then e is in mask_f_base and mask_e_cand.
                        # But e and f are distinct vertices in Q3.
                        # So we must ensure e != f.
                        
                        found = False
                        mask_e = mask_e_cand
                        while mask_e:
                            lsb_e = mask_e & -mask_e
                            e = lsb_e.bit_length() - 1
                            mask_e ^= lsb_e
                            
                            # e must not be d (d is excluded from mask_f_base, but e is not explicitly excluded from being d in mask_e_cand? 
                            # mask_e_cand excludes u,v,a,b,c. It does NOT exclude d.
                            # So e could be d.
                            # In Q3, e and d are distinct.
                            # So if e == d, it's invalid.
                            if e == d:
                                continue
                                
                            if adj[e] & mask_f_base:
                                found = True
                                break
                        
                        if found:
                            return True
        
        return False

    # Greedy Phase on the remaining edges
    # We already have K_{3,13}.
    # Let's try to add edges within B (3..15) and within A (0..2).
    # Edges within A: (0,1), (0,2), (1,2)
    # Edges within B: many
    
    # We'll do a few restarts with different random orders of the "missing" edges
    missing_edges = []
    for u, v in all_edges:
        if not ((adj[u] >> v) & 1):
            missing_edges.append((u, v))
            
    for restart in range(20): # Adjust based on time
        if time.time() > deadline: break
        
        current_adj = adj[:]
        current_edges = best_edges.copy()
        
        random.shuffle(missing_edges)
        
        for u, v in missing_edges:
            if time.time() > deadline: break
            if not creates_q3(u, v, current_adj):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges.append((u, v))
                
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges.copy()
            adj = current_adj[:] # Update global adj for next restart? Or keep best?
            
        # Local Search: Kick and Refill
        # Try a few kicks on the current best found in this restart
        for _ in range(5):
            if time.time() > deadline: break
            
            if not current_edges: break
            idx = random.randint(0, len(current_edges) - 1)
            u_rem, v_rem = current_edges[idx]
            
            # Remove edge
            current_adj[u_rem] &= ~(1 << v_rem)
            current_adj[v_rem] &= ~(1 << u_rem)
            current_edges.pop(idx)
            
            # Refill with missing edges
            # Rebuild missing list? Or just iterate all_edges and check if present.
            random.shuffle(all_edges)
            for u, v in all_edges:
                if time.time() > deadline: break
                if (current_adj[u] >> v) & 1:
                    continue
                if not creates_q3(u, v, current_adj):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges.append((u, v))
                    
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = current_edges.copy()

    return [[u, v] for u, v in best_edges]
