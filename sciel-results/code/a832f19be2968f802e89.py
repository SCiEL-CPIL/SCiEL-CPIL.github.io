import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 16
    # all_edges is list of (u, v)
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    best_edges = []
    best_count = 0
    
    # Precompute neighbor lists for quick access? 
    # Bitmasks are better.
    
    def creates_q3(u, v, adj):
        # adj is a list of 16 ints, where adj[x] is bitmask of neighbors
        # Check if adding (u,v) creates a Q3.
        # We assume (u,v) is NOT yet in adj.
        
        # N_u: neighbors of u excluding v
        # N_v: neighbors of v excluding u
        # Note: adj does NOT contain (u,v) yet.
        
        # To speed up, we can iterate over pairs in N_u and N_v.
        # But we need to be careful with the specific structure.
        
        # Structure:
        # u, v are 000, 001
        # a, b are other neighbors of u (010, 100)
        # c, d are other neighbors of v (011, 101)
        # e is neighbor of a, b (110)
        # f is neighbor of c, d (111)
        # Edges: (u,a), (u,b), (v,c), (v,d), (a,c), (b,d), (a,e), (b,e), (c,f), (d,f), (e,f)
        # Plus (u,v) which is being added.
        
        # So we need to find distinct a,b,c,d,e,f in V \ {u,v} such that:
        # 1. a,b in N(u)
        # 2. c,d in N(v)
        # 3. (a,c) in E
        # 4. (b,d) in E
        # 5. e in N(a) ∩ N(b)
        # 6. f in N(c) ∩ N(d)
        # 7. (e,f) in E
        # 8. e,f distinct from a,b,c,d,u,v
        
        # Optimization:
        # Iterate over a in N(u):
        #   Iterate over b in N(u), b > a:
        #     # a,b are candidates for the "other" neighbors of u
        #     # Now we need c in N(a) and d in N(b) such that c in N(v) and d in N(v)
        #     # Actually, c must be in N(v) ∩ N(a)
        #     # And d must be in N(v) ∩ N(b)
        #     # And (a,c) and (b,d) must be edges.
        #     # Wait, (a,c) is an edge. (b,d) is an edge.
        #     # So c must be in N(v) AND N(a).
        #     # d must be in N(v) AND N(b).
        #     # Also c != d.
        #     # Then e must be in N(a) ∩ N(b).
        #     # f must be in N(c) ∩ N(d).
        #     # And (e,f) must be an edge.
        
        nu = [x for x in range(n) if x != u and x != v and (adj[u] >> x) & 1]
        nv = [x for x in range(n) if x != u and x != v and (adj[v] >> x) & 1]
        
        # To reduce symmetry, we can fix a < b?
        # The roles of a and b are symmetric, and c and d are symmetric.
        # But a is paired with c, b with d.
        
        for i in range(len(nu)):
            a = nu[i]
            for j in range(i+1, len(nu)):
                b = nu[j]
                
                # Candidates for c: neighbors of v that are also neighbors of a
                # Candidates for d: neighbors of v that are also neighbors of b
                # c, d must be distinct.
                # Also c, d must not be a or b? 
                # If c == a, then (v,a) is an edge. But a is neighbor of u.
                # In Q3, (v,a) is NOT an edge. (v is 001, a is 010). Distance 2.
                # So c cannot be a. Similarly d cannot be b.
                # Also c cannot be b? (v,b) is NOT an edge.
                # So c, d must be in N(v) \ {a,b}.
                
                # Let's compute common neighbors
                # common_ca = N(v) ∩ N(a)
                # common_db = N(v) ∩ N(b)
                
                # Iterate c in common_ca
                # Iterate d in common_db
                # Check c != d
                
                # To speed up, precompute N(x) as bitmask?
                # adj is already bitmask.
                
                mask_v = adj[v]
                mask_a = adj[a]
                mask_b = adj[b]
                
                # c is in N(v) and N(a)
                # d is in N(v) and N(b)
                
                # We can iterate over bits in (mask_v & mask_a)
                # and (mask_v & mask_b)
                
                # But we also need to check e and f.
                # e in N(a) & N(b)
                # f in N(c) & N(d)
                # (e,f) in E
                
                # This nested loop might still be slow if degrees are high.
                # Max degree 15.
                # nu size ~ 15.
                # Pairs a,b ~ 100.
                # c candidates ~ 10. d candidates ~ 10.
                # 100 * 10 * 10 = 10,000 checks per edge addition?
                # 120 edges * 10,000 = 1.2M operations.
                # In Python, 1.2M ops is ~0.1s.
                # 115s / 0.1s = 1000 restarts?
                # This seems feasible.
                
                # Let's refine the inner loop.
                
                # Common neighbors of v and a
                # We can just loop x in range(n)
                
                for c in range(n):
                    if c == u or c == v or c == a or c == b: continue
                    if not ((mask_v >> c) & 1): continue
                    if not ((mask_a >> c) & 1): continue
                    
                    for d in range(n):
                        if d == u or d == v or d == a or d == b or d == c: continue
                        if not ((mask_v >> d) & 1): continue
                        if not ((mask_b >> d) & 1): continue
                        
                        # Now we have a,b,c,d
                        # Check if there exists e in N(a) ∩ N(b)
                        # and f in N(c) ∩ N(d)
                        # such that (e,f) is an edge.
                        # e,f must be distinct from u,v,a,b,c,d.
                        
                        # Precompute N(a) ∩ N(b)
                        # mask_e_candidates = mask_a & mask_b
                        # mask_f_candidates = mask_c & mask_d  (where mask_c = adj[c], mask_d = adj[d])
                        
                        mask_c = adj[c]
                        mask_d = adj[d]
                        
                        # e must be in N(a) and N(b)
                        # f must be in N(c) and N(d)
                        # (e,f) must be in E.
                        
                        # Iterate e in N(a) ∩ N(b)
                        #   Check if e is valid (not in used set)
                        #   Then check if there is f in N(c) ∩ N(d) such that (e,f) is edge.
                        #   This is equivalent to: (N(e) & N(c) & N(d)) is non-empty?
                        #   No, f must be in N(c) AND N(d).
                        #   So we need f in (N(c) & N(d)) such that (e,f) in E.
                        #   This means N(e) & N(c) & N(d) must contain some f.
                        
                        # So:
                        # For each e in (N(a) & N(b)):
                        #   if e not in {u,v,a,b,c,d}:
                        #      if (N(e) & N(c) & N(d)) != 0:
                        #          return True
                        #      (Wait, we also need to ensure f is not in {u,v,a,b,c,d,e}? 
                        #       If f is in that set, it might be invalid.
                        #       However, if f is in that set, say f=a, then (e,a) is edge.
                        #       But a is already used.
                        #       So we should mask out the used vertices.)
                        
                        used_mask = 0
                        for x in [u, v, a, b, c, d]:
                            used_mask |= (1 << x)
                            
                        # e candidates: N(a) & N(b) & ~used_mask
                        # But e is also a vertex, so we check (1<<e) in mask.
                        
                        mask_e_cand = mask_a & mask_b & (~used_mask)
                        
                        # Iterate e
                        # Instead of iterating e, we can check:
                        # Does there exist e in mask_e_cand such that (N(e) & N(c) & N(d) & ~used_mask) != 0?
                        # Note: f must also not be in used_mask (except e itself? No, f distinct from e).
                        # So f must be in N(c) & N(d) & ~(used_mask | (1<<e))
                        
                        # This is getting complex to vectorize.
                        # Let's just iterate e.
                        
                        # Get list of e candidates
                        # mask_e_cand is an int.
                        # We can extract bits.
                        
                        # To make it faster, we can precompute for each pair (a,b) the set of common neighbors?
                        # No, adj changes.
                        
                        # Let's try iterating e.
                        # Number of e candidates is small.
                        
                        # Optimization:
                        # If (mask_e_cand == 0): continue
                        
                        # Iterate e
                        # e_mask = mask_e_cand
                        # while e_mask:
                        #    lsb = e_mask & -e_mask
                        #    e = lsb.bit_length() - 1
                        #    ...
                        #    e_mask ^= lsb
                        
                        # Let's implement this.
                        
                        e_mask = mask_e_cand
                        while e_mask:
                            lsb = e_mask & -e_mask
                            e = lsb.bit_length() - 1
                            e_mask ^= lsb
                            
                            # Check f
                            # f must be in N(c) & N(d) & ~(used_mask | (1<<e))
                            mask_f_cand = mask_c & mask_d & ~(used_mask | lsb)
                            
                            # Check if N(e) intersects mask_f_cand
                            if adj[e] & mask_f_cand:
                                return True
                                
        return False

    # Greedy with Local Search
    while time.time() < deadline:
        random.shuffle(all_edges)
        adj = [0] * n
        current_edges = []
        
        # Greedy Phase
        for u, v in all_edges:
            if not creates_q3(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.append((u, v))
                
        # Local Search Phase
        # Try to add more edges?
        # The greedy phase already tried all edges in random order.
        # So current_edges is a maximal set?
        # Not necessarily. If edge A was rejected because of B, and B is still there, A might be addable if we remove B?
        # But we don't remove edges in greedy.
        # So it's a maximal graph with respect to the order.
        
        # To improve, we can try "swaps" or "kicks".
        # Kick: Remove a random edge, then try to add as many edges as possible.
        
        # Let's do a few kicks per restart.
        for _ in range(10): # 10 kicks
            if time.time() > deadline: break
            
            # Pick a random edge to remove
            if not current_edges: break
            idx = random.randint(0, len(current_edges) - 1)
            u_rem, v_rem = current_edges[idx]
            
            # Remove it
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            current_edges.pop(idx)
            
            # Now try to add edges greedily again?
            # Or just try to add the removed one and others?
            # Let's try to add random non-edges.
            # We need a list of non-edges.
            # Or just shuffle all_edges and try to add any that are missing.
            
            random.shuffle(all_edges)
            for u, v in all_edges:
                if time.time() > deadline: break
                # Check if (u,v) is already in graph
                if (adj[u] >> v) & 1:
                    continue
                if not creates_q3(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.append((u, v))
                    
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges.copy()
            
    return [[u, v] for u, v in best_edges]
