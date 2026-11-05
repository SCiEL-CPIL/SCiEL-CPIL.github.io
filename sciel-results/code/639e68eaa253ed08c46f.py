import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n = 16
    vertices = list(range(n))
    
    # Precompute all pairs
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_q3_free(adj):
        # adj is a list of 16 ints (bitmasks)
        # Check all 8-subsets? No, use the edge-addition logic in reverse?
        # Or just check if any Q3 exists.
        # For a full graph check, we can iterate over all edges (u,v) and see if they are part of a Q3.
        # But that's O(E * Check).
        # Since we use this to verify final result, it's okay if it's a bit slower.
        # But we can also just rely on the construction being correct.
        # However, for safety, we verify the best candidate.
        
        # Fast check:
        # For each edge (u, v) in the graph:
        # Check if there is a Q3 containing (u, v).
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    continue
                
                # Candidates for a, b (neighbors of u, not v)
                # a, b must be in N(u) \ N(v)
                # Note: N(u) includes v? No, simple graph.
                # adj[u] has bit v set.
                # We want x such that adj[u][x] and not adj[v][x].
                
                # Bitmask operations
                # N_u = adj[u]
                # N_v = adj[v]
                
                # a, b must be in N_u AND NOT in N_v AND NOT u, v.
                # Mask for "not v" is ~ (1<<v).
                # Mask for "not u" is ~ (1<<u).
                
                mask_not_uv = ~((1 << u) | (1 << v))
                cand_ab_mask = adj[u] & ~adj[v] & mask_not_uv
                cand_cd_mask = adj[v] & ~adj[u] & mask_not_uv
                
                # Get lists
                cand_ab = [x for x in range(n) if (cand_ab_mask >> x) & 1]
                cand_cd = [x for x in range(n) if (cand_cd_mask >> x) & 1]
                
                if len(cand_ab) < 2 or len(cand_cd) < 2:
                    continue
                
                for a, b in itertools.combinations(cand_ab, 2):
                    for c, d in itertools.combinations(cand_cd, 2):
                        # Check distinctness (guaranteed by masks? 
                        # cand_ab is not in N(v), cand_cd is not in N(u).
                        # Can a be in cand_cd? 
                        # a in N(u), a not in N(v).
                        # c in N(v), c not in N(u).
                        # So sets are disjoint.
                        
                        # Check edges a-c, a-d, b-c, b-d
                        if not ((adj[a] >> c) & 1): continue
                        if not ((adj[a] >> d) & 1): continue
                        if not ((adj[b] >> c) & 1): continue
                        if not ((adj[b] >> d) & 1): continue
                        
                        # Find e, f
                        # e in N(a) & N(b) & ~N(c) & ~N(d) & ~{u,v,a,b,c,d}
                        # f in N(c) & N(d) & ~N(a) & ~N(b) & ~{u,v,a,b,c,d}
                        
                        excluded = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d)
                        
                        mask_e = adj[a] & adj[b] & ~adj[c] & ~adj[d] & ~excluded
                        mask_f = adj[c] & adj[d] & ~adj[a] & ~adj[b] & ~excluded
                        
                        # e and f must be connected
                        # Iterate e from mask_e
                        e_list = [x for x in range(n) if (mask_e >> x) & 1]
                        f_list = [x for x in range(n) if (mask_f >> x) & 1]
                        
                        for e in e_list:
                            for f in f_list:
                                if e == f: continue
                                if (adj[e] >> f) & 1:
                                    return False
        return True

    def try_add_edge(adj, u, v):
        # Check if adding (u,v) creates a Q3
        # Same logic as above, but assuming (u,v) is present.
        
        mask_not_uv = ~((1 << u) | (1 << v))
        
        # Note: adj does NOT have (u,v) yet.
        # So N(u) does not contain v. N(v) does not contain u.
        
        cand_ab_mask = adj[u] & ~adj[v] & mask_not_uv
        cand_cd_mask = adj[v] & ~adj[u] & mask_not_uv
        
        cand_ab = [x for x in range(n) if (cand_ab_mask >> x) & 1]
        cand_cd = [x for x in range(n) if (cand_cd_mask >> x) & 1]
        
        if len(cand_ab) < 2 or len(cand_cd) < 2:
            return True # Safe to add
        
        for a, b in itertools.combinations(cand_ab, 2):
            for c, d in itertools.combinations(cand_cd, 2):
                
                # Check edges a-c, a-d, b-c, b-d
                if not ((adj[a] >> c) & 1): continue
                if not ((adj[a] >> d) & 1): continue
                if not ((adj[b] >> c) & 1): continue
                if not ((adj[b] >> d) & 1): continue
                
                excluded = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d)
                
                # e, f candidates
                mask_e = adj[a] & adj[b] & ~adj[c] & ~adj[d] & ~excluded
                mask_f = adj[c] & adj[d] & ~adj[a] & ~adj[b] & ~excluded
                
                e_list = [x for x in range(n) if (mask_e >> x) & 1]
                f_list = [x for x in range(n) if (mask_f >> x) & 1]
                
                for e in e_list:
                    for f in f_list:
                        if e == f: continue
                        if (adj[e] >> f) & 1:
                            return False # Would create Q3
                            
        return True

    best_edges_count = 0
    best_adj = [0] * n
    
    # Run multiple random restarts
    # Strategy: Start with empty graph, add edges in random order.
    # Or start with random graph, remove Q3s, then add.
    
    for restart in range(20): # Adjust based on time
        if time.time() > deadline:
            break
            
        # Initialize empty graph
        adj = [0] * n
        
        # Random order of edges
        edges = all_edges[:]
        random.shuffle(edges)
        
        # Greedy add
        for u, v in edges:
            if time.time() > deadline:
                break
            if try_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Local Search: Try to add missing edges
        # Repeat a few times
        for _ in range(10):
            improved = False
            for u, v in edges:
                if (adj[u] >> v) & 1:
                    continue
                if try_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
            if not improved:
                break
        
        # Count edges
        count = sum(bin(adj[i]).count("1") for i in range(n)) // 2
        
        if count > best_edges_count:
            best_edges_count = count
            best_adj = adj[:]

    # Convert best_adj to list of edges
    final_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                final_edges.append([u, v])
                
    return final_edges

