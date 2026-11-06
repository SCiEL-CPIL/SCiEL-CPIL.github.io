import time
import random
import sys

def construct():
    deadline = time.time() + 110
    n = 40
    best_edges = []
    best_count = 0

    # Precompute all possible edges
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Fast C6 check using bit operations
    def has_c6(adj, u, v):
        """
        Check if adding edge (u,v) creates a C6.
        A C6 through (u,v) is: u-a-b-c-d-v where a,b,c,d are distinct
        and not u,v.
        We iterate over a in N(u), d in N(v), and check if there exists
        b in N(a) \ {u,v} and c in N(d) \ {u,v} such that b-c is an edge
        and b != c and b,d,c not in {u,v,a}.
        
        More efficiently: for each a in N(u), for each d in N(v):
          We need b in N(a) and c in N(d) with b-c edge, 
          b not in {u,v,d}, c not in {u,v,a}, b != c.
          
        Alternative: precompute common neighbors.
        For fixed a in N(u) and d in N(v), we need:
          b in N(a) \ {u,v} and c in N(d) \ {u,v} such that 
          b-c is an edge, b != c, b != d, c != a.
          
        Even faster: 
          Let S_a = N(a) \ {u,v}
          Let S_d = N(d) \ {u,v}
          We need edge between S_a and S_d, excluding edges (a,c) for c in S_d? No.
          Actually we need b in S_a, c in S_d, b-c in E, b != c, b != d, c != a.
          
          Since b in S_a means b != u,v and b ~ a.
          c in S_d means c != u,v and c ~ d.
          We need b-c in E.
          Exclude b = d (but d not in S_a since d not ~ a necessarily... wait d is in N(v), not necessarily N(a))
          Actually if b = d, then d ~ a and d ~ v, and we'd have u-a-d-v which is only length 3, not C6.
          Wait, the path is u-a-b-c-d-v, so b and c are distinct vertices.
          If b = d, then we have u-a-d-c-d-v which is not a valid simple path.
          So we need b != c, b != d, c != a.
          
          Let's just check: for b in N(a) \ {u,v,d}, for c in N(d) \ {u,v,a,b}, check if b-c is edge.
        """
        # To speed up, precompute for each pair (a,d) where a~u, d~v
        # But that's expensive. Let's use a different approach.
        
        # Approach: 
        # N(u) and N(v) are known. 
        # For each a in N(u):
        #   For each d in N(v):
        #     Check if there's an edge between (N(a) \ {u,v}) and (N(d) \ {u,v}) 
        #     with the constraint that the endpoints are not u,v and the two chosen 
        #     vertices b (from N(a)) and c (from N(d)) satisfy b != c, b != d, c != a.
        
        # Even simpler: 
        # common = N(a) & N(d) won't work because b and c don't need to be common neighbors.
        # We need b ~ a and c ~ d and b ~ c.
        
        # Let's try: for a in N(u), for d in N(v):
        #   candidates_b = N(a) & ~(1<<u) & ~(1<<v) & ~(1<<d)
        #   candidates_c = N(d) & ~(1<<u) & ~(1<<v) & ~(1<<a)
        #   Now check if there's any edge between candidates_b and candidates_c.
        #   For each b in candidates_b, check if N(b) & candidates_c is non-empty.
        
        nu = adj[u]
        nv = adj[v]
        
        a_mask = nu & ~(1 << v)
        d_mask = nv & ~(1 << u)
        
        # Iterate over a in N(u)
        a_iter = a_mask
        while a_iter:
            a_bit = a_iter & -a_iter
            a = a_bit.bit_length() - 1
            a_iter ^= a_bit
            
            d_iter = d_mask
            while d_iter:
                d_bit = d_iter & -d_iter
                d = d_bit.bit_length() - 1
                d_iter ^= d_bit
                
                # b in N(a), b not in {u, v, d}
                # c in N(d), c not in {u, v, a}
                # b != c, b ~ c
                b_mask = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << d)
                c_mask = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                
                # Check if any b in b_mask has an edge to some c in c_mask
                b_iter = b_mask
                while b_iter:
                    b_bit = b_iter & -b_iter
                    b = b_bit.bit_length() - 1
                    b_iter ^= b_bit
                    
                    if (adj[b] & c_mask) != 0:
                        return True
        
        return False

    def greedy_build(order, adj, edge_set, deadline):
        added_edges = []
        for u, v in order:
            if time.time() > deadline:
                break
            if (u, v) in edge_set:
                continue
            if (adj[u] >> v) & 1:
                continue
            
            if not has_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
                added_edges.append([u, v])
        return added_edges

    random.seed(42)
    
    # Strategy: Start with 8 K5s as a dense C6-free base (K5 has no C6 since |V|=5 < 6)
    # Then greedily add more edges.
    best_adj = [0] * n
    best_edge_set = set()
    best_edges = []
    
    for k in range(8):
        start = k * 5
        for a in range(5):
            for b in range(a + 1, 5):
                u = start + a
                v = start + b
                best_adj[u] |= (1 << v)
                best_adj[v] |= (1 << u)
                best_edge_set.add((u, v))
                best_edges.append([u, v])
    
    best_count = len(best_edges)
    
    # Multiple attempts with different orderings
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        current_adj = best_adj[:]
        current_edge_set = best_edge_set.copy()
        
        # Generate ordering
        strategy = attempts % 5
        if strategy == 0:
            # Random order
            current_order = all_pairs[:]
            random.shuffle(current_order)
        elif strategy == 1:
            # Degree-based: prefer edges between high-degree vertices
            degrees = [bin(x).count('1') for x in current_adj]
            current_order = sorted(all_pairs, 
                                   key=lambda p: degrees[p[0]] + degrees[p[1]], 
                                   reverse=True)
        elif strategy == 2:
            # Random restart from empty
            current_adj = [0] * n
            current_edge_set = set()
            current_order = all_pairs[:]
            random.shuffle(current_order)
        elif strategy == 3:
            # Start with K5s, but try edges within same K5 blocks first (already present)
            # Then cross-block edges sorted by some heuristic
            cross_edges = [(u,v) for u,v in all_pairs if (u//5) != (v//5)]
            current_order = cross_edges[:]
            random.shuffle(current_order)
        else:
            # Random with bias toward non-adjacent vertex pairs
            current_order = all_pairs[:]
            random.shuffle(current_order)
        
        # Filter out already-present edges
        current_order = [(u,v) for u,v in current_order if (u,v) not in current_edge_set]
        
        added = greedy_build(current_order, current_adj, current_edge_set, deadline - 3)
        
        if len(added) > best_count:
            best_count = len(added)
            best_adj = current_adj[:]
            best_edge_set = current_edge_set.copy()
            best_edges = [list(e) for e in added]
    
    # Local improvement: try to add edges that were previously rejected
    # by trying a different order (might help if the C6 check was too conservative? No, it's exact)
    # But we can try 1-for-1 swaps: remove an edge, try adding two others.
    # This is complex, so let's just do a final greedy pass with random order on the best graph.
    
    if time.time() < deadline - 2:
        final_adj = best_adj[:]
        final_edge_set = best_edge_set.copy()
        final_order = all_pairs[:]
        random.shuffle(final_order)
        final_order = [(u,v) for u,v in final_order if (u,v) not in final_edge_set]
        
        added_final = greedy_build(final_order, final_adj, final_edge_set, deadline - 1)
        if len(added_final) > best_count:
            best_count = len(added_final)
            best_edges = [list(e) for e in added_final]
    
    return best_edges
