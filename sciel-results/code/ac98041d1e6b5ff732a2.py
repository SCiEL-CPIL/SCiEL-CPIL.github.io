import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for 150s limit
    n = 20
    vertices = list(range(n))
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(all_pairs)
    
    # Map pair (i, j) to an index in 0..190
    pair_to_idx = {}
    idx_to_pair = [None] * num_pairs
    for idx, (i, j) in enumerate(all_pairs):
        pair_to_idx[(i, j)] = idx
        idx_to_pair[idx] = (i, j)

    # Precompute triples for K3,3 checking
    triples = list(combinations(vertices, 3))
    
    def popcount(x):
        return x.bit_count()

    def mask_to_edges(adj_mask):
        edges = []
        for i in range(n):
            m = adj_mask[i]
            for j in range(i + 1, n):
                if m & (1 << j):
                    edges.append([i, j])
        return edges

    def is_k33_free(adj_mask):
        for u, v, w in triples:
            common = adj_mask[u] & adj_mask[v] & adj_mask[w]
            if popcount(common) > 2:
                return False
        return True

    def edge_count(adj_mask):
        return sum(m.bit_count() for m in adj_mask) // 2

    def can_add(u, v, adj_mask):
        """
        Check if adding edge (u, v) keeps the graph K3,3-free.
        Optimized: Only check triples that contain u or v.
        """
        bit_v = 1 << v
        bit_u = 1 << u
        new_u = adj_mask[u] | bit_v
        new_v = adj_mask[v] | bit_u
        
        # We need to check any triple (a,b,c) such that adding (u,v) increases 
        # common neighbors of {a,b,c} above 2.
        # This only happens if the triple contains u or v.
        
        checked = set()
        
        # Triples with u
        # Precomputed list of triples containing a vertex is faster
        # But for n=20, iterating all triples is 1140, which is small.
        # However, filtering is faster.
        # Let's use the property: if (u,v) is added, it affects common neighbors of any triple containing u or v.
        
        # Actually, a faster check:
        # A K3,3 violation is created if there exist x,y,z such that x,y,z are all neighbors of u,v,w 
        # where {u,v,w} is a triple? No.
        # The condition is: for any triple {a,b,c}, |N(a) ∩ N(b) ∩ N(c)| <= 2.
        # Adding (u,v) changes N(u) and N(v).
        # So we only need to check triples containing u or v.
        
        # To avoid set overhead in tight loop, we can just iterate all triples?
        # 1140 triples * 20 restarts * many steps might be slow.
        # Let's stick to the filtered approach but optimize it.
        
        # We'll precompute lists of triples containing each vertex once outside the loop?
        # Or just iterate all triples and skip if u not in t and v not in t.
        
        for a, b, c in triples:
            if a != u and a != v and b != u and b != v and c != u and c != v:
                continue
            
            ma = adj_mask[a]
            mb = adj_mask[b]
            mc = adj_mask[c]
            
            if a == u: ma = new_u
            elif a == v: ma = new_v
            if b == u: mb = new_u
            elif b == v: mb = new_v
            if c == u: mc = new_u
            elif c == v: mc = new_v
            
            if popcount(ma & mb & mc) > 2:
                return False
        
        return True

    def greedy_add(adj_mask):
        """Add all possible edges greedily."""
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj_mask[i] & (1 << j)):
                        if can_add(i, j, adj_mask):
                            adj_mask[i] |= (1 << j)
                            adj_mask[j] |= (1 << i)
                            changed = True
        return adj_mask

    def local_search(adj_mask):
        """
        Try to swap edges to increase density or maintain density while allowing better structure.
        Actually, for max edges, we just want to add edges.
        But if stuck, we might need to remove an edge to add two?
        Let's try: Remove one edge, try to add others.
        """
        best_adj = list(adj_mask)
        best_cnt = edge_count(best_adj)
        
        # Try removing each edge and then greedily adding
        # This is expensive: 190 removals * greedy_add
        # But greedy_add is fast if few edges added.
        
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                if adj_mask[i] & (1 << j):
                    # Remove (i, j)
                    new_adj = list(adj_mask)
                    new_adj[i] &= ~(1 << j)
                    new_adj[j] &= ~(1 << i)
                    
                    # Try to add back (i, j) or others?
                    # Just run greedy add on the reduced graph
                    new_adj = greedy_add(new_adj)
                    
                    cnt = edge_count(new_adj)
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_adj = new_adj
                        
        return best_adj, best_cnt

    def smart_repair(adj_mask):
        """
        Deterministically remove edges to remove all K3,3 subgraphs.
        Strategy: Iterate through all triples. If a triple has >2 common neighbors,
        remove an edge from one of the common neighbors to one of the triple vertices.
        We prioritize removing edges that are "least useful" or break most violations.
        """
        for _ in range(100): # Safety limit
            if time.time() > deadline:
                break
            
            # Find violation
            viol = None
            for t in triples:
                u, v, w = t
                common = adj_mask[u] & adj_mask[v] & adj_mask[w]
                if popcount(common) > 2:
                    # Find first common neighbor
                    c_list = [x for x in range(n) if common & (1 << x)]
                    if c_list:
                        # Heuristic: remove edge (u, c_list[0])?
                        # Or the edge that is part of the most violations?
                        # For speed, just remove the first one found.
                        viol = (u, v, w, c_list[0])
                        break
            
            if viol is None:
                break
            
            u, v, w, c = viol
            # Remove edge (u, c)
            adj_mask[u] &= ~(1 << c)
            adj_mask[c] &= ~(1 << u)
            
        return adj_mask

    def structured_start(seed):
        """
        Create a dense structured graph.
        """
        rng = random.Random(seed)
        adj = [0] * n
        # Start with a fairly dense random graph
        # Density 0.7 often works well for n=20
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.7:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    best_adj = None
    best_edges = -1
    
    # Run multiple deterministic restarts
    for seed in range(50): # Increase restarts
        if time.time() > deadline:
            break
        
        adj = structured_start(seed)
        
        # Phase 1: Smart Repair to make it K3,3-free
        adj = smart_repair(adj)
        
        # Phase 2: Greedy Add to maximize edges
        adj = greedy_add(adj)
        
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
            
        # Phase 3: Local Search (Swap-based)
        # Try to improve by removing one edge and re-adding optimally
        if time.time() < deadline - 10:
            adj, cnt = local_search(adj)
            if cnt > best_edges:
                best_edges = cnt
                best_adj = list(adj)
        
        # Phase 4: Try a few random swaps from the current best
        # This helps escape local optima
        for _ in range(10):
            if time.time() > deadline:
                break
            new_adj = list(best_adj)
            # Pick a random edge to remove
            edges_in_graph = []
            for i in range(n):
                for j in range(i+1, n):
                    if new_adj[i] & (1 << j):
                        edges_in_graph.append((i, j))
            
            if not edges_in_graph:
                break
                
            u, v = random.Random(seed + _).choice(edges_in_graph)
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            
            # Greedy add
            new_adj = greedy_add(new_adj)
            cnt = edge_count(new_adj)
            if cnt > best_edges:
                best_edges = cnt
                best_adj = new_adj

    if best_adj is None:
        # Fallback: Empty graph or simple bipartite
        best_adj = [0] * n
        best_edges = 0
        # Construct a bipartite graph K10,10
        for i in range(10):
            for j in range(10, 20):
                best_adj[i] |= (1 << j)
                best_adj[j] |= (1 << i)
        best_edges = 100

    edges = mask_to_edges(best_adj)
    return edges
