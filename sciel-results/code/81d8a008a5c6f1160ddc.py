import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135
    n = 20
    vertices = list(range(n))
    triples = list(combinations(vertices, 3))
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute vertex-to-triples mapping for fast constraint checking
    vertex_to_triples = [[] for _ in range(n)]
    for idx, (a, b, c) in enumerate(triples):
        vertex_to_triples[a].append(idx)
        vertex_to_triples[b].append(idx)
        vertex_to_triples[c].append(idx)

    def popcount(x):
        return x.bit_count()

    def is_k33_free(adj_mask):
        for t in triples:
            u, v, w = t
            common = adj_mask[u] & adj_mask[v] & adj_mask[w]
            if popcount(common) > 2:
                return False
        return True

    def edge_count(adj_mask):
        return sum(m.bit_count() for m in adj_mask) // 2

    def mask_to_edges(adj_mask):
        edges = []
        for i in range(n):
            m = adj_mask[i]
            for j in range(i + 1, n):
                if m & (1 << j):
                    edges.append([i, j])
        return edges

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
        # Actually, adding edge (u,v) changes adj[u] and adj[v].
        # So any triple involving u or v is affected.
        
        # To be safe and fast, we iterate over triples containing u or v.
        # Union of triples containing u and v.
        # Since n=20, this is small.
        
        checked = set()
        
        # Triples with u
        for t_idx in vertex_to_triples[u]:
            checked.add(t_idx)
            a, b, c = triples[t_idx]
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
        
        # Triples with v (that don't have u)
        for t_idx in vertex_to_triples[v]:
            if t_idx in checked:
                continue
            a, b, c = triples[t_idx]
            ma = adj_mask[a]
            mb = adj_mask[b]
            mc = adj_mask[c]
            
            if a == v: ma = new_v
            if b == v: mb = new_v
            if c == v: mc = new_v
            
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

    def deterministic_repair(adj_mask):
        """
        Deterministically remove edges to remove all K3,3 subgraphs.
        Strategy: Iterate through all triples. If a triple has >2 common neighbors,
        remove an edge from one of the common neighbors to one of the triple vertices.
        We prioritize removing edges that are "least useful" (e.g., highest degree vertices? 
        or just first found). For determinism and maximizing edges, we should remove
        the edge that appears in the most violations? 
        Simple deterministic approach: 
        1. Find a violating triple (u,v,w) and a common neighbor x.
        2. Remove edge (u,x) or (v,x) or (w,x).
        3. Repeat.
        To maximize edges, we want to remove the "most central" edge? No, we want to remove
        an edge that breaks the most potential violations. 
        However, a simple greedy removal is often sufficient for structured starts.
        Let's try: For each violating triple, remove edge (u, x) where x is the first common neighbor.
        """
        for _ in range(1000): # Safety limit
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
                        viol = (u, v, w, c_list[0])
                        break
            
            if viol is None:
                break
            
            u, v, w, c = viol
            # Remove edge (u, c)
            # This is a heuristic choice. We could try all 3 and pick the one that
            # results in the fewest new violations, but that's expensive.
            # For now, deterministic removal.
            adj_mask[u] &= ~(1 << c)
            adj_mask[c] &= ~(1 << u)
            
        return adj_mask

    def structured_start(seed):
        """
        Create a dense structured graph.
        We'll use a circulant graph or a random regular graph as a base.
        Since we need K3,3 free, high density is hard.
        Let's try a random graph with high density and then repair.
        """
        rng = random.Random(seed)
        adj = [0] * n
        # Start with a fairly dense random graph
        # K3,3 free max edges is roughly O(n^2). For n=20, maybe ~150-180?
        # Let's try density 0.6
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.6:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    best_adj = None
    best_edges = -1
    
    # Deterministic sweep over different structured starts
    # We will try several seeds and also some specific structures
    
    # Structure 1: Complete graph minus a matching? No, that has K3,3.
    # Structure 2: Turan graph? T(20, 2) is bipartite, no K3,3. 
    # T(20,2) has 10*10 = 100 edges.
    # T(20,3) has no K3,3? 
    # K3,3 is a subgraph of K3,3. 
    # Turan graphs T(n, r) are K_{r+1}-free.
    # K3,3 is not a clique. So Turan graphs don't directly apply to K3,3-free.
    # However, bipartite graphs can have K3,3.
    
    # Let's stick to random starts with deterministic repair, but optimize the repair.
    # The current program's weakness is likely getting stuck in local optima with random search.
    # Deterministic repair from a very dense start might find a better global optimum
    # if the repair step is smart.
    
    # Improved Repair:
    # Instead of removing the first found edge, we can count how many times each edge is "guilty"
    # of being part of a violation and remove the most guilty edge.
    
    def smart_repair(adj_mask):
        for _ in range(100):
            if time.time() > deadline:
                break
            
            # Count violations per edge
            # Edge (u,v) is guilty if there exists a triple (a,b,c) containing u (or v)
            # such that v (or u) is a common neighbor and count > 2.
            # This is complex. 
            # Simpler: Find all violating triples. For each, add "guilt" to edges (u,c), (v,c), (w,c).
            # Remove the edge with highest guilt.
            
            guilt = {}
            found_violation = False
            
            for t in triples:
                u, v, w = t
                common = adj_mask[u] & adj_mask[v] & adj_mask[w]
                if popcount(common) > 2:
                    found_violation = True
                    for c in range(n):
                        if common & (1 << c):
                            # Edges (u,c), (v,c), (w,c) are candidates for removal
                            # Only if they exist
                            if adj_mask[u] & (1 << c):
                                key = (min(u,c), max(u,c))
                                guilt[key] = guilt.get(key, 0) + 1
                            if adj_mask[v] & (1 << c):
                                key = (min(v,c), max(v,c))
                                guilt[key] = guilt.get(key, 0) + 1
                            if adj_mask[w] & (1 << c):
                                key = (min(w,c), max(w,c))
                                guilt[key] = guilt.get(key, 0) + 1
            
            if not found_violation:
                break
            
            if not guilt:
                # Should not happen if found_violation is True and edges exist
                break
            
            # Find edge with max guilt
            max_guilt_edge = max(guilt, key=guilt.get)
            u, v = max_guilt_edge
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            
        return adj_mask

    # Run multiple deterministic restarts with smart repair
    for seed in range(20):
        if time.time() > deadline:
            break
        
        adj = structured_start(seed)
        
        # Phase 1: Smart Repair
        adj = smart_repair(adj)
        
        # Phase 2: Greedy Add to maximize edges
        adj = greedy_add(adj)
        
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
            
        # Phase 3: Local polish (small random swaps to escape local optima)
        # Since we are changing strategy to deterministic, let's keep a small stochastic polish
        # but rely heavily on the structured start + smart repair.
        
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
