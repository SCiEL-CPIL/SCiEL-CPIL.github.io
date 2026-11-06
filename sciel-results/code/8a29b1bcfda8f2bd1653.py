"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _creates_q3(adj, u, v):
    """
    Check if adding edge (u, v) to the graph represented by adj creates a Q3 subgraph.
    The new Q3 must contain the edge (u, v).
    """
    # Get neighbors of u and v in the CURRENT graph (before adding u-v)
    # Note: adj[u] and adj[v] do NOT contain each other yet.
    nu = list(adj[u])
    nv = list(adj[v])
    
    # We need to pick 2 neighbors a, b from nu and 2 neighbors c, d from nv.
    # a, b must be non-adjacent.
    # c, d must be non-adjacent.
    
    # Pre-filter: find all non-adjacent pairs in nu and nv
    pairs_u = []
    for i in range(len(nu)):
        for j in range(i+1, len(nu)):
            a, b = nu[i], nu[j]
            if b not in adj[a]: # a and b are not adjacent
                pairs_u.append((a, b))
                
    pairs_v = []
    for i in range(len(nv)):
        for j in range(i+1, len(nv)):
            c, d = nv[i], nv[j]
            if d not in adj[c]: # c and d are not adjacent
                pairs_v.append((c, d))
                
    if not pairs_u or not pairs_v:
        return False
        
    # Iterate through combinations
    for a, b in pairs_u:
        # Common neighbors of a and b (candidates for e)
        common_ab = adj[a] & adj[b]
        
        for c, d in pairs_v:
            # Common neighbors of c and d (candidates for f)
            common_cd = adj[c] & adj[d]
            
            # Check cross edges condition:
            # (a~c and b~d) OR (a~d and b~c)
            cross1 = (c in adj[a] and d in adj[b])
            cross2 = (d in adj[a] and c in adj[b])
            
            if not (cross1 or cross2):
                continue
            
            # We need e in common_ab and f in common_cd such that e~f.
            # Also, e and f must be distinct from u, v, a, b, c, d.
            # Since e is neighbor of a,b, and a,b are neighbors of u, 
            # e is distance 2 from u.
            # f is distance 2 from v.
            
            # Optimization: iterate over smaller set
            if len(common_ab) < len(common_cd):
                for e in common_ab:
                    if e in (u, v, a, b, c, d):
                        continue
                    # Check if e is connected to any f in common_cd
                    # f must be in common_cd AND neighbor of e
                    # And f != e
                    for f in common_cd:
                        if f == e:
                            continue
                        if f in (u, v, a, b, c, d):
                            continue
                        if e in adj[f]:
                            return True
            else:
                for f in common_cd:
                    if f in (u, v, a, b, c, d):
                        continue
                    for e in common_ab:
                        if e == f:
                            continue
                        if e in (u, v, a, b, c, d):
                            continue
                        if e in adj[f]:
                            return True
                            
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    best_edges = 0
    best_adj = [set() for _ in range(n)]
    best_edge_list = []
    
    # Try multiple random orders
    for seed in range(100):
        if time.time() > deadline - 10:
            break
            
        random.seed(seed)
        adj = [set() for _ in range(n)]
        current_edge_list = []
        
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if time.time() > deadline:
                break
            
            # Check if adding (u,v) creates Q3
            if not _creates_q3(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                current_edge_list.append((u, v))
        
        if len(current_edge_list) > best_edges:
            best_edges = len(current_edge_list)
            best_adj = [set(s) for s in adj]
            best_edge_list = current_edge_list[:]
            
    return [[u, v] for u, v in best_edge_list]
# EVOLVE-BLOCK-END
