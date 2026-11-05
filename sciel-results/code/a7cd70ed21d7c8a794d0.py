"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n = 20
    best_edges = []
    best_score = 0
    
    # Multiple restarts with different seeds
    for seed in range(200):
        if time.time() > deadline - 5.0:
            break
        
        edges = _greedy_construct(n, seed)
        score = len(edges)
        
        if score > best_score:
            best_score = score
            best_edges = edges
    
    return best_edges

def _greedy_construct(n, seed):
    """
    Greedy construction of K2,3-free graph.
    Start empty, add edges in random order if they don't create K2,3.
    """
    random.seed(seed)
    
    # Generate all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(all_edges)
    
    # Track adjacency for quick common neighbor count
    adj = [set() for _ in range(n)]
    edges = []
    
    for u, v in all_edges:
        # Check if adding edge (u,v) would create a K2,3
        # K2,3 means: there exist two vertices a,b with at least 3 common neighbors
        # Adding edge (u,v) creates K2,3 if:
        # 1. u and v have >= 2 common neighbors (then u,v + those 2 neighbors form K2,2, but we need K2,3)
        # Actually, K2,3 is two vertices with 3 common neighbors.
        # Adding edge (u,v) could create K2,3 if:
        # - There exist vertices x,y such that {x,y} have >= 3 common neighbors after adding (u,v)
        # This happens if u and v have >= 2 common neighbors (call them c1, c2, c3...)
        #   Then for pair (c1, c2), they both connect to u and v, so if they have 1 more common neighbor, that's K2,3
        #   Actually simpler: after adding (u,v), check if any pair has >= 3 common neighbors.
        
        # Efficient check: 
        # Adding (u,v) affects common neighbor counts for pairs involving u or v, and for pair (u,v) itself.
        
        # Common neighbors of u and v:
        common_uv = adj[u] & adj[v]
        
        # For pair (u,v): they now have |common_uv| common neighbors. Need < 3, so |common_uv| <= 2.
        if len(common_uv) >= 3:
            continue
        
        # For pairs (u, w) where w in common_uv: 
        #   u and w share neighbors: adj[u] ∩ adj[w] includes v (new) and all in adj[u]∩adj[w]
        #   Before adding (u,v): common neighbors of u and w = adj[u] ∩ adj[w]
        #   After: (adj[u] ∩ adj[w]) ∪ {v} if v in adj[w]... wait, v is not in adj[w] necessarily.
        #   Actually, v is added to adj[u], so new common neighbors of u and w = (adj[u] ∪ {v}) ∩ adj[w]
        #   = (adj[u] ∩ adj[w]) ∪ ({v} ∩ adj[w])
        #   Since v is not yet in adj[w] (we're checking before adding), {v} ∩ adj[w] = empty unless v in adj[w]
        #   But we haven't added v to adj[u] yet. Let me think more carefully.
        
        # Let's just check: after tentatively adding (u,v), does any pair have >= 3 common neighbors?
        # Pairs to check: (u,v), (u,w) for w in adj[v], (v,w) for w in adj[u]
        
        ok = True
        
        # Check pair (u,v): common neighbors = adj[u] ∩ adj[v]
        if len(adj[u] & adj[v]) >= 3:
            ok = False
        
        if ok:
            # Check pairs (u, w) for w in adj[v]
            for w in adj[v]:
                if w == u:
                    continue
                # Common neighbors of u and w after adding (u,v):
                # adj[u] gains v, so new_adj_u = adj[u] ∪ {v}
                # common = (adj[u] ∪ {v}) ∩ adj[w]
                common = adj[u] & adj[w]
                if v in adj[w]:
                    common = common | {v}
                # Also u is now adjacent to v, but that doesn't directly add to common neighbors of u,w
                # Common neighbors of u and w are vertices x such that x~u and x~w
                # After adding u-v: x~u means x in adj[u] or x=v; x~w means x in adj[w]
                # So common = (adj[u] ∩ adj[w]) ∪ ({v} ∩ adj[w])
                # = (adj[u] ∩ adj[w]) ∪ ({v} if v in adj[w] else empty)
                if len(common) >= 3:
                    ok = False
                    break
        
        if ok:
            # Check pairs (v, w) for w in adj[u]
            for w in adj[u]:
                if w == v:
                    continue
                common = adj[v] & adj[w]
                if u in adj[w]:
                    common = common | {u}
                if len(common) >= 3:
                    ok = False
                    break
        
        if ok:
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
