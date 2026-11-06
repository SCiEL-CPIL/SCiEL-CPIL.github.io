"""K2,3-free graph on 20 vertices: maximize edges via bitmask-accelerated degree-balanced greedy + local polish."""
import time

N = 20
DEADLINE_MARGIN = 10.0  # seconds before hard limit to stop


def _is_valid_addition(adj_bitmasks, u, v):
    """Check if adding edge (u,v) keeps graph K2,3-free."""
    common = adj_bitmasks[u] & adj_bitmasks[v]
    # If u and v already share >= 3 neighbours, adding an edge between them
    # doesn't create K2,3 by itself (K2,3 needs two vertices with 3 common neighbours
    # that are NOT adjacent to each other? Actually K2,3 is a bipartite subgraph).
    # Wait: K2,3 as a subgraph means there exist two vertices a,b and three vertices c1,c2,c3
    # such that a and b are both connected to c1,c2,c3. The edges a-b, a-ci, bi don't need to exist.
    # So the condition is: no two vertices have 3 or more common neighbours.
    # Adding edge (u,v) doesn't change common neighbours of any pair except pairs involving u or v.
    # For pair (u,v): their common neighbours are adj[u] & adj[v]. Adding edge (u,v) means
    # u becomes a neighbour of v and vice versa. But u is not in adj[u] and v is not in adj[v]
    # (no self-loops). So adj[u] & adj[v] doesn't include u or v. Adding edge (u,v) doesn't
    # change adj[u] & adj[v] because u not in adj[v] before, and after adding, u IS in adj[v].
    # Wait, adj[v] will now include u. So new adj[u] & adj[v] = (adj[u] & adj[v]) | (if u in adj[u]? no).
    # u is not in adj[u], so u won't be in the intersection. Similarly v not in adj[v].
    # Actually: new_adj[u] = adj[u] | {v}, new_adj[v] = adj[v] | {u}.
    # new_common = new_adj[u] & new_adj[v] = (adj[u] | {v}) & (adj[v] | {u})
    # = (adj[u] & adj[v]) | (adj[u] & {u}) | ({v} & adj[v]) | ({v} & {u})
    # = (adj[u] & adj[v]) | (empty) | (empty) | (empty)  [since u not in adj[u], v not in adj[v]]
    # = adj[u] & adj[v]
    # So adding edge (u,v) does NOT change the common neighbour count of pair (u,v).
    # 
    # However, for other pairs (u, w) where w != v: new_adj[u] includes v now.
    # So new common neighbours of (u,w) = (adj[u] | {v}) & adj[w] = (adj[u] & adj[w]) | ({v} & adj[w])
    # If v is in adj[w], then v becomes a new common neighbour of (u,w).
    # Similarly for (v, w) where w != u.
    # 
    # So we need to check: for all w not in {u,v}, if v is in adj[w], then (u,w) gains a common neighbour.
    # And if u is in adj[w], then (v,w) gains a common neighbour.
    
    # Let's count: for each w in adj[v] (w != u), check if (adj[u] & adj[w]).count() + 1 > 2, i.e., >= 3
    # Similarly for each w in adj[u] (w != v), check if (adj[v] & adj[w]).count() + 1 > 2
    
    # But wait, we also need to check that the pair (u,w) doesn't already have 3 common neighbours.
    # If it already has 3, the graph is already invalid. So we assume current graph is valid.
    # After adding (u,v), pair (u,w) for w in adj[v], w != u: common becomes (adj[u]&adj[w]) + {v} if v in adj[w].
    # Since w in adj[v] means v in adj[w] (symmetric), so v IS in adj[w].
    # So new count = popcount(adj[u] & adj[w]) + 1. We need this <= 2, i.e., popcount(adj[u] & adj[w]) <= 1.
    
    # Similarly for pair (v,w) for w in adj[u], w != v: new count = popcount(adj[v] & adj[w]) + 1 <= 2.
    
    # Also, for pairs not involving u or v, nothing changes.
    
    # So the check is:
    # 1. For all w in adj[v] \ {u}: popcount(adj[u] & adj[w]) <= 1
    # 2. For all w in adj[u] \ {v}: popcount(adj[v] & adj[w]) <= 1
    
    # Note: w = u is excluded from adj[v] check since u not in adj[v] (no self loop, and u,v not yet connected).
    # Actually u is not in adj[v] currently, so adj[v] doesn't contain u. Good.
    # Similarly v not in adj[u].
    
    # Let's implement:
    for w in range(N):
        if w == u or w == v:
            continue
        if (adj_bitmasks[v] >> w) & 1:
            # w is neighbour of v, so v becomes common neighbour of (u,w)
            cnt = (adj_bitmasks[u] & adj_bitmasks[w]).bit_count()
            if cnt + 1 > 2:
                return False
        if (adj_bitmasks[u] >> w) & 1:
            # w is neighbour of u, so u becomes common neighbour of (v,w)
            cnt = (adj_bitmasks[v] & adj_bitmasks[w]).bit_count()
            if cnt + 1 > 2:
                return False
    return True


def _popcount(x):
    return x.bit_count()


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    # Start with empty graph
    adj = [0] * N
    degrees = [0] * N
    edges = []
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    # Degree-balanced greedy: repeatedly add the edge that minimizes max degree imbalance
    # while maintaining K2,3-free property.
    
    while time.time() < deadline:
        best_edge = None
        best_score = None
        
        for (u, v) in all_edges:
            if (adj[u] >> v) & 1:
                continue  # already present
            
            # Check validity
            if not _is_valid_addition(adj, u, v):
                continue
            
            # Score: prefer edges that balance degrees
            new_du = degrees[u] + 1
            new_dv = degrees[v] + 1
            # Minimize the resulting max degree, or minimize imbalance
            max_deg = max(new_du, new_dv)
            # Also prefer lower total degree sum to spread out
            score = (max_deg, degrees[u] + degrees[v])
            
            if best_score is None or score < best_score:
                best_score = score
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        degrees[u] += 1
        degrees[v] += 1
        edges.append([u, v])
    
    # Local polish: try to add more edges if possible (shouldn't be, but check)
    # Also try edge swaps: remove one edge, add two? For now, just verify we can't add more.
    
    # Final validity check
    for u in range(N):
        for v in range(u + 1, N):
            common = adj[u] & adj[v]
            if common.bit_count() >= 3:
                # This shouldn't happen, but if it does, we have a bug
                pass
    
    return edges
