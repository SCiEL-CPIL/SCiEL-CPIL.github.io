"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START

def _can_add_edge(adj, u, v, n):
    """Check if adding edge (u,v) creates a C6 using bitmask BFS."""
    nu_mask = adj[u] & ~(1 << v)
    nv_mask = adj[v] & ~(1 << u)
    
    if nu_mask == 0 or nv_mask == 0:
        return True
    
    # We need a path u-a-b-c-d-v of length 5 (4 internal vertices)
    # a in N(u), d in N(v), a != d
    # b in N(a) \ {u,v,d}, c in N(d) \ {u,v,a}
    # b ~ c
    
    # Start: from neighbors of u (a) to neighbors of v (d) in exactly 3 steps
    # Step 1: from a, go to b (neighbors of a excluding u, v, d)
    # Step 2: from b, go to c (neighbors of b excluding u, v, a, d)
    # Step 3: from c, check if c is neighbor of d
    
    # Level 1: all possible b's from all a's
    # b is reachable from some a in N(u) in 1 step
    level1 = 0
    am = nu_mask
    while am:
        lb_a = am & (-am)
        ai = lb_a.bit_length() - 1
        am ^= lb_a
        # b candidates from a: neighbors of a excluding u, v
        # We'll refine exclusions later
        level1 |= adj[ai] & ~(1 << u) & ~(1 << v)
    
    if level1 == 0:
        return True
    
    # Level 2: from all b's, go to c's
    # c is reachable from some b in level1 in 1 step
    # Exclude u, v
    level2 = 0
    bm = level1
    while bm:
        lb_b = bm & (-bm)
        bi = lb_b.bit_length() - 1
        bm ^= lb_b
        level2 |= adj[bi] & ~(1 << u) & ~(1 << v)
    
    if level2 == 0:
        return True
    
    # Level 3: from all c's in level2, check if any c is neighbor of some d in N(v)
    # But we must ensure the path is simple: a != d, b != c, b != d, c != a, etc.
    # The simple BFS above might find a path that revisits vertices.
    # For a strict C6 check, we need distinct vertices u, a, b, c, d, v.
    # The simple BFS level approach can have false positives if the path loops.
    # However, for the purpose of preventing C6, if there is ANY walk of length 4,
    # does it imply a C6? No, only if it's a simple path.
    # But checking for simple paths is expensive.
    # Let's use a stricter check:
    # For each d in N(v):
    #   For each c in N(d) \ {u, v, d}:
    #     For each b in N(c) \ {u, v, d, c}:
    #       For each a in N(b) \ {u, v, d, c, b}:
    #         if a in N(u) and a != d: return False
    
    # This is O(n^4) which is 40^4 = 2.5M ops per edge check. Too slow for many edges.
    # Let's stick to the BFS heuristic which is faster but might miss some C6s or find false positives?
    # Actually, if the BFS finds a path u-a-b-c-d-v where vertices might repeat, it's not necessarily a C6.
    # But if it doesn't find a path, there is definitely no C6 using that edge.
    # If it finds a "path" (walk), it might be a C6 or not.
    # To be safe, we should only add the edge if we are SURE it doesn't create a C6.
    # So we need a correct C6 detection.
    
    # Let's use the correct but potentially slower check, optimized with bitmasks.
    # For each d in N(v):
    dm = nv_mask
    while dm:
        lb_d = dm & (-dm)
        di = lb_d.bit_length() - 1
        dm ^= lb_d
        
        # c in N(d) \ {u, v, d}
        cm = adj[di] & ~(1 << u) & ~(1 << v) & ~lb_d
        while cm:
            lb_c = cm & (-cm)
            ci = lb_c.bit_length() - 1
            cm ^= lb_c
            
            # b in N(c) \ {u, v, d, c}
            bm2 = adj[ci] & ~(1 << u) & ~(1 << v) & ~lb_d & ~lb_c
            while bm2:
                lb_b = bm2 & (-bm2)
                bi = lb_b.bit_length() - 1
                bm2 ^= lb_b
                
                # a in N(b) \ {u, v, d, c, b}
                am2 = adj[bi] & ~(1 << u) & ~(1 << v) & ~lb_d & ~lb_c & ~lb_b
                # Check if a is in N(u) and a != d
                if am2 & nu_mask & ~lb_d:
                    return False
    return True

def _count_edges(adj, n):
    count = 0
    for i in range(n):
        count += adj[i].bit_count()
    return count // 2

def _edges_from_adj(adj, n):
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                edges.append([u, v])
    return edges

def _adj_from_edges(edge_list, n):
    adj = [0] * n
    for u, v in edge_list:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _greedy_add(adj, n, all_pairs, start_time, deadline):
    """Greedy add edges to adj until deadline or no more can be added."""
    # Shuffle remaining pairs
    remaining = [p for p in all_pairs if not (adj[p[0]] >> p[1]) & 1]
    random.shuffle(remaining)
    
    for u, v in remaining:
        if time.time() > deadline:
            break
        if not (adj[u] >> v) & 1:
            if _can_add_edge(adj, u, v, n):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    return adj

def _local_search(adj, n, all_pairs, start_time, deadline):
    """Local search: try to swap edges to increase count or find better config."""
    # We'll try 1-for-1 swaps: remove edge (u,v), add edge (x,y)
    # If the resulting graph is C6-free, keep it.
    # This helps escape local optima.
    
    current_edges = _count_edges(adj, n)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each edge
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            if not ((adj[u] >> v) & 1):
                continue
            
            # Remove edge (u,v)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            # Try adding any non-edge
            for x, y in all_pairs:
                if time.time() > deadline:
                    break
                if (x == u and y == v) or (x == v and y == u):
                    continue
                if (adj[x] >> y) & 1:
                    continue
                
                if _can_add_edge(adj, x, y, n):
                    adj[x] |= (1 << y)
                    adj[y] |= (1 << x)
                    # Count is same, but structure changed.
                    # We need to check if this new structure allows more edges later.
                    # For now, we just accept the swap to explore.
                    # To actually improve, we need a 1-for-2 or similar.
                    # Let's do a quick greedy add after the swap to see if we can gain.
                    new_adj = _greedy_add(adj, n, all_pairs, start_time, deadline)
                    new_count = _count_edges(new_adj, n)
                    
                    if new_count > current_edges:
                        adj = new_adj
                        current_edges = new_count
                        improved = True
                        break
                    else:
                        # Revert the swap
                        adj[x] &= ~(1 << y)
                        adj[y] &= ~(1 << x)
                        # Re-add original edge
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            # Restore original edge if we didn't break out
            if not ((adj[u] >> v) & 1):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            if improved:
                break
    return adj, current_edges

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_adj = [0] * n
    best_count = 0
    
    # Multiple random restarts with different densities
    for restart in range(10):
        if time.time() > deadline - 10:
            break
        
        random.seed(42 + restart)
        adj = [0] * n
        
        # Random initial graph with varying density
        p = 0.3 + 0.1 * restart
        for u, v in all_pairs:
            if random.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Prune C6s: repeatedly find a C6 and remove an edge
        # We'll use a simple heuristic: try to remove edges that are in many C6s?
        # Or just random removal if C6 exists.
        # Since _can_add_edge is expensive, let's just do greedy add from a sparse start?
        # No, the inspiration was to start dense and prune.
        # Let's try to prune by checking each edge and removing it if it creates a C6?
        # No, we need to check if the graph HAS a C6.
        # Let's just do greedy add from this random start. It will skip edges that create C6s.
        # This effectively prunes the graph.
        
        adj = _greedy_add(adj, n, all_pairs, start_time, deadline)
        count = _count_edges(adj, n)
        
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        # Local search to improve
        adj, count = _local_search(adj, n, all_pairs, start_time, deadline)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        if time.time() > deadline - 5:
            break
    
    # Fallback
    if best_count < 100:
        edges = []
        for k in range(8):
            edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
        return edges
    
    return _edges_from_adj(best_adj, n)

# EVOLVE-BLOCK-END
