import time
import random
import sys

def _make_adj(edges, n=16):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_triangle_free_add(adj, u, v):
    return (adj[u] & adj[v]) == 0

def _is_triangle_free(adj):
    n = len(adj)
    for u in range(n):
        mask = adj[u]
        while mask:
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            mask ^= lb
            if adj[v] & adj[u] & ~(1 << u) & ~(1 << v):
                return False
    return True

def _is_3_colorable(adj, deadline=None):
    n = len(adj)
    color = [0] * n
    # Precompute adjacency masks
    adj_masks = adj[:]
    
    def backtrack():
        if deadline and time.time() > deadline:
            return None
        
        # MRV with bit operations
        best_v = -1
        min_avail = 4
        best_avail_colors = 0
        
        for v in range(n):
            if color[v] == 0:
                unavail = 0
                mask = adj_masks[v]
                while mask:
                    lb = mask & (-mask)
                    u = lb.bit_length() - 1
                    mask ^= lb
                    if color[u] != 0:
                        unavail |= (1 << (color[u]-1))
                avail_colors = 7 ^ unavail
                count = bin(avail_colors).count('1')
                if count == 0:
                    return False
                if count < min_avail:
                    min_avail = count
                    best_v = v
                    best_avail_colors = avail_colors
                    if count == 1:
                        break
        
        if best_v == -1:
            return True
            
        c = 1
        while c <= 3:
            if best_avail_colors & (1 << (c-1)):
                color[best_v] = c
                result = backtrack()
                if result is not None:
                    return result
                color[best_v] = 0
            c += 1
        return False

    result = backtrack()
    return result if result is not None else True

def _build_grotzsch_edges():
    """
    Returns edges of the Grötzsch graph (Mycielski of C5).
    Vertices 0..4 form C5.
    Vertices 5..9 are clones.
    Vertex 10 is the root.
    """
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    edges = [list(e) for e in cycle]
    for u, w in cycle:
        edges += [[u, 5 + w], [w, 5 + u]]
    edges += [[10, 5 + u] for u in range(5)]
    return edges

def _greedy_complete_with_core(adj, edges, edge_set, deadline, seed=0):
    """
    Greedily add edges that maintain triangle-freeness and non-3-colorability.
    Uses a fixed order for determinism within a seed.
    """
    n = 16
    rng = random.Random(seed)
    
    # Generate all possible edges not yet in the graph
    all_possible = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in edge_set]
    rng.shuffle(all_possible)
    
    added = 0
    for u, v in all_possible:
        if time.time() > deadline:
            break
        if _is_triangle_free_add(adj, u, v):
            # Tentatively add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not _is_3_colorable(adj, deadline=deadline):
                edge_set.add((u, v))
                edges.append([u, v])
                added += 1
            else:
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
    
    return edges, adj, edge_set, added

def _local_search_improve(adj, edges, edge_set, count, deadline, seed=0):
    """
    Local search: try adding edges, and if stuck, try removing one edge and re-adding.
    """
    n = 16
    rng = random.Random(seed)
    improved = True
    total_added = 0
    
    while improved and time.time() < deadline:
        improved = False
        # Try adding edges in random order
        all_possible = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in edge_set]
        rng.shuffle(all_possible)
        
        for u, v in all_possible:
            if time.time() > deadline:
                break
            if _is_triangle_free_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if not _is_3_colorable(adj, deadline=deadline):
                    edge_set.add((u, v))
                    edges.append([u, v])
                    count += 1
                    improved = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        # If no improvement, try edge swap
        if not improved and time.time() < deadline - 5 and edges:
            # Remove a random edge
            idx = rng.randrange(len(edges))
            u, v = edges.pop(idx)
            edge_set.discard((u, v))
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            count -= 1
            
            # Re-add edges greedily
            all_possible = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in edge_set]
            rng.shuffle(all_possible)
            
            for eu, ev in all_possible:
                if time.time() > deadline:
                    break
                if _is_triangle_free_add(adj, eu, ev):
                    adj[eu] |= (1 << ev)
                    adj[ev] |= (1 << eu)
                    if not _is_3_colorable(adj, deadline=deadline):
                        edge_set.add((eu, ev))
                        edges.append([eu, ev])
                        count += 1
            
            if count > 0:
                improved = True
    
    return edges, adj, edge_set, count

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n = 16
    
    # Start with Grötzsch graph on vertices 0..10
    base_edges = _build_grotzsch_edges()
    best_edges = list(base_edges)
    best_adj = _make_adj(best_edges, n)
    best_edge_set = set((min(u,v), max(u,v)) for u,v in best_edges)
    best_count = len(best_edges)
    
    # Function to validate and update best
    def try_update(edges, adj, edge_set, count):
        nonlocal best_edges, best_adj, best_edge_set, best_count
        if count > best_count:
            if _is_triangle_free(adj) and not _is_3_colorable(adj, deadline=start_time + 115.0):
                best_edges = edges
                best_adj = adj
                best_edge_set = edge_set
                best_count = count
                return True
        return False
    
    # Phase 1: Multiple greedy completions from the core
    for seed in range(80):
        if time.time() > deadline - 20:
            break
        edges = list(base_edges)
        adj = _make_adj(edges, n)
        edge_set = set((min(u,v), max(u,v)) for u,v in edges)
        edges, adj, edge_set, _ = _greedy_complete_with_core(adj, edges, edge_set, deadline, seed)
        try_update(edges, adj, edge_set, len(edges))
    
    # Phase 2: Local search on the best found so far
    if time.time() < deadline - 10:
        edges = list(best_edges)
        adj = _make_adj(edges, n)
        edge_set = set((min(u,v), max(u,v)) for u,v in edges)
        edges, adj, edge_set, count = _local_search_improve(adj, edges, edge_set, len(edges), deadline, seed=123)
        try_update(edges, adj, edge_set, count)
    
    # Phase 3: More local search with different seeds
    if time.time() < deadline - 5:
        for seed in range(5, 20):
            if time.time() > deadline - 2:
                break
            edges = list(best_edges)
            adj = _make_adj(edges, n)
            edge_set = set((min(u,v), max(u,v)) for u,v in edges)
            edges, adj, edge_set, count = _local_search_improve(adj, edges, edge_set, len(edges), deadline, seed=seed)
            try_update(edges, adj, edge_set, count)
    
    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Edges: {len(edges)}")
