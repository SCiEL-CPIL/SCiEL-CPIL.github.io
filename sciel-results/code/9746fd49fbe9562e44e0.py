import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin before 150s limit
    
    n = 20
    best_edges = []
    best_score = 0
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Phase 1: Multiple greedy constructions with different strategies
    for seed in range(3000):
        if time.time() > deadline - 30.0:
            break
        
        if seed < 1500:
            # Random greedy
            edges = _greedy_random(n, all_edges, seed)
        else:
            # Degree-biased greedy (prefer low-degree endpoints)
            edges = _greedy_degree_biased(n, all_edges, seed)
        
        score = len(edges)
        
        if score > best_score:
            best_score = score
            best_edges = edges
    
    # Phase 2: Local search on the best solution
    if time.time() < deadline - 5.0 and best_edges:
        improved = _local_search(n, best_edges, all_edges, deadline - 3.0)
        if len(improved) > best_score:
            best_edges = improved
            best_score = len(improved)
    
    # Phase 3: If time remains, try local search on a few other good solutions
    if time.time() < deadline - 10.0:
        for seed in range(200, 400):
            if time.time() > deadline - 10.0:
                break
            edges = _greedy_degree_biased(n, all_edges, seed + 100000)
            if len(edges) >= best_score - 3:
                improved = _local_search(n, edges, all_edges, deadline - 5.0)
                if len(improved) > best_score:
                    best_edges = improved
                    best_score = len(improved)
    
    return best_edges

def _greedy_random(n, all_edges, seed):
    """Standard random-order greedy."""
    random.seed(seed)
    adj = [0] * n
    edges = []
    
    shuffled = all_edges[:]
    random.shuffle(shuffled)
    
    for u, v in shuffled:
        if _can_add(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
    
    return edges

def _greedy_degree_biased(n, all_edges, seed):
    """Greedy that prefers edges between lower-degree vertices."""
    random.seed(seed)
    adj = [0] * n
    edges = []
    deg = [0] * n
    
    # Shuffle all edges
    shuffled = all_edges[:]
    random.shuffle(shuffled)
    
    # Do passes, each time trying to add edges, prioritizing low-degree pairs
    for _pass in range(4):
        # Sort remaining edges by degree sum (ascending) + random tiebreak
        remaining = [e for e in shuffled if not ((1 << e[1]) & adj[e[0]])]
        if not remaining:
            break
        remaining.sort(key=lambda e: (deg[e[0]] + deg[e[1]], random.random()))
        
        added_this_pass = 0
        for u, v in remaining:
            if (1 << v) & adj[u]:
                continue
            if _can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                edges.append([u, v])
                added_this_pass += 1
        
        if added_this_pass == 0:
            break
    
    return edges

def _can_add(adj, u, v):
    """Check if adding edge (u,v) keeps graph K2,3-free. Uses bitmasks."""
    n = len(adj)
    # u and v must not already be adjacent
    if (1 << v) & adj[u]:
        return False
    
    # Check pair (u, v): common neighbors after adding edge
    # Common neighbors of u and v are vertices adjacent to both
    # After adding (u,v), u is now adjacent to v and v to u, but that doesn't
    # change common neighbors of u and v (common neighbors are OTHER vertices)
    if (adj[u] & adj[v]).bit_count() >= 3:
        return False
    
    # Check pairs (u, w) for all w != u, v
    # After adding (u,v): v becomes a neighbor of u
    # Common neighbors of u and w = (adj[u] & adj[w]) plus possibly v
    mask_v = 1 << v
    mask_u = 1 << u
    
    for w in range(n):
        if w == u or w == v:
            continue
        
        # Common neighbors of u and w after adding (u,v)
        # = (adj[u] & adj[w]) | {v if v in adj[w]}
        cuw = (adj[u] & adj[w]).bit_count()
        if adj[w] & mask_v:
            cuw += 1
        if cuw >= 3:
            return False
        
        # Common neighbors of v and w after adding (u,v)
        # = (adj[v] & adj[w]) | {u if u in adj[w]}
        cvw = (adj[v] & adj[w]).bit_count()
        if adj[w] & mask_u:
            cvw += 1
        if cvw >= 3:
            return False
    
    return True

def _local_search(n, edges, all_edges, deadline):
    """
    Local search: try to add single edges, then try 1-remove-2-add swaps.
    Optimized for speed with precomputed adjacency bitmasks.
    """
    adj = [0] * n
    edge_set = set()
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((min(u, v), max(u, v)))
    
    current_edges = [list(e) for e in edges]
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # Phase A: Try to add any single missing edge (repeated until none found)
        any_added = True
        while any_added and time.time() < deadline:
            any_added = False
            for u, v in all_edges:
                if time.time() > deadline:
                    return [[a, b] for a, b in edge_set]
                if (min(u, v), max(u, v)) in edge_set:
                    continue
                if _can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((min(u, v), max(u, v)))
                    any_added = True
                    improved = True
        
        if time.time() > deadline:
            return [[a, b] for a, b in edge_set]
        
        # Phase B: Try 1-remove-2-add swaps
        edge_list = list(edge_set)
        random.shuffle(edge_list)
        
        for e in edge_list:
            if time.time() > deadline:
                return [[a, b] for a, b in edge_set]
            
            u1, v1 = e
            
            # Remove edge e
            adj[u1] ^= (1 << v1)
            adj[v1] ^= (1 << u1)
            edge_set.remove(e)
            
            # Find candidates that can be added after removal
            candidates = []
            for u2, v2 in all_edges:
                if (min(u2, v2), max(u2, v2)) in edge_set:
                    continue
                if _can_add(adj, u2, v2):
                    candidates.append((u2, v2))
            
            # Try to find two compatible edges from candidates
            found = False
            max_check = min(len(candidates), 25)
            
            for i in range(max_check):
                if found:
                    break
                u2, v2 = candidates[i]
                
                # Tentatively add first candidate
                adj[u2] |= (1 << v2)
                adj[v2] |= (1 << u2)
                edge_set.add((min(u2, v2), max(u2, v2)))
                
                for j in range(i + 1, max_check):
                    u3, v3 = candidates[j]
                    if (min(u3, v3), max(u3, v3)) in edge_set:
                        continue
                    if _can_add(adj, u3, v3):
                        adj[u3] |= (1 << v3)
                        adj[v3] |= (1 << u3)
                        edge_set.add((min(u3, v3), max(u3, v3)))
                        found = True
                        break
                
                if not found:
                    # Revert first addition
                    adj[u2] ^= (1 << v2)
                    adj[v2] ^= (1 << u2)
                    edge_set.remove((min(u2, v2), max(u2, v2)))
            
            if found:
                # Net gain: removed 1, added 2 => +1 edge
                improved = True
                break
            else:
                # Revert removal
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                edge_set.add(e)
    
    return [[a, b] for a, b in edge_set]
