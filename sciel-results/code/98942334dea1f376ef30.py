"""Dense K3,3-free graphs on 20 vertices using bitmask representation."""
# EVOLVE-BLOCK-START
import time
import random

def _is_valid(adj, n):
    """Check if graph is K3,3-free using bitmasks."""
    for a in range(n):
        for b in range(a + 1, n):
            common_ab = adj[a] & adj[b]
            if common_ab.bit_count() < 2:
                continue
            for c in range(b + 1, n):
                if (common_ab & adj[c]).bit_count() >= 3:
                    return False
    return True

def _count_edges(adj):
    return sum(a.bit_count() for a in adj) // 2

def _try_add(adj, u, v, n):
    """Try adding edge (u,v). Returns True if still valid."""
    if adj[u] & (1 << v):
        return True
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    if _is_valid(adj, n):
        return True
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)
    return False

def _greedy_fill_exhaustive(adj, n, deadline):
    """Exhaustively try adding all missing edges in random order."""
    missing = [(u, v) for u in range(n) for v in range(u + 1, n)
               if not (adj[u] & (1 << v))]
    random.shuffle(missing)
    for u, v in missing:
        if time.time() > deadline:
            break
        _try_add(adj, u, v, n)

def _two_edge_swap(adj, n, deadline, max_swaps=40):
    """Try 2-edge swaps: remove one edge, then greedily add up to 2 new edges."""
    existing = [(u, v) for u in range(n) for v in range(u + 1, n)
                if adj[u] & (1 << v)]
    if not existing:
        return adj, False
    random.shuffle(existing)
    
    for rem_u, rem_v in existing[:max_swaps]:
        if time.time() > deadline:
            break
        orig_count = _count_edges(adj)
        
        adj[rem_u] &= ~(1 << rem_v)
        adj[rem_v] &= ~(1 << rem_u)
        
        missing = [(u, v) for u in range(n) for v in range(u + 1, n)
                   if not (adj[u] & (1 << v))]
        random.shuffle(missing)
        added = 0
        for u, v in missing:
            if time.time() > deadline:
                break
            if _try_add(adj, u, v, n):
                added += 1
                if added >= 2:
                    break
        
        new_count = _count_edges(adj)
        if new_count > orig_count:
            return adj, True
        
        adj[rem_u] |= (1 << rem_v)
        adj[rem_v] |= (1 << rem_u)
    
    return adj, False

def _one_edge_swap(adj, n, deadline, max_swaps=30):
    """Try 1-edge swaps: remove one edge, greedily fill, check if better."""
    existing = [(u, v) for u in range(n) for v in range(u + 1, n)
                if adj[u] & (1 << v)]
    if not existing:
        return adj, False
    random.shuffle(existing)
    
    for rem_u, rem_v in existing[:max_swaps]:
        if time.time() > deadline:
            break
        orig_count = _count_edges(adj)
        
        adj[rem_u] &= ~(1 << rem_v)
        adj[rem_v] &= ~(1 << rem_u)
        
        _greedy_fill_exhaustive(adj, n, deadline)
        
        new_count = _count_edges(adj)
        if new_count > orig_count:
            return adj, True
        
        adj[rem_u] |= (1 << rem_v)
        adj[rem_v] |= (1 << rem_u)
    
    return adj, False

def _random_graph(n, density=0.4):
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < density:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    return adj

def _build_edges_from_adj(adj, n):
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                edges.append([u, v])
    return edges

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_adj = [0] * n
    best_count = 0
    
    # Seed with K2,18
    for a in range(2):
        for b in range(2, n):
            best_adj[a] |= (1 << b)
            best_adj[b] |= (1 << a)
    best_count = _count_edges(best_adj)
    
    # Try a random start
    rand_adj = _random_graph(n, 0.35)
    if _is_valid(rand_adj, n):
        rc = _count_edges(rand_adj)
        if rc > best_count:
            best_count = rc
            best_adj = rand_adj[:]
    
    current_adj = best_adj[:]
    no_improve = 0
    max_no_improve = 100
    
    while time.time() < deadline:
        # Phase 1: Exhaustive greedy fill
        _greedy_fill_exhaustive(current_adj, n, deadline)
        
        # Phase 2: 2-edge swaps
        current_adj, improved = _two_edge_swap(current_adj, n, deadline, max_swaps=40)
        if improved:
            no_improve = 0
        else:
            no_improve += 1
        
        # Phase 3: 1-edge swaps
        if time.time() < deadline:
            current_adj, improved = _one_edge_swap(current_adj, n, deadline, max_swaps=30)
            if improved:
                no_improve = 0
            else:
                no_improve += 1
        
        # Track best
        cur_count = _count_edges(current_adj)
        if cur_count > best_count:
            best_count = cur_count
            best_adj = current_adj[:]
            no_improve = 0
        
        if no_improve >= max_no_improve:
            if time.time() > deadline - 5:
                break
            
            if random.random() < 0.5:
                current_adj = _random_graph(n, random.uniform(0.25, 0.5))
                if not _is_valid(current_adj, n):
                    current_adj = [0] * n
                    for a in range(2):
                        for b in range(2, n):
                            current_adj[a] |= (1 << b)
                            current_adj[b] |= (1 << a)
            else:
                existing = [(u, v) for u in range(n) for v in range(u + 1, n)
                           if current_adj[u] & (1 << v)]
                random.shuffle(existing)
                for u, v in existing[:random.randint(2, 4)]:
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
            
            no_improve = 0
    
    return _build_edges_from_adj(best_adj, n)
# EVOLVE-BLOCK-END
