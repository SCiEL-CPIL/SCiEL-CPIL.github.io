import time
import random

_START = time.time()
_DEADLINE = _START + 108


def _can_add_edge(u, v, adj_masks):
    """
    Check if adding edge (u, v) creates a C6.
    Equivalent to checking if there is a path of length 4 from u to v
    in G \\ {u, v}.
    """
    # Fast rejection: if u and v are already connected by a path of length
    # 1, 2, or 3, adding (u,v) would create a cycle of length < 6. But the
    # graph is simple, so u-v edge doesn't exist yet. We need to check for
    # paths of length 2, 3, or 4? No, only C6 matters. Adding (u,v) creates
    # a C6 iff there is a path u-a-b-c-d-v where a,b,c,d are distinct and
    # not u,v. This is a path of length 5 between u and v in G.
    
    # But we also need to make sure no shorter cycles are formed? The task
    # says "containing no six-cycle". It doesn't say no other cycles. So we
    # only care about C6.
    
    # DFS/BFS from u, looking for v at exactly distance 5, with all
    # intermediate nodes distinct and not u,v.
    
    # For efficiency, we can do bidirectional or use the mask approach.
    
    # Approach: BFS from u up to depth 4, tracking visited masks.
    # At depth 5, check if we can reach v.
    
    # Actually, let's do a simple DFS with pruning.
    
    initial_mask = (1 << u) | (1 << v)
    stack = []
    
    # Start from neighbors of u (excluding v)
    neighbors_u = adj_masks[u] & ~(1 << v)
    if not neighbors_u:
        return False
    
    while neighbors_u:
        bit = neighbors_u & -neighbors_u
        a = bit.bit_length() - 1
        neighbors_u ^= bit
        stack.append((a, 1, initial_mask | (1 << a)))
    
    while stack:
        node, depth, mask = stack.pop()
        
        if depth == 4:
            # Check if node is connected to v
            if adj_masks[node] & (1 << v):
                return True
            continue
        
        # Get unvisited neighbors
        neighbors = adj_masks[node] & ~mask
        while neighbors:
            bit = neighbors & -neighbors
            nb = bit.bit_length() - 1
            neighbors ^= bit
            stack.append((nb, depth + 1, mask | (1 << nb)))
    
    return False


def _greedy_with_local_search(n=40, seed=0, max_iter=200):
    rng = random.Random(seed)
    adj_masks = [0] * n
    edges = set()
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(all_edges)
    
    # Greedy Phase
    for u, v in all_edges:
        if not _can_add_edge(u, v, adj_masks):
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            edges.add((u, v))
    
    # Local Search
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter and time.time() < _DEADLINE - 3:
        improved = False
        iter_count += 1
        
        # Phase 1: Try to add any non-edge
        non_edges = [(u, v) for u, v in all_edges if (u, v) not in edges]
        rng.shuffle(non_edges)
        
        added = False
        for u, v in non_edges:
            if not _can_add_edge(u, v, adj_masks):
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges.add((u, v))
                added = True
                improved = True
                break
        
        if added:
            continue
        
        # Phase 2: Try 1-swap: remove one edge, add one non-edge
        edge_list = list(edges)
        rng.shuffle(edge_list)
        
        swapped = False
        for u, v in edge_list:
            # Remove edge (u, v)
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)
            edges.remove((u, v))
            
            # Try to add a non-edge
            non_edges_curr = [(x, y) for x, y in all_edges if (x, y) not in edges]
            rng.shuffle(non_edges_curr)
            
            added_back = False
            for x, y in non_edges_curr:
                if not _can_add_edge(x, y, adj_masks):
                    adj_masks[x] |= (1 << y)
                    adj_masks[y] |= (1 << x)
                    edges.add((x, y))
                    added_back = True
                    improved = True
                    break
            
            if not added_back:
                # Revert
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges.add((u, v))
            
            if improved:
                break
    
    return list(edges)


def _spectral_init(n=40, target_edges=105, seed=0):
    """
    Initialize with a random regular-ish graph with approximately target_edges edges.
    """
    rng = random.Random(seed)
    adj_masks = [0] * n
    edges = set()
    
    # Start with a random graph with about target_edges edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(all_edges)
    
    for u, v in all_edges[:target_edges]:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        edges.add((u, v))
    
    return adj_masks, edges


def _polish(adj_masks, edges, n=40, max_iter=50, seed=0):
    """
    Final polish: try to add edges, then try swaps.
    """
    rng = random.Random(seed)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter and time.time() < _DEADLINE - 2:
        improved = False
        iter_count += 1
        
        # Try to add
        non_edges = [(u, v) for u, v in all_edges if (u, v) not in edges]
        rng.shuffle(non_edges)
        for u, v in non_edges:
            if not _can_add_edge(u, v, adj_masks):
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges.add((u, v))
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-swap
        edge_list = list(edges)
        rng.shuffle(edge_list)
        for u, v in edge_list:
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)
            edges.remove((u, v))
            
            non_edges_curr = [(x, y) for x, y in all_edges if (x, y) not in edges]
            rng.shuffle(non_edges_curr)
            for x, y in non_edges_curr:
                if not _can_add_edge(x, y, adj_masks):
                    adj_masks[x] |= (1 << y)
                    adj_masks[y] |= (1 << x)
                    edges.add((x, y))
                    improved = True
                    break
            
            if not improved:
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges.add((u, v))
            
            if improved:
                break
    
    return list(edges)


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 108
    
    best_edges = []
    best_count = 0
    
    for seed in range(2000):
        if time.time() > _DEADLINE - 2:
            break
        
        try:
            # Use different initialization strategies
            if seed % 3 == 0:
                # Pure random greedy
                edges = _greedy_with_local_search(40, seed, max_iter=150)
            elif seed % 3 == 1:
                # Spectral-ish init then polish
                adj_masks, edges_set = _spectral_init(40, 100, seed)
                edges = _polish(adj_masks, edges_set, 40, max_iter=80, seed=seed)
            else:
                # Greedy with more iterations
                edges = _greedy_with_local_search(40, seed, max_iter=300)
            
            count = len(edges)
            if count > best_count:
                best_count = count
                best_edges = edges
                if best_count >= 110:
                    break
        except Exception:
            continue
    
    return [[u, v] for u, v in best_edges]
