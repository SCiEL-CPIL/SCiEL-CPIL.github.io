import time
import random
import itertools

_START = time.time()
_DEADLINE = _START + 145

def _has_path_5(u, v, adj_masks):
    """Check if there is a simple path of exactly 5 edges from u to v.
    This means adding edge (u,v) would create a C6."""
    # We need a path u = v0, v1, v2, v3, v4, v5 = v with all distinct vertices
    # and (u,v) not already in the path (it's not since we're checking before adding)
    
    # DFS with bitmask
    # State: (current_node, depth, visited_mask)
    # We start at u with depth 0, need to reach v at depth 5
    
    # Quick rejection: if u and v have no common neighbors, can't have path of length 2
    # But we need length 5, so this doesn't directly help
    
    # Use iterative DFS with explicit stack
    # Optimize: at depth 4, we just need to check if neighbor is v
    
    stack = [(u, 0, 1 << u)]
    while stack:
        node, depth, mask = stack.pop()
        if depth == 5:
            if node == v:
                return True
            continue
        if depth == 4:
            # Next node must be v
            if adj_masks[node] & (1 << v) and not (mask & (1 << v)):
                return True
            continue
        neighbors = adj_masks[node]
        while neighbors:
            bit = neighbors & -neighbors
            nb = bit.bit_length() - 1
            neighbors ^= bit
            if mask & (1 << nb):
                continue
            if nb == v and depth != 4:
                continue
            stack.append((nb, depth + 1, mask | (1 << nb)))
    return False

def _has_path_5_fast(u, v, adj_masks):
    """Faster C6 check using bidirectional BFS from both ends."""
    # Path of length 5: u - a - b - c - d - v
    # We can split: find a neighbor a of u, then check if there's a path of length 3 from a to v avoiding u and v
    # Or: find path of length 2 from u and path of length 3 from v and check intersection
    
    # Better: enumerate all length-2 paths from u: u-a-b
    # Then check if there's a path of length 3 from b to v avoiding u, a, and all vertices in the length-2 path
    # Actually, let's do: for each neighbor a of u, for each neighbor b of a (b != u),
    #   check if there's a path of length 3 from b to v avoiding {u, a, b}
    
    # Even better: precompute nothing, just do smart DFS
    # Let me try a different approach: meet in the middle
    # Path length 5: u-a-b-c-d-v
    # Split at b: path of length 2 from u to b (u-a-b), path of length 3 from b to v (b-c-d-v)
    # We need all 6 vertices distinct
    
    # Enumerate all pairs (a, b) where a is neighbor of u, b is neighbor of a, b != u
    # Then for each such b, check if there's a path of length 3 from b to v avoiding {u, a, b}
    
    # Path of length 3 from b to v: b-c-d-v where c is neighbor of b, d is neighbor of c, v is neighbor of d
    # All distinct from {u, a, b}
    
    forbidden_base = (1 << u) | (1 << v)  # u and v are forbidden in the middle
    
    for a in range(40):
        if not (adj_masks[u] & (1 << a)):
            continue
        forbidden_a = forbidden_base | (1 << a)
        for b in range(40):
            if b == u or b == v or b == a:
                continue
            if not (adj_masks[a] & (1 << b)):
                continue
            forbidden_b = forbidden_a | (1 << b)
            # Now check if there's a path b-c-d-v of length 3
            # c is neighbor of b, c not in forbidden_b
            # d is neighbor of c, d not in forbidden_b, d != c
            # v is neighbor of d
            for c in range(40):
                if forbidden_b & (1 << c):
                    continue
                if not (adj_masks[b] & (1 << c)):
                    continue
                forbidden_c = forbidden_b | (1 << c)
                # d must be neighbor of c and neighbor of v, d not in forbidden_c
                common = adj_masks[c] & adj_masks[v]
                common &= ~forbidden_c
                if common:
                    return True
    return False

def _greedy_construct(n=40, seed=0, deadline=None):
    if deadline is None:
        deadline = _DEADLINE
    rng = random.Random(seed)
    adj_masks = [0] * n
    edges = set()
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(all_edges)
    
    # Greedy Phase
    for u, v in all_edges:
        if time.time() > deadline - 2:
            break
        if not _has_path_5_fast(u, v, adj_masks):
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            edges.add((u, v))
            
    return edges, adj_masks

def _local_search(edges, adj_masks, n=40, seed=0, deadline=None):
    if deadline is None:
        deadline = _DEADLINE
    rng = random.Random(seed)
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    non_edges = [(u, v) for u, v in all_edges if (u, v) not in edges]
    rng.shuffle(non_edges)
    
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for u, v in non_edges:
            if (u, v) in edges:
                continue
            if not _has_path_5_fast(u, v, adj_masks):
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges.add((u, v))
                improved = True
                
    # Try edge swaps: remove one edge, try to add two non-edges
    if time.time() < deadline - 3:
        for _ in range(100):
            if time.time() > deadline - 3:
                break
            edge_list = list(edges)
            if not edge_list:
                break
            ru, rv = rng.choice(edge_list)
            adj_masks[ru] &= ~(1 << rv)
            adj_masks[rv] &= ~(1 << ru)
            edges.discard((ru, rv))
            
            # Try to add non-edges
            added = 0
            for u, v in non_edges:
                if (u, v) in edges:
                    continue
                if not _has_path_5_fast(u, v, adj_masks):
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    edges.add((u, v))
                    added += 1
                    if added >= 2:
                        break
            if added < 2:
                # Try more
                for u, v in non_edges:
                    if (u, v) in edges:
                        continue
                    if not _has_path_5_fast(u, v, adj_masks):
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)
                        edges.add((u, v))
                        added += 1
                        if added >= 2:
                            break
    
    return edges, adj_masks

def _simulated_annealing(edges, adj_masks, n=40, seed=0, deadline=None):
    if deadline is None:
        deadline = _DEADLINE
    rng = random.Random(seed)
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    non_edges = [(u, v) for u, v in all_edges if (u, v) not in edges]
    rng.shuffle(non_edges)
    
    current_edges = len(edges)
    best_edges = len(edges)
    best_adj = adj_masks[:]
    best_edge_set = set(edges)
    
    T = 5.0
    cooling = 0.999
    
    while time.time() < deadline - 3:
        T *= cooling
        if T < 0.01:
            break
            
        # Random move: either add a non-edge, remove an edge, or swap
        move_type = rng.random()
        
        if move_type < 0.4:
            # Try to add a non-edge
            if not non_edges:
                continue
            u, v = rng.choice(non_edges)
            if (u, v) in edges:
                continue
            if _has_path_5_fast(u, v, adj_masks):
                continue
            # Accept the move
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            edges.add((u, v))
            non_edges.remove((u, v))
            current_edges += 1
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj_masks[:]
                best_edge_set = set(edges)
                
        elif move_type < 0.7:
            # Try to remove an edge
            edge_list = list(edges)
            if not edge_list:
                continue
            ru, rv = rng.choice(edge_list)
            adj_masks[ru] &= ~(1 << rv)
            adj_masks[rv] &= ~(1 << ru)
            edges.discard((ru, rv))
            non_edges.append((ru, rv))
            current_edges -= 1
            # Always accept removal (it's a valid move)
            
        else:
            # Swap: remove one edge, try to add two non-edges
            edge_list = list(edges)
            if not edge_list:
                continue
            ru, rv = rng.choice(edge_list)
            adj_masks[ru] &= ~(1 << rv)
            adj_masks[rv] &= ~(1 << ru)
            edges.discard((ru, rv))
            non_edges.append((ru, rv))
            current_edges -= 1
            
            # Try to add non-edges
            added = 0
            for u, v in non_edges:
                if (u, v) in edges:
                    continue
                if not _has_path_5_fast(u, v, adj_masks):
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    edges.add((u, v))
                    non_edges.remove((u, v))
                    current_edges += 1
                    added += 1
                    if added >= 2:
                        break
            
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj_masks[:]
                best_edge_set = set(edges)
    
    return best_edge_set, best_adj, best_edges

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 145
    
    best_edges = set()
    best_count = 0
    
    n = 40
    
    # Phase 1: Multiple greedy constructions with different seeds
    for seed in range(200):
        if time.time() > _DEADLINE - 5:
            break
        try:
            edges, adj_masks = _greedy_construct(n, seed)
            count = len(edges)
            if count > best_count:
                best_count = count
                best_edges = set(edges)
                if count >= 105:
                    break
        except Exception:
            continue
    
    # Phase 2: Local search on the best result
    if time.time() < _DEADLINE - 5:
        try:
            edges, adj_masks = _local_search(set(best_edges), [0]*n, n, 0)
            # Rebuild adj_masks
            adj_masks = [0] * n
            for u, v in edges:
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
            count = len(edges)
            if count > best_count:
                best_count = count
                best_edges = set(edges)
        except Exception:
            pass
    
    # Phase 3: More greedy + local search combinations
    for seed in range(200, 1000):
        if time.time() > _DEADLINE - 3:
            break
        try:
            edges, adj_masks = _greedy_construct(n, seed)
            if time.time() < _DEADLINE - 3:
                edges, adj_masks = _local_search(edges, adj_masks, n, seed)
            count = len(edges)
            if count > best_count:
                best_count = count
                best_edges = set(edges)
                if count >= 105:
                    break
        except Exception:
            continue
    
    return [[u, v] for u, v in best_edges]
