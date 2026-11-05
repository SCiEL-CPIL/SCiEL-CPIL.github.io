import time
import random

_START = time.time()
_DEADLINE = _START + 115

def _has_path_5(u, v, adj_masks):
    """
    Returns True if there is a simple path of length 5 from u to v.
    """
    stack = [(u, 0, 1 << u)]
    while stack:
        node, depth, mask = stack.pop()
        if depth == 5:
            if node == v:
                return True
            continue
        
        if depth == 4:
            if adj_masks[node] & (1 << v) and not (mask & (1 << v)):
                return True
            continue
        
        if depth == 3:
            if adj_masks[node] & adj_masks[v] & ~mask:
                return True
            continue
        
        if depth == 2:
            neighbors_v = adj_masks[v] & ~mask
            while neighbors_v:
                bit = neighbors_v & -neighbors_v
                y = bit.bit_length() - 1
                neighbors_v ^= bit
                if adj_masks[y] & adj_masks[node] & ~(mask | (1 << v)):
                    return True
            continue
        
        neighbors = adj_masks[node] & ~mask
        while neighbors:
            bit = neighbors & -neighbors
            nb = bit.bit_length() - 1
            neighbors ^= bit
            if nb == v and depth < 4:
                continue
            stack.append((nb, depth + 1, mask | (1 << nb)))
    return False

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 115
    
    n = 40
    best_edges = []
    best_count = 0
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for seed in range(50000):
        if time.time() > _DEADLINE - 3:
            break
        
        rng = random.Random(seed)
        adj_masks = [0] * n
        edges_set = set()
        
        # Greedy Phase
        shuffled_edges = all_edges[:]
        rng.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if time.time() > _DEADLINE - 5:
                break
            if not _has_path_5(u, v, adj_masks):
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges_set.add((u, v))
        
        # Local Search: iteratively try to improve
        improved = True
        while improved and time.time() < _DEADLINE - 5:
            improved = False
            
            # Try adding non-edges
            non_edges = [(u, v) for u, v in all_edges if (u, v) not in edges_set]
            rng.shuffle(non_edges)
            
            for u, v in non_edges:
                if time.time() > _DEADLINE - 5:
                    break
                if not _has_path_5(u, v, adj_masks):
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    edges_set.add((u, v))
                    improved = True
            
            # Try 1-for-2 swaps: remove one edge, add two
            if time.time() > _DEADLINE - 5:
                break
            
            edge_list = list(edges_set)
            rng.shuffle(edge_list)
            
            for ru, rv in edge_list:
                if time.time() > _DEADLINE - 5:
                    break
                
                # Remove edge
                adj_masks[ru] &= ~(1 << rv)
                adj_masks[rv] &= ~(1 << ru)
                edges_set.discard((ru, rv))
                
                # Try to add non-edges
                non_edges = [(u, v) for u, v in all_edges if (u, v) not in edges_set]
                rng.shuffle(non_edges)
                
                added = 0
                for u, v in non_edges:
                    if time.time() > _DEADLINE - 5:
                        break
                    if not _has_path_5(u, v, adj_masks):
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)
                        edges_set.add((u, v))
                        added += 1
                        if added >= 2:
                            break
                
                if added == 0:
                    # Revert
                    adj_masks[ru] |= (1 << rv)
                    adj_masks[rv] |= (1 << ru)
                    edges_set.add((ru, rv))
                elif added >= 1:
                    improved = True
        
        count = len(edges_set)
        if count > best_count:
            best_count = count
            best_edges = list(edges_set)
            if best_count >= 106:
                pass

    return [[u, v] for u, v in best_edges]
