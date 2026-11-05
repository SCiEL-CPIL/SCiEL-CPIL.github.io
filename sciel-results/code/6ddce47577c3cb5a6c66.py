import time
import random

_START = time.time()
_DEADLINE = _START + 115

def _has_path_5(u, v, adj_masks):
    """
    Returns True if there is a simple path of length 5 from u to v.
    Uses bitmask for visited nodes.
    Optimized with pruning for depths 2, 3, 4.
    """
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
        
        if depth == 3:
            # Need path of length 2: node - x - v
            # x must be in adj(node) and adj(v) and not in mask
            if adj_masks[node] & adj_masks[v] & ~mask:
                return True
            continue
        
        if depth == 2:
            # Need path of length 3: node - x - y - v
            # y is in adj(v). x is in adj(y) and adj(node).
            # Check if any y in adj(v) \ mask has a common neighbor with node in \ mask
            neighbors_v = adj_masks[v] & ~mask
            while neighbors_v:
                bit = neighbors_v & -neighbors_v
                y = bit.bit_length() - 1
                neighbors_v ^= bit
                # x must be in adj(y) and adj(node) and not in mask
                # Note: x cannot be v (v is not in mask, but adj(y) might contain v? 
                # If y is neighbor of v, v is in adj(y). 
                # But we are looking for x. 
                # If x=v, then path is node-x-v... wait.
                # Path is node-x-y-v. 
                # x is intermediate.
                # adj(y) contains v. adj(node) might contain v? 
                # If adj(node) & adj(y) contains v, then x=v.
                # But v is the target. 
                # Is x allowed to be v? No, simple path.
                # v is not in mask. 
                # So we must exclude v from candidates for x.
                # mask | (1 << v)
                if adj_masks[y] & adj_masks[node] & ~(mask | (1 << v)):
                    return True
            continue
        
        # Depth 0 or 1
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
    
    # Pre-generate all edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Use multiple seeds
    for seed in range(20000):
        if time.time() > _DEADLINE - 2:
            break
        
        rng = random.Random(seed)
        adj_masks = [0] * n
        edges = set()
        
        # Greedy Phase
        shuffled_edges = all_edges[:]
        rng.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if time.time() > _DEADLINE - 5:
                break
            if not _has_path_5(u, v, adj_masks):
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges.add((u, v))
        
        # Local Search Phase
        # 1. Try to add non-edges
        non_edges = [(u, v) for u, v in all_edges if (u, v) not in edges]
        rng.shuffle(non_edges)
        
        improved = True
        while improved and time.time() < _DEADLINE - 5:
            improved = False
            for u, v in non_edges:
                if (u, v) in edges:
                    continue
                if not _has_path_5(u, v, adj_masks):
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    edges.add((u, v))
                    improved = True
        
        # 2. Try edge swaps (remove 1, add 1 or 2)
        # We try to replace low-degree edges or random edges
        if time.time() < _DEADLINE - 5:
            # Limit swaps per seed to save time, but allow more than before
            for _ in range(200): 
                if time.time() > _DEADLINE - 5:
                    break
                if not edges:
                    break
                
                # Select an edge to remove
                # Heuristic: Prefer edges that are "redundant" or incident to high-degree nodes?
                # For now, random selection is a baseline.
                edge_list = list(edges)
                ru, rv = rng.choice(edge_list)
                
                # Remove edge
                adj_masks[ru] &= ~(1 << rv)
                adj_masks[rv] &= ~(1 << ru)
                edges.discard((ru, rv))
                
                # Try to add non-edges
                # We want to add at least 1 edge to maintain count, ideally 2 to gain.
                # We scan non-edges.
                added = 0
                # To speed up, we can shuffle non_edges once per swap attempt or use a static list
                # Shuffling 1000+ edges is cheap.
                non_edges_copy = non_edges[:]
                rng.shuffle(non_edges_copy)
                
                for u, v in non_edges_copy:
                    if (u, v) in edges:
                        continue
                    if not _has_path_5(u, v, adj_masks):
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)
                        edges.add((u, v))
                        added += 1
                        if added >= 2: # Net gain +1
                            break
                        # If we only added 1, we break to restart the swap loop?
                        # Or continue trying to add more?
                        # The current code breaks at 2. If added==1, it finishes the loop.
                        # If added < 1, we revert.
                
                if added == 0:
                    # Revert
                    adj_masks[ru] |= (1 << rv)
                    adj_masks[rv] |= (1 << ru)
                    edges.add((ru, rv))
                elif added == 1:
                    # Net zero. We changed the structure. 
                    # This might allow future additions.
                    # We accept this state.
                    pass

        count = len(edges)
        if count > best_count:
            best_count = count
            best_edges = list(edges)
            if best_count >= 106:
                # We found a graph better than known best. 
                # We can continue to see if we can get 107, but likely not.
                # However, the task says maximize.
                pass

    return [[u, v] for u, v in best_edges]
