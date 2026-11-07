"""Combined: bitmask exhaustive search + branch-and-bound CSP-style pruning + local search polish."""
from itertools import combinations
import time

# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adjacency_bitmasks():
    n = P['n']
    adj_mask = [0] * n
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    return adj_mask

def _popcount(x):
    """Fast popcount using Brian Kernighan's algorithm."""
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c

def _distinct_degree_count_mask(mask, adj_mask):
    """Count distinct degrees in induced subgraph represented by bitmask."""
    degrees = set()
    for v in range(P['n']):
        if mask & (1 << v):
            deg = _popcount(mask & adj_mask[v])
            degrees.add(deg)
    return len(degrees)

def _local_search(mask, adj_mask, deadline):
    """Local search: try swapping vertices to improve distinct degree count."""
    n = P['n']
    min_v = P['minimum_vertices']
    best_mask = mask
    best_score = _distinct_degree_count_mask(mask, adj_mask)
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        # Try all single swaps
        for out_v in range(n):
            if not (mask & (1 << out_v)):
                continue
            for in_v in range(n):
                if mask & (1 << in_v):
                    continue
                if time.time() > deadline:
                    return best_mask, best_score
                new_mask = (mask | (1 << in_v)) & ~(1 << out_v)
                # Check size constraint
                if _popcount(new_mask) < min_v:
                    continue
                score = _distinct_degree_count_mask(new_mask, adj_mask)
                if score > best_score:
                    best_score = score
                    best_mask = new_mask
                    improved = True
                    mask = new_mask
                    break
            if improved:
                break
    return best_mask, best_score

def _two_swap_search(mask, adj_mask, deadline):
    """Try two-vertex swaps for more powerful local search."""
    n = P['n']
    min_v = P['minimum_vertices']
    best_mask = mask
    best_score = _distinct_degree_count_mask(mask, adj_mask)
    
    in_vertices = [v for v in range(n) if not (mask & (1 << v))]
    out_vertices = [v for v in range(n) if (mask & (1 << v))]
    
    if len(in_vertices) < 2 or len(out_vertices) < 2:
        return best_mask, best_score
    
    for i in range(len(out_vertices)):
        for j in range(i + 1, len(out_vertices)):
            if time.time() > deadline:
                break
            v1 = out_vertices[i]
            v2 = out_vertices[j]
            base_mask = mask & ~(1 << v1) & ~(1 << v2)
            for a in range(len(in_vertices)):
                for b in range(a + 1, len(in_vertices)):
                    if time.time() > deadline:
                        break
                    w1 = in_vertices[a]
                    w2 = in_vertices[b]
                    new_mask = base_mask | (1 << w1) | (1 << w2)
                    score = _distinct_degree_count_mask(new_mask, adj_mask)
                    if score > best_score:
                        best_score = score
                        best_mask = new_mask
    return best_mask, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin before 90s kill
    n = P['n']
    min_v = P['minimum_vertices']
    adj_mask = _build_adjacency_bitmasks()
    
    all_vertices = list(range(n))
    best_score = 0
    best_mask = 0
    
    # Phase 1: Exhaustive search over subset sizes from large to small
    # With branch-and-bound style pruning: if current best >= size, skip
    for size in range(n, min_v - 1, -1):
        if time.time() > deadline:
            break
        # If we already have as many distinct degrees as this size allows, stop
        if best_score >= size:
            break
        
        for subset in combinations(all_vertices, size):
            if time.time() > deadline:
                break
            mask = 0
            for v in subset:
                mask |= (1 << v)
            score = _distinct_degree_count_mask(mask, adj_mask)
            if score > best_score:
                best_score = score
                best_mask = mask
                # Theoretical max for this size is size (all distinct)
                if best_score == size:
                    break
        if time.time() > deadline:
            break
    
    # Phase 2: Local search polish on the best solution found
    if best_mask > 0 and time.time() < deadline:
        polished_mask, polished_score = _local_search(best_mask, adj_mask, deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_mask = polished_mask
        
        # Phase 2b: Two-swap local search for more powerful moves
        if time.time() < deadline - 5:
            two_mask, two_score = _two_swap_search(best_mask, adj_mask, deadline)
            if two_score > best_score:
                best_score = two_score
                best_mask = two_mask
            # Follow up with single-swap search after two-swap improvement
            if time.time() < deadline - 2:
                polished_mask2, polished_score2 = _local_search(best_mask, adj_mask, deadline)
                if polished_score2 > best_score:
                    best_score = polished_score2
                    best_mask = polished_mask2
    
    # Phase 3: Random restarts with local search if we haven't found high score
    if best_score < 10 and time.time() < deadline - 5:
        import random
        random.seed(42)
        for _ in range(300):
            if time.time() > deadline - 3:
                break
            # Random subset of size between min_v and n
            sz = random.randint(min_v, n)
            subset = random.sample(all_vertices, sz)
            mask = 0
            for v in subset:
                mask |= (1 << v)
            score = _distinct_degree_count_mask(mask, adj_mask)
            if score > best_score:
                best_score = score
                best_mask = mask
            # Local search from this starting point
            if time.time() < deadline - 3:
                pm, ps = _local_search(mask, adj_mask, deadline)
                if ps > best_score:
                    best_score = ps
                    best_mask = pm
    
    # Final polish
    if best_mask > 0 and time.time() < deadline - 1:
        final_mask, final_score = _local_search(best_mask, adj_mask, deadline)
        if final_score > best_score:
            best_score = final_score
            best_mask = final_mask
    
    # Convert best_mask to list of vertex indices
    result = [int(v) for v in range(n) if best_mask & (1 << v)]
    return result
# EVOLVE-BLOCK-END
