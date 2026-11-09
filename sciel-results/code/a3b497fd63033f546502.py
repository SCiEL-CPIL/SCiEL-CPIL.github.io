"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

_N = P['n']
_min_vertices = P['minimum_vertices']

# Precompute adjacency as bitmask integers
_adj_mask = [0] * _N
for (u, v) in P['edges']:
    _adj_mask[u] |= (1 << v)
    _adj_mask[v] |= (1 << u)

def _score_mask(mask):
    """Compute number of distinct degrees in induced subgraph defined by mask."""
    degs = set()
    m = mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        deg = (_adj_mask[v] & mask).bit_count()
        degs.add(deg)
        m ^= lsb
    return len(degs)

def _local_search(mask, deadline, best_score, best_mask):
    """Hill-climbing local search with 1-opt and 2-opt moves."""
    n = _N
    min_v = _min_vertices
    
    cur_score = _score_mask(mask)
    improved = True
    
    while improved:
        if time.time() > deadline:
            break
        improved = False
        
        # Try adding a vertex
        for v in range(n):
            if not (mask & (1 << v)):
                new_mask = mask | (1 << v)
                s = _score_mask(new_mask)
                if s > cur_score:
                    cur_score = s
                    mask = new_mask
                    improved = True
                    break
        
        # Try removing a vertex (if size > min_v)
        if mask.bit_count() > min_v:
            verts = [v for v in range(n) if mask & (1 << v)]
            for v in verts:
                new_mask = mask & ~(1 << v)
                if new_mask.bit_count() >= min_v:
                    s = _score_mask(new_mask)
                    if s > cur_score:
                        cur_score = s
                        mask = new_mask
                        improved = True
                        break
        
        # Try 1-swap: remove one, add another
        if not improved:
            verts = [v for v in range(n) if mask & (1 << v)]
            for v_out in verts:
                if time.time() > deadline:
                    break
                for v_in in range(n):
                    if not (mask & (1 << v_in)):
                        new_mask = (mask & ~(1 << v_out)) | (1 << v_in)
                        if new_mask.bit_count() >= min_v:
                            s = _score_mask(new_mask)
                            if s > cur_score:
                                cur_score = s
                                mask = new_mask
                                improved = True
                                break
                if improved:
                    break
        
        # Try 2-swap: remove two, add two
        if not improved and mask.bit_count() >= min_v + 2:
            verts = [v for v in range(n) if mask & (1 << v)]
            not_verts = [v for v in range(n) if not (mask & (1 << v))]
            for i in range(len(verts)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(verts)):
                    v1, v2 = verts[i], verts[j]
                    base = mask & ~(1 << v1) & ~(1 << v2)
                    for k in range(len(not_verts)):
                        if time.time() > deadline:
                            break
                        for l in range(k + 1, len(not_verts)):
                            v3, v4 = not_verts[k], not_verts[l]
                            new_mask = base | (1 << v3) | (1 << v4)
                            if new_mask.bit_count() >= min_v:
                                s = _score_mask(new_mask)
                                if s > cur_score:
                                    cur_score = s
                                    mask = new_mask
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if cur_score > best_score:
            best_score = cur_score
            best_mask = mask
            
        if cur_score == mask.bit_count():
            break
    
    return best_score, best_mask

def construct():
    start = time.time()
    deadline = start + 80.0  # Safe margin before 90s kill
    best_score = -1
    best_mask = 0
    
    n = _N
    min_v = _min_vertices
    
    # Phase 1: Exhaustive search - try larger subsets first
    for k in range(n, min_v - 1, -1):
        if time.time() > deadline:
            break
        # Theoretical max for k vertices is k distinct degrees
        if best_score >= k:
            continue
        for combo in combinations(range(n), k):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            score = _score_mask(mask)
            if score > best_score:
                best_score = score
                best_mask = mask
                if best_score == k:
                    # Perfect: all k vertices have distinct degrees
                    # No larger subset can do better than n, and we're going down in k
                    # But a larger subset might have n distinct degrees
                    # Actually we're iterating from n down, so if we found k distinct in size k,
                    # we should continue to check if we can get more with larger subsets
                    # But we already checked larger subsets, so this is optimal for this k
                    pass
    
    # Phase 2: Local search polish
    if time.time() < deadline - 5.0 and best_mask != 0:
        best_score, best_mask = _local_search(best_mask, deadline - 5.0, best_score, best_mask)
    
    # Phase 3: Random restarts if time remains
    if time.time() < deadline - 2.0:
        rng = random.Random(42)
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            
            k = rng.randint(min_v, n)
            verts = rng.sample(range(n), k)
            mask = 0
            for v in verts:
                mask |= (1 << v)
            
            best_score, best_mask = _local_search(mask, deadline - 2.0, best_score, best_mask)
            
            if best_score == n:
                break
    
    # Convert best_mask to list of vertex indices
    result = []
    for v in range(n):
        if best_mask & (1 << v):
            result.append(int(v))
            
    if not result:
        result = [int(v) for v in range(min_v)]
        
    return result
# EVOLVE-BLOCK-END
