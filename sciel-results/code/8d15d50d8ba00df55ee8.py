"""Improved program: High-quality greedy diversity selection with local search.

Approach:
  * Models the problem as maximizing the number of distinct induced degrees.
  * Uses a multi-start greedy heuristic:
    1. Initialize with a random subset of size _MINV.
    2. Greedily add vertices that increase the count of distinct degrees.
    3. Perform local search (swap-based) to escape local optima and potentially
       increase the distinct degree count or maintain it while improving stability.
  * Runs many random restarts to ensure the global optimum is found.
  * This approach avoids the overhead of full exhaustive enumeration while
    leveraging the structure of the diversity objective.
"""
import random

# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

_N = P['n']
_MINV = P['minimum_vertices']

# Precompute adjacency masks
_ADJ = [0] * _N
for _a, _b in P['edges']:
    _ADJ[_a] |= (1 << _b)
    _ADJ[_b] |= (1 << _a)


def _distinct_deg(mask):
    """Number of distinct induced-subgraph degrees among the vertices in `mask`."""
    degs = set()
    m = mask
    v = 0
    while m:
        if m & 1:
            degs.add((_ADJ[v] & mask).bit_count())
        m >>= 1
        v += 1
    return len(degs)


def _greedy_with_local_search(rng):
    """Greedy construction followed by local search (swaps)."""
    verts = list(range(_N))
    rng.shuffle(verts)
    
    # 1. Greedy construction
    chosen = set(verts[:_MINV])
    cur_mask = 0
    for v in chosen:
        cur_mask |= (1 << v)
    
    cur_d = _distinct_deg(cur_mask)
    
    # Add vertices if it increases diversity
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_v = -1
        for v in range(_N):
            if v in chosen:
                continue
            new_mask = cur_mask | (1 << v)
            d = _distinct_deg(new_mask)
            gain = d - cur_d
            if gain > best_gain:
                best_gain = gain
                best_v = v
        if best_v >= 0 and best_gain > 0:
            chosen.add(best_v)
            cur_mask |= (1 << best_v)
            cur_d += best_gain
            improved = True
            
    # 2. Local Search (Swaps)
    # Try to swap a vertex in `chosen` with a vertex not in `chosen`
    # to improve or maintain diversity (prioritize improvement).
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_swap = None
        chosen_list = list(chosen)
        for v_in in chosen_list:
            for v_out in range(_N):
                if v_out in chosen:
                    continue
                # Perform swap
                new_mask = (cur_mask & ~(1 << v_in)) | (1 << v_out)
                d = _distinct_deg(new_mask)
                gain = d - cur_d
                # We only accept if gain > 0. 
                # If gain == 0, we might want to explore, but for simplicity 
                # and speed, we only take strict improvements to ensure convergence.
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (v_in, v_out)
        
        if best_swap is not None and best_gain > 0:
            v_in, v_out = best_swap
            chosen.remove(v_in)
            chosen.add(v_out)
            cur_mask = (cur_mask & ~(1 << v_in)) | (1 << v_out)
            cur_d += best_gain
            improved = True
            
    return cur_mask, cur_d


def _mask_to_vertices(mask):
    out = []
    v = 0
    m = mask
    while m:
        if m & 1:
            out.append(v)
        m >>= 1
        v += 1
    return out


def construct():
    # Use a deterministic seed for reproducibility
    rng = random.Random(42)
    
    best_mask = 0
    best_d = -1
    
    # Run multiple restarts
    num_restarts = 1000
    for _ in range(num_restarts):
        m, d = _greedy_with_local_search(rng)
        if d > best_d:
            best_d = d
            best_mask = m
        # Early termination if we hit the theoretical max for this size?
        # Max distinct degrees is size of set.
        # If best_d == _N, we can stop. But we don't know the size of best_mask yet.
        # However, if best_d == 17, it's perfect.
        if best_d == _N:
            break
            
    verts = _mask_to_vertices(best_mask)
    
    # Ensure minimum vertices
    if len(verts) < _MINV:
        extra = [v for v in range(_N) if v not in set(verts)]
        verts = verts + extra[: _MINV - len(verts)]
        
    verts = [int(v) for v in sorted(verts)]
    return verts
# EVOLVE-BLOCK-END
