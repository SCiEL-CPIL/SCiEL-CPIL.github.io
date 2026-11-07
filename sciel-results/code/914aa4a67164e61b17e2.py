"""Program: Maximum diversity selection via multi-seed greedy heuristic.

Approach:
  * Model the problem as selecting a subset S of vertices to maximize the number
    of distinct degrees in the induced subgraph G[S].
  * Use a greedy heuristic that iteratively adds the vertex maximizing the
    number of new distinct degree values.
  * Initialize with multiple random seeds to escape local optima.
  * Enhance the greedy result with local search (swap moves) to further
    improve the diversity score.
  * This avoids the overhead of exhaustive search while leveraging the
    approximate optimization loop.
"""
import random

# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

_N = P['n']
_MINV = P['minimum_vertices']

# Precompute the adjacency of every vertex as a 17-bit mask.
_ADJ = [0] * _N
for _a, _b in P['edges']:
    _ADJ[_a] |= (1 << _b)
    _ADJ[_b] |= (1 << _a)


def _distinct_deg(mask):
    """Number of distinct induced-subgraph degrees among the vertices in `mask`."""
    degs = set()
    v = 0
    m = mask
    while m:
        if m & 1:
            degs.add((_ADJ[v] & mask).bit_count())
        m >>= 1
        v += 1
    return len(degs)


def _greedy_add(mask, cur_d):
    """Iteratively add the vertex that maximizes new distinct degrees."""
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_v = -1
        for v in range(_N):
            if mask & (1 << v):
                continue
            new_mask = mask | (1 << v)
            d = _distinct_deg(new_mask)
            gain = d - cur_d
            if gain > best_gain:
                best_gain = gain
                best_v = v
        if best_v >= 0 and best_gain > 0:
            mask |= (1 << best_v)
            cur_d += best_gain
            improved = True
    return mask, cur_d


def _local_search(mask, cur_d):
    """Try swapping vertices to improve distinct degrees."""
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_swap = None
        # Try all pairs (u in mask, v not in mask)
        for u in range(_N):
            if not (mask & (1 << u)):
                continue
            for v in range(_N):
                if mask & (1 << v):
                    continue
                new_mask = (mask & ~(1 << u)) | (1 << v)
                d = _distinct_deg(new_mask)
                gain = d - cur_d
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (u, v)
        if best_swap:
            u, v = best_swap
            mask = (mask & ~(1 << u)) | (1 << v)
            cur_d += best_gain
            improved = True
    return mask, cur_d


def _run_heuristic(rng):
    """Run one instance of the greedy + local search heuristic."""
    # Initialize with random set of size _MINV
    verts = list(range(_N))
    rng.shuffle(verts)
    initial_mask = 0
    for v in verts[:_MINV]:
        initial_mask |= (1 << v)
    
    cur_d = _distinct_deg(initial_mask)
    
    # Phase 1: Greedy addition
    mask, cur_d = _greedy_add(initial_mask, cur_d)
    
    # Phase 2: Local search (swap optimization)
    mask, cur_d = _local_search(mask, cur_d)
    
    return mask, cur_d


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
    # Deterministic random seeds for reproducibility
    rng = random.Random(20240517)
    
    best_mask = 0
    best_d = -1
    
    # Run multiple instances of the heuristic
    # 1000 iterations is fast for N=17 and likely finds the global optimum
    for _ in range(1000):
        m, d = _run_heuristic(rng)
        if d > best_d:
            best_d = d
            best_mask = m
        elif d == best_d and m.bit_count() < best_mask.bit_count():
            # Prefer smaller sets if scores are equal? Or larger? 
            # Task says "at least 8". Smaller set with same score is "tighter" but score is same.
            # Usually we want max score. If tie, any is fine. 
            # Let's stick to first found or max score.
            pass

    verts = _mask_to_vertices(best_mask)
    
    # Defensive: guarantee at least _MINV distinct Python-int vertices
    if len(verts) < _MINV:
        extra = [v for v in range(_N) if v not in set(verts)]
        verts = verts + extra[: _MINV - len(verts)]
        
    verts = [int(v) for v in sorted(verts)]
    return verts
# EVOLVE-BLOCK-END
