"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

_START_TIME = time.time()
_TIME_LIMIT = 85.0  # safety margin before 90s kill

def _time_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_distances(adj_masks, n, start):
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        m = adj_masks[u]
        while m:
            lb = m & -m
            v = lb.bit_length() - 1
            m ^= lb
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _check_solution_fast(n, base_edges, added_edges):
    adj_masks = [0] * n
    for u, v in base_edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
    for u, v in added_edges:
        if (adj_masks[u] >> v) & 1:
            return False
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
    for i in range(n):
        for j in range(i + 1, n):
            if (adj_masks[i] >> j) & 1:
                if (adj_masks[i] & adj_masks[j]) != 0:
                    return False
    for start in range(n):
        dist = _bfs_distances(adj_masks, n, start)
        if -1 in dist:
            return False
        if max(dist) > 4:
            return False
    return True

def _is_triangle_free_add_mask(adj_masks, u, v):
    return (adj_masks[u] & adj_masks[j]) == 0 if False else (adj_masks[u] & adj_masks[v]) == 0

def _build_adj_masks(n, base_edges, added_edges):
    adj_masks = [0] * n
    for u, v in base_edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
    for u, v in added_edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
    return adj_masks

def _diameter_of(adj_masks, n):
    max_d = 0
    for start in range(n):
        dist = _bfs_distances(adj_masks, n, start)
        if -1 in dist:
            return 999
        d = max(dist)
        if d > max_d:
            max_d = d
    return max_d

def _try_pattern(n, base_edges, pattern_edges):
    """Check if a specific set of added edges is valid. Return True if valid."""
    return _check_solution_fast(n, base_edges, pattern_edges)

def _prune_edges(n, base, added):
    """Aggressively remove any edge whose removal keeps the solution valid."""
    for _ in range(5):
        if not _time_ok():
            break
        changed = True
        while changed and _time_ok():
            changed = False
            for i in range(len(added) - 1, -1, -1):
                if not _time_ok():
                    break
                new_added = added[:i] + added[i + 1:]
                if _check_solution_fast(n, base, new_added):
                    added = new_added
                    changed = True
    return added

def _greedy_construct(n, base_edges):
    """Greedy: add edges that most reduce diameter, maintaining triangle-free."""
    adj_masks = [0] * n
    for u, v in base_edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
    
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
    
    added = []
    added_set = set()
    
    # Precompute candidate edges sorted by distance (shorter first for greedy)
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in base_set:
                candidates.append((j - i, i, j))
    candidates.sort()
    
    while _time_ok():
        diam = _diameter_of(adj_masks, n)
        if diam <= 4:
            break
        
        best_edge = None
        best_diam = diam + 1
        
        for dist, i, j in candidates:
            if not _time_ok():
                break
            if (i, j) in added_set:
                continue
            # Triangle-free check
            if (adj_masks[i] & adj_masks[j]) != 0:
                continue
            # Temporarily add
            adj_masks[i] |= (1 << j)
            adj_masks[j] |= (1 << i)
            d = _diameter_of(adj_masks, n)
            adj_masks[i] ^= (1 << j)
            adj_masks[j] ^= (1 << i)
            
            if d < best_diam:
                best_diam = d
                best_edge = (i, j)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        added.append([u, v])
        added_set.add((u, v))
    
    return added

def construct():
    n = P['n']
    base = P['edges']
    base_set = set()
    for u, v in base:
        base_set.add((min(u, v), max(u, v)))
    
    # Strategy 1: Try specific structured patterns (fast, deterministic)
    # Pattern A: (i, i+6) for i=0..17
    pattern_a = [[i, i + 6] for i in range(0, 18) if i + 6 < n and (i, i + 6) not in base_set]
    if _try_pattern(n, base, pattern_a):
        pruned = _prune_edges(n, base, pattern_a)
        return [[int(u), int(v)] for u, v in pruned]
    
    # Pattern B: (i, i+8) for i=0..15
    pattern_b = [[i, i + 8] for i in range(0, 16) if i + 8 < n and (i, i + 8) not in base_set]
    if _try_pattern(n, base, pattern_b):
        pruned = _prune_edges(n, base, pattern_b)
        return [[int(u), int(v)] for u, v in pruned]
    
    # Pattern C: (i, i+5) for i=0..18, then (i, i+10) for i=0..13
    pattern_c = []
    for i in range(0, 19):
        if i + 5 < n and (i, i + 5) not in base_set:
            pattern_c.append([i, i + 5])
    for i in range(0, 14):
        if i + 10 < n and (i, i + 10) not in base_set:
            pattern_c.append([i, i + 10])
    if _try_pattern(n, base, pattern_c):
        pruned = _prune_edges(n, base, pattern_c)
        return [[int(u), int(v)] for u, v in pruned]
    
    # Pattern D: (i, i+7) for i=0..16
    pattern_d = [[i, i + 7] for i in range(0, 17) if i + 7 < n and (i, i + 7) not in base_set]
    if _try_pattern(n, base, pattern_d):
        pruned = _prune_edges(n, base, pattern_d)
        return [[int(u), int(v)] for u, v in pruned]
    
    # Pattern E: (i, i+9) for i=0..14
    pattern_e = [[i, i + 9] for i in range(0, 15) if i + 9 < n and (i, i + 9) not in base_set]
    if _try_pattern(n, base, pattern_e):
        pruned = _prune_edges(n, base, pattern_e)
        return [[int(u), int(v)] for u, v in pruned]
    
    # Strategy 2: Greedy construction + prune
    if _time_ok():
        added = _greedy_construct(n, base)
        if _check_solution_fast(n, base, added):
            added = _prune_edges(n, base, added)
            return [[int(u), int(v)] for u, v in added]
    
    # Strategy 3: Add all triangle-free edges, then prune aggressively
    if _time_ok():
        adj_masks = [0] * n
        for u, v in base:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        added2 = []
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in base_set:
                    if _is_triangle_free_add_mask(adj_masks, i, j):
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
                        added2.append([i, j])
        added2 = _prune_edges(n, base, added2)
        if _check_solution_fast(n, base, added2):
            return [[int(u), int(v)] for u, v in added2]
    
    return []

# EVOLVE-BLOCK-END
