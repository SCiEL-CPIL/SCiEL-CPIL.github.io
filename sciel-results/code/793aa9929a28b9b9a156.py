"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

_START_TIME = time.time()
_TIME_LIMIT = 85.0

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

def _compute_all_distances(adj_masks, n):
    all_dist = []
    for s in range(n):
        all_dist.append(_bfs_distances(adj_masks, n, s))
    return all_dist

def _get_diameter(all_dist):
    max_d = 0
    for dist in all_dist:
        for d in dist:
            if d > max_d:
                max_d = d
    return max_d

def _check_valid(n, base_edges, added_edges):
    """Check: no duplicates, triangle-free, connected, diameter <= 4."""
    adj_masks = [0] * n
    base_set = set()
    for u, v in base_edges:
        u, v = min(u,v), max(u,v)
        base_set.add((u,v))
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
    
    for u, v in added_edges:
        u, v = min(u,v), max(u,v)
        if (u,v) in base_set:
            return False
        if (adj_masks[u] >> v) & 1:
            return False
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        base_set.add((u,v))
    
    # Triangle-free check
    for i in range(n):
        for j in range(i+1, n):
            if (adj_masks[i] >> j) & 1:
                if (adj_masks[i] & adj_masks[j]) != 0:
                    return False
    
    # Connected + diameter
    all_dist = _compute_all_distances(adj_masks, n)
    for dist in all_dist:
        if -1 in dist:
            return False
        if max(dist) > 4:
            return False
    return True

def _build_adj_masks(n, base_edges, added_edges):
    adj_masks = [0] * n
    for u, v in base_edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
    for u, v in added_edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
    return adj_masks

def _try_pattern(n, base, pattern_edges):
    """Try a specific set of pattern edges, then prune."""
    if not _time_ok():
        return None
    
    # Verify pattern edges are valid (not in base, triangle-free)
    adj_masks = _build_adj_masks(n, base, [])
    base_set = set()
    for u, v in base:
        base_set.add((min(u,v), max(u,v)))
    
    valid_pattern = []
    for u, v in pattern_edges:
        u, v = min(u,v), max(u,v)
        if (u,v) in base_set:
            continue
        if (adj_masks[u] >> v) & 1:
            continue
        if (adj_masks[u] & adj_masks[v]) != 0:
            continue
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        base_set.add((u,v))
        valid_pattern.append([u, v])
    
    if not valid_pattern:
        return None
    
    if not _check_valid(n, base, valid_pattern):
        return None
    
    # Prune: try removing edges one by one
    added = valid_pattern
    changed = True
    while changed and _time_ok():
        changed = False
        for i in range(len(added) - 1, -1, -1):
            if not _time_ok():
                break
            trial = added[:i] + added[i+1:]
            if _check_valid(n, base, trial):
                added = trial
                changed = True
    
    if _check_valid(n, base, added):
        return added
    return None

def _greedy_construct(n, base):
    """Greedy: add edges that most reduce the diameter."""
    adj_masks = _build_adj_masks(n, base, [])
    base_set = set()
    for u, v in base:
        base_set.add((min(u,v), max(u,v)))
    
    # Precompute all valid bipartite candidate edges
    candidates = []
    for i in range(n):
        for j in range(i+1, n):
            if (i,j) in base_set:
                continue
            if i % 2 == j % 2:
                continue
            if (adj_masks[i] & adj_masks[j]) == 0:
                candidates.append((i,j))
    
    added = []
    all_dist = _compute_all_distances(adj_masks, n)
    
    while _time_ok():
        diam = _get_diameter(all_dist)
        if diam <= 4:
            break
        
        best_edge = None
        best_gain = 0
        
        for i, j in candidates:
            if (i,j) in base_set:
                continue
            if (adj_masks[i] & adj_masks[j]) != 0:
                continue
            
            gain = 0
            dist_i = all_dist[i]
            dist_j = all_dist[j]
            for k in range(n):
                old_d = max(dist_i[k], dist_j[k])
                new_d = min(dist_i[k], 1 + dist_j[k], dist_j[k], 1 + dist_i[k])
                if old_d > new_d:
                    gain += old_d - new_d
            
            if gain > best_gain:
                best_gain = gain
                best_edge = (i,j)
        
        if best_edge is None or best_gain == 0:
            break
        
        u, v = best_edge
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        base_set.add((min(u,v), max(u,v)))
        added.append([u, v])
        all_dist = _compute_all_distances(adj_masks, n)
    
    return added

def _local_search(n, base, added):
    """Try to remove edges, and try swaps to reduce count."""
    if not added:
        return added
    
    changed = True
    while changed and _time_ok():
        changed = False
        for i in range(len(added) - 1, -1, -1):
            if not _time_ok():
                break
            trial = added[:i] + added[i+1:]
            if _check_valid(n, base, trial):
                added = trial
                changed = True
    
    improved = True
    while improved and _time_ok():
        improved = False
        adj_masks = _build_adj_masks(n, base, added)
        base_set = set()
        for u, v in base:
            base_set.add((min(u,v), max(u,v)))
        for u, v in added:
            base_set.add((min(u,v), max(u,v)))
        
        candidates = []
        for i in range(n):
            for j in range(i+1, n):
                if (i,j) in base_set:
                    continue
                if i % 2 == j % 2:
                    continue
                if (adj_masks[i] & adj_masks[j]) == 0:
                    candidates.append((i,j))
        
        for i in range(len(added)):
            if not _time_ok():
                break
            removed = added[i]
            trial_without = added[:i] + added[i+1:]
            if _check_valid(n, base, trial_without):
                added = trial_without
                improved = True
                break
        
        if improved:
            continue
        
        for i in range(len(added)):
            if not _time_ok():
                break
            removed = added[i]
            for cand in candidates:
                if not _time_ok():
                    break
                if cand == removed or cand == (removed[1], removed[0]):
                    continue
                if (min(cand[0],cand[1]), max(cand[0],cand[1])) in base_set:
                    continue
                trial = added[:i] + added[i+1:] + [list(cand)]
                if _check_valid(n, base, trial):
                    added = trial
                    improved = True
                    break
            if improved:
                break
    
    return added

def _gen_patterns(n, base):
    """Generate multiple structured patterns based on path symmetry."""
    base_set = set()
    for u, v in base:
        base_set.add((min(u,v), max(u,v)))
    
    patterns = []
    
    # Pattern family: skip edges (i, i+k) for various k
    for k in [5, 6, 7, 8, 9, 10, 11, 12]:
        pat = []
        for i in range(n):
            j = i + k
            if j < n:
                key = (min(i,j), max(i,j))
                if key not in base_set:
                    if i % 2 != j % 2:  # bipartite check
                        pat.append([i, j])
        if pat:
            patterns.append(pat)
    
    # Pattern family: symmetric shortcuts from center
    center = n // 2
    for span in [6, 7, 8, 9, 10, 11, 12]:
        pat = []
        for i in range(n):
            j = i + span
            if j < n:
                key = (min(i,j), max(i,j))
                if key not in base_set and i % 2 != j % 2:
                    pat.append([i, j])
        if pat:
            patterns.append(pat)
    
    # Pattern family: hub-and-spoke style
    for hub in [6, 8, 10, 12, 14, 16, 18]:
        pat = []
        for i in range(n):
            if i == hub:
                continue
            dist = abs(i - hub)
            if dist >= 4 and dist <= 12:
                key = (min(i, hub), max(i, hub))
                if key not in base_set and i % 2 != hub % 2:
                    pat.append([i, hub])
        if pat:
            patterns.append(pat)
    
    # Pattern family: two hubs
    for h1 in [6, 8, 10]:
        for h2 in [14, 16, 18]:
            if abs(h1 - h2) < 8:
                continue
            pat = []
            for i in range(n):
                if i == h1 or i == h2:
                    continue
                for hub in [h1, h2]:
                    dist = abs(i - hub)
                    if dist >= 4 and dist <= 10:
                        key = (min(i, hub), max(i, hub))
                        if key not in base_set and i % 2 != hub % 2:
                            if [i, hub] not in pat and [hub, i] not in pat:
                                pat.append([i, hub])
                                break
            if pat:
                patterns.append(pat)
    
    # Pattern family: layered shortcuts
    for layer1 in [5, 6, 7]:
        for layer2 in [10, 11, 12]:
            pat = []
            for i in range(n):
                j1 = i + layer1
                j2 = i + layer2
                if j1 < n:
                    key1 = (min(i,j1), max(i,j1))
                    if key1 not in base_set and i % 2 != j1 % 2:
                        pat.append([i, j1])
                if j2 < n:
                    key2 = (min(i,j2), max(i,j2))
                    if key2 not in base_set and i % 2 != j2 % 2:
                        pat.append([i, j2])
            if pat:
                patterns.append(pat)
    
    return patterns

def construct():
    n = P['n']
    base = P['edges']
    
    best_added = None
    
    # Try structured patterns first (fast, deterministic)
    patterns = _gen_patterns(n, base)
    
    for pat in patterns:
        if not _time_ok():
            break
        result = _try_pattern(n, base, pat)
        if result is not None:
            if best_added is None or len(result) < len(best_added):
                best_added = result
    
    # Greedy construction as backup
    greedy_result = _greedy_construct(n, base)
    if greedy_result:
        if best_added is None or len(greedy_result) < len(best_added):
            best_added = greedy_result
    
    # Local search refinement
    if best_added is not None:
        best_added = _local_search(n, base, best_added)
    
    # Final verification
    if best_added is not None and _check_valid(n, base, best_added):
        return [[int(u), int(v)] for u, v in best_added]
    
    # Fallback: dense bipartite + aggressive prune
    adj_masks = [0] * n
    for u, v in base:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
    base_set = set()
    for u, v in base:
        base_set.add((min(u,v), max(u,v)))
    
    dense = []
    for i in range(n):
        for j in range(i+1, n):
            if (i,j) in base_set:
                continue
            if i % 2 == j % 2:
                continue
            dense.append([i, j])
    
    changed = True
    while changed and _time_ok():
        changed = False
        for i in range(len(dense) - 1, -1, -1):
            if not _time_ok():
                break
            trial = dense[:i] + dense[i+1:]
            if _check_valid(n, base, trial):
                dense = trial
                changed = True
    
    if _check_valid(n, base, dense):
        return [[int(u), int(v)] for u, v in dense]
    
    return []

# EVOLVE-BLOCK-END
