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

def _is_valid(n, base_edges, added_edges):
    """Check if the graph formed by base + added edges is valid."""
    adj_masks = [0] * n
    base_set = set()
    
    # Add base edges
    for u, v in base_edges:
        u, v = min(u,v), max(u,v)
        if (u,v) in base_set:
            return False # Duplicate in base? Unlikely but check
        base_set.add((u,v))
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    # Add candidate edges
    for u, v in added_edges:
        u, v = min(u,v), max(u,v)
        if (u,v) in base_set:
            return False # Duplicate
        if (adj_masks[u] >> v) & 1:
            return False # Already exists
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        base_set.add((u,v))
        
    # Triangle-free check
    # Since we only add bipartite edges to a path, it should be triangle-free.
    # But let's verify strictly.
    for i in range(n):
        for j in range(i+1, n):
            if (adj_masks[i] >> j) & 1:
                if (adj_masks[i] & adj_masks[j]) != 0:
                    return False
                    
    # Connectivity and Diameter
    all_dist = _compute_all_distances(adj_masks, n)
    for dist in all_dist:
        if -1 in dist:
            return False
        if max(dist) > 4:
            return False
            
    return True

def _greedy_construct(n, base_edges):
    adj_masks = [0] * n
    base_set = set()
    for u, v in base_edges:
        u, v = min(u,v), max(u,v)
        base_set.add((u,v))
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    # Generate all valid bipartite candidates
    candidates = []
    for i in range(n):
        for j in range(i+1, n):
            if (i,j) in base_set:
                continue
            if i % 2 == j % 2:
                continue
            # Check triangle-freeness with current graph
            # If we add (i,j), do they share a neighbor?
            if (adj_masks[i] & adj_masks[j]) == 0:
                candidates.append((i,j))
                
    added = []
    
    while _time_ok():
        all_dist = _compute_all_distances(adj_masks, n)
        diam = _get_diameter(all_dist)
        if diam <= 4:
            break
            
        best_edge = None
        best_gain = 0
        
        for i, j in candidates:
            # Check if still valid to add (triangle-free)
            if (adj_masks[i] & adj_masks[j]) != 0:
                continue
                
            # Calculate gain: number of pairs (u,v) with dist > 4 that become <= 4
            # Or simply sum of distance reductions?
            # Let's count pairs that are "fixed".
            gain = 0
            dist_i = all_dist[i]
            dist_j = all_dist[j]
            
            for u in range(n):
                for v in range(u+1, n):
                    old_d = all_dist[u][v]
                    if old_d > 4:
                        # New distance via i-j edge
                        # dist(u,v) = min(dist(u,v), dist(u,i)+1+dist(j,v), dist(u,j)+1+dist(i,v))
                        new_d = min(old_d, dist_i[u] + 1 + dist_j[v], dist_j[u] + 1 + dist_i[v])
                        if new_d <= 4:
                            gain += 1
                            
            if gain > best_gain:
                best_gain = gain
                best_edge = (i,j)
                
        if best_edge is None or best_gain == 0:
            break
            
        u, v = best_edge
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        base_set.add((u,v))
        added.append([u, v])
        
    return added

def _local_search_minimize(n, base_edges, added_edges):
    """Try to remove edges to minimize count."""
    added = list(added_edges)
    changed = True
    while changed and _time_ok():
        changed = False
        for i in range(len(added) - 1, -1, -1):
            if not _time_ok():
                break
            trial = added[:i] + added[i+1:]
            if _is_valid(n, base_edges, trial):
                added = trial
                changed = True
    return added

def construct():
    n = P['n']
    base = P['edges']
    
    # 1. Greedy construction
    added = _greedy_construct(n, base)
    
    # 2. Local search to minimize
    added = _local_search_minimize(n, base, added)
    
    # 3. Verify
    if _is_valid(n, base, added):
        return [[int(u), int(v)] for u, v in added]
    
    # Fallback: Dense bipartite + prune
    # This is a safety net if greedy fails (unlikely)
    adj_masks = [0] * n
    base_set = set()
    for u, v in base:
        u, v = min(u,v), max(u,v)
        base_set.add((u,v))
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    dense = []
    for i in range(n):
        for j in range(i+1, n):
            if (i,j) in base_set:
                continue
            if i % 2 == j % 2:
                continue
            if (adj_masks[i] & adj_masks[j]) == 0:
                dense.append([i, j])
                
    # Prune dense
    changed = True
    while changed and _time_ok():
        changed = False
        for i in range(len(dense) - 1, -1, -1):
            if not _time_ok():
                break
            trial = dense[:i] + dense[i+1:]
            if _is_valid(n, base, trial):
                dense = trial
                changed = True
                
    if _is_valid(n, base, dense):
        return [[int(u), int(v)] for u, v in dense]
        
    return []

# EVOLVE-BLOCK-END
