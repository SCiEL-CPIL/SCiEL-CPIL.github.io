"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_graph():
    n = P['n']
    adj = [[False] * n for _ in range(n)]
    adj_mask = [0] * n  # bitmask of neighbors for each vertex
    for u, v in P['edges']:
        adj[u][v] = True
        adj[v][u] = True
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    return adj, adj_mask

def _is_clique_mask(mask, adj_mask, n):
    """Check if mask forms a clique using bitmask operations."""
    if mask == 0:
        return True
    # For each vertex in mask, all other vertices in mask must be neighbors
    m = mask
    while m:
        v_bit = m & (-m)
        v = v_bit.bit_length() - 1
        # All other vertices in mask must be adjacent to v
        others = mask ^ v_bit
        if others & ~adj_mask[v] != 0:
            return False
        m ^= v_bit
    return True

def _is_independent_mask(mask, adj_mask, n):
    """Check if mask forms an independent set using bitmask operations."""
    if mask == 0:
        return True
    # For each vertex in mask, no other vertex in mask should be a neighbor
    m = mask
    while m:
        v_bit = m & (-m)
        v = v_bit.bit_length() - 1
        others = mask ^ v_bit
        if others & adj_mask[v] != 0:
            return False
        m ^= v_bit
    return True

def _precompute_valid_parts(adj_mask, n):
    """Precompute for each mask whether it's a valid part (clique or independent set)."""
    valid = [False] * (1 << n)
    valid[0] = False
    for mask in range(1, 1 << n):
        if _is_clique_mask(mask, adj_mask, n) or _is_independent_mask(mask, adj_mask, n):
            valid[mask] = True
    return valid

def _precompute_max_clique_indep(adj_mask, n):
    """For each mask, precompute the max clique and max independent set sizes within it."""
    max_clique = [0] * (1 << n)
    max_indep = [0] * (1 << n)
    
    for mask in range(1, 1 << n):
        # Max clique in mask
        mc = 0
        sub = mask
        while sub:
            if _is_clique_mask(sub, adj_mask, n):
                pc = sub.bit_count()
                if pc > mc:
                    mc = pc
            sub = (sub - 1) & mask
        max_clique[mask] = mc
        
        # Max independent set in mask
        mi = 0
        sub = mask
        while sub:
            if _is_independent_mask(sub, adj_mask, n):
                pc = sub.bit_count()
                if pc > mi:
                    mi = pc
            sub = (sub - 1) & mask
        max_indep[mask] = mi
    
    return max_clique, max_indep

def _greedy_upper_bound(adj_mask, n):
    """Get a greedy upper bound on the number of parts."""
    remaining = (1 << n) - 1
    parts = 0
    while remaining:
        # Find largest valid part containing the lowest vertex
        v_bit = remaining & (-remaining)
        v = v_bit.bit_length() - 1
        best_part = v_bit
        # Try to extend the part
        for mask in range(1, 1 << n):
            if not (mask & v_bit):
                continue
            if mask & remaining != mask:
                continue
            if _is_clique_mask(mask, adj_mask, n) or _is_independent_mask(mask, adj_mask, n):
                if mask.bit_count() > best_part.bit_count():
                    best_part = mask
        remaining ^= best_part
        parts += 1
    return parts

def _backtrack_min_parts(adj_mask, n, deadline):
    """Find minimum number of parts to partition all vertices."""
    full_mask = (1 << n) - 1
    valid = _precompute_valid_parts(adj_mask, n)
    
    # Precompute: for each vertex, list of valid masks containing that vertex
    masks_by_vertex = [[] for _ in range(n)]
    for mask in range(1, 1 << n):
        if valid[mask]:
            for v in range(n):
                if mask & (1 << v):
                    masks_by_vertex[v].append(mask)
    
    # Sort: prefer larger masks first
    for v in range(n):
        masks_by_vertex[v].sort(key=lambda m: m.bit_count(), reverse=True)
    
    # Get initial upper bound
    ub = _greedy_upper_bound(adj_mask, n)
    best_k = [ub]
    best_partition = [None]
    nodes = [0]
    
    def dfs(remaining, k_used, path, target_k):
        nodes[0] += 1
        if time.time() > deadline or nodes[0] > 5000000:
            return False
        if remaining == 0:
            if k_used <= target_k:
                best_k[0] = k_used
                best_partition[0] = list(path)
            return True
        if k_used >= target_k:
            return False
        # Lower bound: at least ceil(popcount(remaining) / max_part_size) more parts needed
        # Simple lower bound: 1 if remaining non-empty
        # Pick vertex with fewest options (most constrained)
        best_v = -1
        best_count = float('inf')
        m = remaining
        while m:
            v_bit = m & (-m)
            v = v_bit.bit_length() - 1
            # Count valid parts containing v that are subsets of remaining
            cnt = 0
            for pm in masks_by_vertex[v]:
                if pm & remaining == pm:
                    cnt += 1
                    if cnt >= best_count:
                        break
            if cnt < best_count:
                best_count = cnt
                best_v = v
                if cnt == 1:
                    break
            m ^= v_bit
        
        v = best_v
        for part_mask in masks_by_vertex[v]:
            if part_mask & remaining != part_mask:
                continue
            path.append(part_mask)
            if dfs(remaining ^ part_mask, k_used + 1, path, target_k):
                path.pop()
                return True
            path.pop()
        return False
    
    # Try decreasing target_k from ub down to 1
    for target_k in range(ub, 0, -1):
        nodes[0] = 0
        best_partition[0] = None
        if dfs(full_mask, 0, [], target_k):
            if best_partition[0] is not None:
                parts = []
                for m in best_partition[0]:
                    parts.append([i for i in range(n) if m & (1 << i)])
                return parts
    return None

def construct():
    adj, adj_mask = _build_graph()
    n = P['n']
    
    deadline = time.time() + 50  # safety margin
    
    result = _backtrack_min_parts(adj_mask, n, deadline)
    if result is not None:
        # Ensure integers and lists
        parts = []
        for part in result:
            parts.append([int(v) for v in part])
        return parts
    
    # Fallback: greedy
    es = {tuple(e) for e in P['edges']}
    parts = []
    for v in range(n):
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i + 1:]]
            if all(hits) or not any(hits):
                part.append(v)
                break
        else:
            parts.append([v])
    return parts

# EVOLVE-BLOCK-END
