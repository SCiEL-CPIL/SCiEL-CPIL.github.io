"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_graph():
    n = P['n']
    adj = [[False] * n for _ in range(n)]
    for u, v in P['edges']:
        adj[u][v] = True
        adj[v][u] = True
    return adj

def _precompute_valid_parts(adj):
    """Precompute for each mask whether it's a valid part (clique or independent set)."""
    n = len(adj)
    valid = [False] * (1 << n)
    valid[0] = False  # empty set is not a valid part
    for mask in range(1, 1 << n):
        # Get vertices in mask
        verts = [i for i in range(n) if mask & (1 << i)]
        # Check clique
        is_clique = True
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                if not adj[verts[i]][verts[j]]:
                    is_clique = False
                    break
            if not is_clique:
                break
        if is_clique:
            valid[mask] = True
            continue
        # Check independent set
        is_indep = True
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                if adj[verts[i]][verts[j]]:
                    is_indep = False
                    break
            if not is_indep:
                break
        if is_indep:
            valid[mask] = True
    return valid

def _backtrack_min_parts(adj, valid, n, deadline):
    """Find minimum number of parts to partition all vertices."""
    full_mask = (1 << n) - 1
    nodes = [0]
    
    # Precompute: for each vertex, list of valid masks containing that vertex
    # Sort by popcount (larger first) to find good partitions faster
    masks_by_vertex = [[] for _ in range(n)]
    for mask in range(1, 1 << n):
        if valid[mask]:
            for v in range(n):
                if mask & (1 << v):
                    masks_by_vertex[v].append(mask)
    # Sort each list: larger masks first (fewer parts is more likely)
    for v in range(n):
        masks_by_vertex[v].sort(key=lambda m: m.bit_count(), reverse=True)
    
    best_k = [n]  # upper bound
    best_partition = [None]
    
    def dfs(remaining, k_used, path):
        nodes[0] += 1
        if time.time() > deadline or nodes[0] > 1000000:
            return False
        if remaining == 0:
            if k_used < best_k[0]:
                best_k[0] = k_used
                best_partition[0] = list(path)
            return True
        # Lower bound: need at least 1 more part
        if k_used >= best_k[0]:
            return False
        # Pick smallest vertex in remaining
        v = (remaining & (-remaining)).bit_length() - 1
        # Try all valid parts containing v that are subset of remaining
        for part_mask in masks_by_vertex[v]:
            if part_mask & remaining != part_mask:
                continue
            path.append(part_mask)
            if dfs(remaining ^ part_mask, k_used + 1, path):
                pass  # continue to find better
            path.pop()
        return False
    
    # Try k=1, 2, ..., until we find a valid partition
    for target_k in range(1, n + 1):
        best_k[0] = target_k
        best_partition[0] = None
        nodes[0] = 0
        dfs(full_mask, 0, [])
        if best_partition[0] is not None:
            # Found a partition with target_k parts
            # Convert masks to lists
            parts = []
            for m in best_partition[0]:
                parts.append([i for i in range(n) if m & (1 << i)])
            return parts
    return None

def construct():
    adj = _build_graph()
    n = P['n']
    valid = _precompute_valid_parts(adj)
    
    deadline = time.time() + 55  # safety margin
    
    result = _backtrack_min_parts(adj, valid, n, deadline)
    if result is not None:
        # Ensure integers and lists
        parts = []
        for part in result:
            parts.append([int(v) for v in part])
        return parts
    
    # Fallback: greedy from seed
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
