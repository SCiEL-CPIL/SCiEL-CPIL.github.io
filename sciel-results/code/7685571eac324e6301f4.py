import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin

    # Try to find a graph with n=20
    # If that fails, try smaller n
    
    # We will search for n=20 first.
    # If we find a valid graph, return it.
    
    result = _search_graph(20, deadline - 5)
    if result:
        return result

    # Fallback to smaller n if 20 failed
    for n in [19, 18, 17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 5:
            break
        result = _search_graph(n, deadline - 5)
        if result:
            return result

    # Final fallback: a trivially valid small graph
    return {'n': 5, 'edges': []}

def _search_graph(n, deadline):
    rng = random.Random(42)
    
    # Initialize random graph
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < 0.5:
                adj[i] |= 1 << j
                adj[j] |= 1 << i

    # Define which sizes to track during search
    # Sizes 6 and 7 are the most critical and frequent
    # We track 6, 7, 8 to be safe
    track_sizes = [6, 7, 8]
    
    # Precompute subsets for tracked sizes
    subsets = []
    masks = []
    sizes = []
    for size in track_sizes:
        for subset in itertools.combinations(range(n), size):
            mask = 0
            for v in subset:
                mask |= 1 << v
            subsets.append(subset)
            masks.append(mask)
            sizes.append(size)
            
    num_subsets = len(subsets)
    is_reg = [False] * num_subsets
    viol_indices = []
    
    # Initial check
    for i in range(num_subsets):
        if _is_regular(adj, subsets[i], masks[i]):
            is_reg[i] = True
            viol_indices.append(i)
            
    if not viol_indices:
        # Check all sizes to be sure
        if _is_valid_full(adj, n):
            return _to_result(adj, n)
        # If not valid full, but no tracked violations, we might have larger violations.
        # We can try to continue search or return None.
        # Let's try a few more steps or just return None to try smaller n?
        # Actually, if we track 6,7,8 and they are clean, 9+ are very unlikely.
        # But we must verify.
        pass

    # Map edge to list of subset indices containing it
    edge_to_indices = {}
    for i, subset in enumerate(subsets):
        s_set = set(subset)
        for u in subset:
            for v in subset:
                if u < v:
                    key = (u, v)
                    if key not in edge_to_indices:
                        edge_to_indices[key] = []
                    edge_to_indices[key].append(i)

    # Local Search
    while time.time() < deadline:
        if not viol_indices:
            # No violations in tracked sizes
            if _is_valid_full(adj, n):
                return _to_result(adj, n)
            else:
                # There are violations in untracked sizes (9+)
                # We can try to find them and flip an edge in one
                viol_subset = _find_violation_full(adj, n)
                if viol_subset:
                    u, v = rng.sample(viol_subset, 2)
                    _flip_edge(adj, u, v)
                    # Update tracked subsets
                    _update_violations(adj, subsets, masks, is_reg, viol_indices, edge_to_indices, (u,v))
                else:
                    return _to_result(adj, n)
            continue

        # Pick a random violating subset from tracked ones
        idx = rng.choice(viol_indices)
        subset = subsets[idx]
        u, v = rng.sample(subset, 2)
        
        # Flip edge
        _flip_edge(adj, u, v)
        
        # Update violations for all tracked subsets containing (u,v)
        _update_violations(adj, subsets, masks, is_reg, viol_indices, edge_to_indices, (u,v))

    # Time up, return None (or best effort if we want, but must be valid)
    # Check if current state is valid
    if _is_valid_full(adj, n):
        return _to_result(adj, n)
    return None

def _is_regular(adj, subset, mask):
    if len(subset) < 2:
        return True
    degs = set()
    for v in subset:
        degs.add((adj[v] & mask).bit_count())
    return len(degs) == 1

def _is_valid_full(adj, n):
    # Check all subsets of size >= 6
    for size in range(6, n+1):
        for subset in itertools.combinations(range(n), size):
            mask = 0
            for v in subset:
                mask |= 1 << v
            if _is_regular(adj, subset, mask):
                return False
    return True

def _find_violation_full(adj, n):
    for size in range(6, n+1):
        for subset in itertools.combinations(range(n), size):
            mask = 0
            for v in subset:
                mask |= 1 << v
            if _is_regular(adj, subset, mask):
                return subset
    return None

def _flip_edge(adj, u, v):
    adj[u] ^= 1 << v
    adj[v] ^= 1 << u

def _update_violations(adj, subsets, masks, is_reg, viol_indices, edge_to_indices, edge):
    affected = edge_to_indices.get(edge, [])
    for idx in affected:
        old_reg = is_reg[idx]
        new_reg = _is_regular(adj, subsets[idx], masks[idx])
        if old_reg != new_reg:
            is_reg[idx] = new_reg
            if new_reg:
                viol_indices.append(idx)
            else:
                # Remove from viol_indices
                # Find index of idx in viol_indices
                # To make this O(1), we could maintain a map idx -> pos in viol_indices
                # For now, use list.remove which is O(N).
                # Given N is small (violations), this is okay.
                try:
                    viol_indices.remove(idx)
                except ValueError:
                    pass

def _to_result(adj, n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return {'n': n, 'edges': edges}
