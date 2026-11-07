"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_graph():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_clique(vertices, adj):
    for i, u in enumerate(vertices):
        for v in vertices[i+1:]:
            if v not in adj[u]:
                return False
    return True

def _is_independent(vertices, adj):
    for i, u in enumerate(vertices):
        for v in vertices[i+1:]:
            if v in adj[u]:
                return False
    return True

def _is_valid_part(vertices, adj):
    if not vertices:
        return False
    if _is_clique(vertices, adj):
        return True
    if _is_independent(vertices, adj):
        return True
    return False

def _precompute_valid_parts(n, adj):
    """Precompute all valid subsets (cliques or independent sets) as bitmasks."""
    valid_masks = set()
    all_vertices = list(range(n))
    for size in range(1, n+1):
        from itertools import combinations
        for combo in combinations(all_vertices, size):
            if _is_valid_part(list(combo), adj):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                valid_masks.add(mask)
    return valid_masks

def _greedy_upper_bound(n, adj, valid_masks):
    """Get a greedy upper bound on the number of parts."""
    remaining = (1 << n) - 1
    count = 0
    while remaining:
        # Find the largest valid subset of remaining vertices
        best_mask = 0
        best_size = 0
        # Try all valid masks that are subsets of remaining
        for vm in valid_masks:
            if vm & remaining == vm:
                sz = bin(vm).count('1')
                if sz > best_size:
                    best_size = sz
                    best_mask = vm
        if best_mask == 0:
            # Shouldn't happen, but fallback
            # Find any single vertex
            for v in range(n):
                if remaining & (1 << v):
                    best_mask = 1 << v
                    break
        remaining &= ~best_mask
        count += 1
    return count

def _backtrack_min_parts(n, adj, valid_masks, deadline):
    """Backtracking search for minimum number of parts."""
    all_vertices_mask = (1 << n) - 1
    
    # Sort vertices by degree (descending) for better pruning
    degrees = [len(adj[v]) for v in range(n)]
    vertex_order = sorted(range(n), key=lambda v: -degrees[v])
    
    best_count = n  # Upper bound: each vertex its own part
    
    # Greedy to get initial upper bound
    greedy_ub = _greedy_upper_bound(n, adj, valid_masks)
    best_count = min(best_count, greedy_ub)
    
    # Try to find a solution with fewer parts
    for target in range(1, best_count):
        if time.time() > deadline:
            break
        
        found = [False]
        
        def backtrack(remaining, parts_used, parts_list):
            if time.time() > deadline:
                return True  # Signal timeout
            if found[0]:
                return True
            if parts_used >= target:
                return False
            if remaining == 0:
                found[0] = True
                return True
            
            # Find the highest-degree vertex in remaining
            # to use as a branching point
            v = -1
            max_deg = -1
            for u in range(n):
                if remaining & (1 << u):
                    if degrees[u] > max_deg:
                        max_deg = degrees[u]
                        v = u
            
            if v == -1:
                found[0] = True
                return True
            
            # Try all valid parts containing v that are subsets of remaining
            v_bit = 1 << v
            candidates = []
            for vm in valid_masks:
                if vm & v_bit and (vm & remaining) == vm:
                    candidates.append(vm)
            
            # Sort candidates by size (larger first for better pruning)
            candidates.sort(key=lambda m: -bin(m).count('1'))
            
            for vm in candidates:
                new_remaining = remaining & ~vm
                parts_list.append(vm)
                if backtrack(new_remaining, parts_used + 1, parts_list):
                    return True
                parts_list.pop()
            
            return False
        
        parts_list = []
        if backtrack(all_vertices_mask, 0, parts_list):
            if found[0]:
                best_count = target
                break
    
    return best_count

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = P['n']
    adj = _build_graph()
    
    # Precompute valid parts
    valid_masks = _precompute_valid_parts(n, adj)
    
    # Find minimum number of parts
    min_parts = _backtrack_min_parts(n, adj, valid_masks, deadline)
    
    # Now construct the actual partition with min_parts parts
    # Use backtracking to find a valid partition
    all_vertices_mask = (1 << n) - 1
    degrees = [len(adj[v]) for v in range(n)]
    
    result_parts = None
    
    def construct_partition(remaining, parts_used, target, parts_list):
        if time.time() > deadline:
            return None
        if parts_used >= target:
            if remaining == 0:
                return parts_list[:]
            return None
        if remaining == 0:
            if parts_used < target:
                # We have fewer parts than target, which is fine
                return parts_list[:]
            return None
        
        # Find highest-degree vertex in remaining
        v = -1
        max_deg = -1
        for u in range(n):
            if remaining & (1 << u):
                if degrees[u] > max_deg:
                    max_deg = degrees[u]
                    v = u
        
        if v == -1:
            return parts_list[:] if parts_used <= target else None
        
        v_bit = 1 << v
        candidates = []
        for vm in valid_masks:
            if vm & v_bit and (vm & remaining) == vm:
                candidates.append(vm)
        
        candidates.sort(key=lambda m: -bin(m).count('1'))
        
        for vm in candidates:
            new_remaining = remaining & ~vm
            parts_list.append(vm)
            result = construct_partition(new_remaining, parts_used + 1, target, parts_list)
            parts_list.pop()
            if result is not None:
                return result
        
        return None
    
    # Try to construct partition with min_parts
    parts_list = []
    result_parts = construct_partition(all_vertices_mask, 0, min_parts, parts_list)
    
    # If that failed (timeout or other issue), use greedy
    if result_parts is None:
        remaining = all_vertices_mask
        result_parts = []
        while remaining and time.time() < deadline:
            best_mask = 0
            best_size = 0
            for vm in valid_masks:
                if vm & remaining == vm:
                    sz = bin(vm).count('1')
                    if sz > best_size:
                        best_size = sz
                        best_mask = vm
            if best_mask == 0:
                for v in range(n):
                    if remaining & (1 << v):
                        best_mask = 1 << v
                        break
            result_parts.append(best_mask)
            remaining &= ~best_mask
    
    # Convert masks to lists of vertices
    final_parts = []
    for mask in result_parts:
        part = []
        for v in range(n):
            if mask & (1 << v):
                part.append(v)
        if part:
            final_parts.append(part)
    
    # Verify the partition is valid
    # Check all vertices are covered exactly once
    seen = set()
    for part in final_parts:
        for v in part:
            if v in seen:
                # Invalid - fallback to greedy
                break
            seen.add(v)
        else:
            continue
        break
    else:
        if len(seen) == n:
            # Valid partition
            return [list(p) for p in final_parts]
    
    # Fallback: simple greedy
    es = {tuple(e) for e in P['edges']}
    parts = []
    for v in range(n):
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
            if all(hits) or not any(hits):
                part.append(v)
                break
        else:
            parts.append([v])
    return parts
# EVOLVE-BLOCK-END
