# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_graph():
    n = P['n']
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_clique_mask(adj, mask):
    """Check if all vertices in mask form a clique."""
    if mask == 0:
        return True
    m = mask
    while m:
        lb = m & (-m)
        v = lb.bit_length() - 1
        if (adj[v] & mask) != (mask ^ lb):
            return False
        m ^= lb
    return True

def _is_independent_mask(adj, mask):
    """Check if all vertices in mask form an independent set."""
    if mask == 0:
        return True
    m = mask
    while m:
        lb = m & (-m)
        v = lb.bit_length() - 1
        if adj[v] & mask:
            return False
        m ^= lb
    return True

def _can_add_to_clique(adj, part_mask, v):
    """Check if vertex v can be added to a clique part."""
    vm = 1 << v
    return (adj[v] & part_mask) == part_mask

def _can_add_to_independent(adj, part_mask, v):
    """Check if vertex v can be added to an independent set part."""
    vm = 1 << v
    return (adj[v] & part_mask) == 0

def _greedy_partition(adj):
    """Greedy heuristic to get an initial upper bound."""
    n = P['n']
    part_masks = []
    for v in range(n):
        vm = 1 << v
        best_idx = -1
        for i, pm in enumerate(part_masks):
            if _can_add_to_clique(adj, pm, v) or _can_add_to_independent(adj, pm, v):
                best_idx = i
                break
        if best_idx >= 0:
            part_masks[best_idx] |= vm
        else:
            part_masks.append(vm)
    result = []
    for pm in part_masks:
        lst = [v for v in range(n) if pm & (1 << v)]
        result.append(lst)
    return result

def _exact_search(adj, deadline):
    """Exact branch-and-bound to find minimum cochromatic number."""
    n = P['n']
    full_mask = (1 << n) - 1
    
    best_k = [n]
    best_partition = [_greedy_partition(adj)]
    
    # Cache: (part_mask, v) -> bool for whether v can be added
    # We'll use a dictionary keyed by (mask, v)
    compat_cache = {}
    
    def get_compat(part_mask, v):
        key = (part_mask, v)
        if key not in compat_cache:
            compat_cache[key] = _can_add_to_clique(adj, part_mask, v) or _can_add_to_independent(adj, part_mask, v)
        return compat_cache[key]
    
    def try_assign(mask, parts_masks, k_used, k_max):
        """Try to assign remaining vertices (in mask) to k_max parts."""
        if time.time() > deadline:
            return False
        if mask == 0:
            if k_used < best_k[0]:
                best_k[0] = k_used
                best_partition[0] = []
                for pm in parts_masks:
                    if pm:
                        lst = [v for v in range(n) if pm & (1 << v)]
                        best_partition[0].append(lst)
            return True
        
        verts = [v for v in range(n) if mask & (1 << v)]
        
        # Choose vertex with fewest options (most constrained)
        def count_options(v):
            c = 0
            for pm in parts_masks:
                if get_compat(pm, v):
                    c += 1
            if k_used < k_max:
                c += 1  # can always create new part
            return c
        
        verts.sort(key=count_options)
        v = verts[0]
        vm = 1 << v
        
        # Track which part masks we've already tried to break symmetry
        tried_masks = set()
        
        # Try adding v to each existing part
        for i in range(len(parts_masks)):
            pm = parts_masks[i]
            if pm in tried_masks:
                continue
            tried_masks.add(pm)
            if get_compat(pm, v):
                parts_masks[i] |= vm
                if try_assign(mask ^ vm, parts_masks, k_used, k_max):
                    parts_masks[i] = pm
                    return True
                parts_masks[i] = pm
        
        # Try creating a new part if we haven't exceeded k_max
        if k_used < k_max:
            parts_masks.append(vm)
            if try_assign(mask ^ vm, parts_masks, k_used + 1, k_max):
                parts_masks.pop()
                return True
            parts_masks.pop()
        
        return False
    
    initial_parts = _greedy_partition(adj)
    best_k[0] = len(initial_parts)
    best_partition[0] = initial_parts
    
    # Try smaller k values
    for k in range(best_k[0] - 1, 0, -1):
        if time.time() > deadline:
            break
        parts_masks = []
        if try_assign(full_mask, parts_masks, 0, k):
            best_k[0] = k
            best_partition[0] = []
            for pm in parts_masks:
                if pm:
                    lst = [v for v in range(n) if pm & (1 << v)]
                    best_partition[0].append(lst)
            break
    
    return best_partition[0]

def construct():
    start_time = time.time()
    deadline = start_time + 75.0

    adj = _build_graph()
    greedy_parts = _greedy_partition(adj)
    
    try:
        result = _exact_search(adj, deadline)
    except Exception:
        result = greedy_parts
        
    # Validation and cleanup
    if not isinstance(result, list):
        result = greedy_parts
        
    final_result = []
    seen = set()
    valid = True
    for part in result:
        if not isinstance(part, (list, tuple)):
            valid = False
            break
        for v in part:
            if not isinstance(v, int) or isinstance(v, bool):
                valid = False
                break
            if v in seen:
                valid = False
                break
            seen.add(v)
        if not valid:
            break
            
    if not valid or len(seen) != P['n']:
        result = greedy_parts
        final_result = []
        for part in result:
            final_result.append([int(v) for v in part])
    else:
        for part in result:
            final_result.append([int(v) for v in part])
            
    return final_result
# EVOLVE-BLOCK-END
