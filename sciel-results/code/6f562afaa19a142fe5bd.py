# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_graph():
    n = P['n']
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_clique_mask(adj, mask):
    if mask == 0:
        return True
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        if (adj[v] & mask) != (mask ^ lsb):
            return False
        m ^= lsb
    return True

def _is_independent_mask(adj, mask):
    if mask == 0:
        return True
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        if adj[v] & mask:
            return False
        m ^= lsb
    return True

def _precompute_valid_masks(adj):
    """Precompute all valid (clique or independent) masks."""
    n = P['n']
    valid = [False] * (1 << n)
    valid[0] = True
    for mask in range(1, 1 << n):
        valid[mask] = _is_clique_mask(adj, mask) or _is_independent_mask(adj, mask)
    return valid

def _lower_bound(adj, valid):
    """Compute a lower bound on the cochromatic number."""
    n = P['n']
    full_mask = (1 << n) - 1
    
    # The cochromatic number is at least the maximum of:
    # 1. The clique number (each clique must be in separate parts or same clique part)
    # 2. The independence number of complement (chromatic number of complement)
    
    # Simple lower bound: we need at least ceil(n / alpha) parts where alpha is max independent set size
    # And at least ceil(n / omega) where omega is max clique size
    
    max_clique_size = 1
    max_indep_size = 1
    
    for mask in range(1, 1 << n):
        if valid[mask]:
            popcount = bin(mask).count('1')
            if _is_clique_mask(adj, mask):
                max_clique_size = max(max_clique_size, popcount)
            if _is_independent_mask(adj, mask):
                max_indep_size = max(max_indep_size, popcount)
    
    # Lower bound: at least max(ceil(n/max_clique), ceil(n/max_indep))
    lb1 = (n + max_clique_size - 1) // max_clique_size
    lb2 = (n + max_indep_size - 1) // max_indep_size
    return max(lb1, lb2, 1)

def _backtrack_search(adj, valid, k, deadline):
    """Try to find a partition into exactly k valid parts using backtracking."""
    n = P['n']
    full_mask = (1 << n) - 1
    
    # Sort vertices by degree (descending) for better pruning
    degrees = [bin(adj[v]).count('1') for v in range(n)]
    vertex_order = sorted(range(n), key=lambda v: -degrees[v])
    
    parts = [0] * k  # bitmask for each part
    part_types = [None] * k  # 'clique', 'indep', or None (undetermined)
    
    def can_add(part_idx, v_mask, ptype):
        """Check if v_mask can be added to part part_idx of type ptype."""
        if ptype is None:
            # Try both types
            trial = parts[part_idx] | v_mask
            return _is_clique_mask(adj, trial) or _is_independent_mask(adj, trial)
        elif ptype == 'clique':
            trial = parts[part_idx] | v_mask
            return _is_clique_mask(adj, trial)
        else:  # indep
            trial = parts[part_idx] | v_mask
            return _is_independent_mask(adj, trial)
    
    def determine_type(part_idx, v_mask):
        """Determine what type a part should be after adding v_mask."""
        trial = parts[part_idx] | v_mask
        is_c = _is_clique_mask(adj, trial)
        is_i = _is_independent_mask(adj, trial)
        if is_c and is_i:
            return None  # Can be either
        elif is_c:
            return 'clique'
        elif is_i:
            return 'indep'
        else:
            return 'invalid'
    
    def solve(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            # All vertices assigned, check no empty parts
            for i in range(k):
                if parts[i] == 0:
                    return False
            return True
        
        v = vertex_order[idx]
        v_mask = 1 << v
        
        # Try adding to existing parts
        for i in range(k):
            if can_add(i, v_mask, part_types[i]):
                old_parts = parts[i]
                old_type = part_types[i]
                parts[i] |= v_mask
                
                new_type = determine_type(i, v_mask)
                if new_type != 'invalid':
                    part_types[i] = new_type if new_type is not None else old_type
                    if solve(idx + 1):
                        return True
                    part_types[i] = old_type
                
                parts[i] = old_parts
        
        # Try creating a new part (only if we haven't used all k parts)
        for i in range(k):
            if parts[i] == 0:
                parts[i] = v_mask
                part_types[i] = None  # Single vertex is both
                if solve(idx + 1):
                    return True
                parts[i] = 0
                part_types[i] = None
        
        return False
    
    if solve(0):
        # Reconstruct partition
        result = []
        for i in range(k):
            if parts[i] != 0:
                lst = [v for v in range(n) if parts[i] & (1 << v)]
                result.append(lst)
        return result
    
    return None

def _exact_dp(adj, valid, deadline):
    """Exact DP to find minimum cochromatic number."""
    n = P['n']
    full_mask = (1 << n) - 1
    
    INF = n + 1
    dp = [INF] * (1 << n)
    dp[0] = 0
    
    for mask in range(1, 1 << n):
        if time.time() > deadline:
            return None
        if valid[mask]:
            dp[mask] = 1
            continue
        lsb = mask & (-mask)
        sub = mask
        while sub > 0:
            if sub & lsb and valid[sub]:
                rest = mask ^ sub
                if dp[rest] + 1 < dp[mask]:
                    dp[mask] = dp[rest] + 1
            sub = (sub - 1) & mask
    
    if dp[full_mask] >= INF:
        return None
    
    # Reconstruct partition
    parts_masks = []
    mask = full_mask
    while mask > 0:
        lsb = mask & (-mask)
        sub = mask
        best_sub = None
        while sub > 0:
            if sub & lsb and valid[sub]:
                rest = mask ^ sub
                if dp[rest] + 1 == dp[mask]:
                    best_sub = sub
                    break
            sub = (sub - 1) & mask
        if best_sub is None:
            return None
        parts_masks.append(best_sub)
        mask ^= best_sub
    
    result = []
    for pm in parts_masks:
        lst = [v for v in range(n) if pm & (1 << v)]
        result.append(lst)
    return result

def _greedy_partition(adj):
    """Greedy heuristic to get an initial upper bound."""
    n = P['n']
    parts = []
    part_masks = []
    for v in range(n):
        vm = 1 << v
        best_idx = -1
        for i, pm in enumerate(part_masks):
            trial = pm | vm
            if _is_clique_mask(adj, trial) or _is_independent_mask(adj, trial):
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

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s kill
    
    adj = _build_graph()
    n = P['n']
    
    # Get greedy solution as fallback
    greedy_parts = _greedy_partition(adj)
    greedy_count = len(greedy_parts)
    
    # Precompute valid masks
    valid = _precompute_valid_masks(adj)
    
    # Compute lower bound
    lb = _lower_bound(adj, valid)
    
    result = None
    
    # Try backtracking search from lower bound up to greedy count
    for k in range(lb, greedy_count + 1):
        if time.time() > deadline:
            break
        if k >= greedy_count:
            break  # No need to try k >= greedy_count, we already have that solution
        bt_result = _backtrack_search(adj, valid, k, deadline)
        if bt_result is not None:
            result = bt_result
            break
    
    # If backtracking didn't find a better solution, try DP
    if result is None and time.time() < deadline:
        dp_result = _exact_dp(adj, valid, deadline)
        if dp_result is not None:
            result = dp_result
    
    # Fallback to greedy
    if result is None:
        result = greedy_parts
    
    # Validation and cleanup
    if not isinstance(result, list):
        result = greedy_parts
    
    final_result = []
    seen = set()
    valid_result = True
    for part in result:
        if not isinstance(part, (list, tuple)):
            valid_result = False
            break
        for v in part:
            if not isinstance(v, int) or isinstance(v, bool):
                valid_result = False
                break
            if v in seen:
                valid_result = False
                break
            seen.add(v)
        if not valid_result:
            break
            
    if not valid_result or len(seen) != n:
        result = greedy_parts
        final_result = []
        for part in result:
            final_result.append([int(v) for v in part])
    else:
        for part in result:
            final_result.append([int(v) for v in part])
            
    return final_result
# EVOLVE-BLOCK-END
