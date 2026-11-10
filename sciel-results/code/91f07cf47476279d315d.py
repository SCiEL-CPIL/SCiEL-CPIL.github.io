import time
import sys
from math import ceil, log2

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Precompute popcount
    popcount = [0] * (1 << n)
    for m in range(1, 1 << n):
        popcount[m] = popcount[m >> 1] + (m & 1)
    
    # Precompute is_clique and is_independent for all masks
    is_clique = [False] * (1 << n)
    is_independent = [False] * (1 << n)
    is_clique[0] = True
    is_independent[0] = True
    
    for mask in range(1, 1 << n):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[mask] = True
            
        if is_independent[rest] and (rest & adj[v]) == 0:
            is_independent[mask] = True
    
    is_valid_part = [False] * (1 << n)
    for m in range(1, 1 << n):
        is_valid_part[m] = is_clique[m] or is_independent[m]
    
    # Precompute max_part_size[mask]: max size of a valid part (clique or indep) subset of mask
    max_part_size = [0] * (1 << n)
    # Iterate through all valid parts and update their supersets
    # Optimization: Only iterate valid parts
    valid_parts_list = [m for m in range(1, 1 << n) if is_valid_part[m]]
    
    # To avoid O(3^n) if there are many valid parts, we can do this:
    # For each mask, max_part_size[mask] = max(popcount[sub] for sub in valid_parts if sub & mask == sub)
    # But updating supersets is better if we are careful.
    # Let's just iterate all masks and find max valid subset? No.
    # Let's iterate valid parts and update supersets.
    # Number of valid parts is roughly 2^n.
    # Sum over valid parts S of 2^(n - |S|).
    # For n=14, this is fast enough.
    
    for S in valid_parts_list:
        sz = popcount[S]
        # Iterate all supersets of S
        # Supersets of S are masks M such that M | S == M
        # Equivalent to: M = S | X where X is a subset of (full_mask ^ S)
        comp = full_mask ^ S
        sub = comp
        while True:
            M = S | sub
            if sz > max_part_size[M]:
                max_part_size[M] = sz
            if sub == 0:
                break
            sub = (sub - 1) & comp
            
    # For each vertex, precompute valid parts containing it
    valid_parts_with = [[] for _ in range(n)]
    for m in valid_parts_list:
        sz = popcount[m]
        # For each vertex in m, add this part
        mm = m
        while mm:
            lsb = mm & -mm
            v = lsb.bit_length() - 1
            valid_parts_with[v].append((sz, m))
            mm ^= lsb
            
    # Sort each list by size descending (larger parts first is often better)
    for v in range(n):
        valid_parts_with[v].sort(key=lambda x: -x[0])
        
    # Precompute global lower bound
    global_max_part = max_part_size[full_mask]
    lb_global = ceil(n / global_max_part) if global_max_part > 0 else n

    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    # Memoization
    memo = {}

    def can_partition(remaining_mask, k_used, k_limit):
        if remaining_mask == 0:
            return True
        if k_used == k_limit:
            return False
            
        # Lower bound check
        rem_size = popcount[remaining_mask]
        if rem_size == 0:
            return True
        max_p = max_part_size[remaining_mask]
        if max_p == 0:
            return False
        lb_rem = ceil(rem_size / max_p)
        if k_used + lb_rem > k_limit:
            return False

        key = (remaining_mask, k_used)
        if key in memo:
            return memo[key]

        # MCV: Pick vertex with fewest candidates
        # Candidates must be subsets of remaining_mask
        best_v = -1
        min_count = float('inf')
        best_candidates = []
        
        mm = remaining_mask
        while mm:
            lsb = mm & -mm
            v = lsb.bit_length() - 1
            mm ^= lsb
            
            # Count valid parts for v that are subsets of remaining_mask
            count = 0
            cands = []
            for sz, S in valid_parts_with[v]:
                if S & remaining_mask == S:
                    count += 1
                    cands.append(S)
                    # Early break if count already exceeds min_count? 
                    # We need the actual list for the best_v, so we can't break too early if we want to store cands.
                    # But we can stop counting if count >= min_count, as we only need the min.
                    # However, we need the cands for the chosen v.
                    # So let's just count first, then fetch? Or fetch and count.
            if count < min_count:
                min_count = count
                best_v = v
                best_candidates = cands
                if min_count == 0:
                    break
                if min_count == 1:
                    # Can't get better than 1
                    # But we should check if other vertices have 0? 
                    # If min_count is 1, it's the best possible non-zero. 
                    # If another vertex has 0, we must pick that one (failure).
                    # So we can't break unless we know no 0s exist.
                    # But if we find 0, we break. If we find 1, we continue to check for 0.
                    pass

        if best_v == -1 or min_count == 0:
            memo[key] = False
            return False

        # Try candidates for best_v
        # Sort candidates? They are already in valid_parts_with order (size desc).
        # We just filtered them.
        for S in best_candidates:
            new_remaining = remaining_mask ^ S
            if can_partition(new_remaining, k_used + 1, k_limit):
                memo[key] = True
                return True
        
        memo[key] = False
        return False

    def find_partition(remaining_mask, k_used, k_limit):
        if remaining_mask == 0:
            return []
        if k_used == k_limit:
            return None
            
        rem_size = popcount[remaining_mask]
        if rem_size == 0:
            return []
        max_p = max_part_size[remaining_mask]
        if max_p == 0:
            return None
        lb_rem = ceil(rem_size / max_p)
        if k_used + lb_rem > k_limit:
            return None

        # MCV
        best_v = -1
        min_count = float('inf')
        best_candidates = []
        
        mm = remaining_mask
        while mm:
            lsb = mm & -mm
            v = lsb.bit_length() - 1
            mm ^= lsb
            
            count = 0
            cands = []
            for sz, S in valid_parts_with[v]:
                if S & remaining_mask == S:
                    count += 1
                    cands.append(S)
            if count < min_count:
                min_count = count
                best_v = v
                best_candidates = cands
                if min_count == 0:
                    break

        if best_v == -1 or min_count == 0:
            return None

        for S in best_candidates:
            new_remaining = remaining_mask ^ S
            sub = find_partition(new_remaining, k_used + 1, k_limit)
            if sub is not None:
                verts = []
                m = S
                while m:
                    l = m & -m
                    verts.append(l.bit_length() - 1)
                    m ^= l
                return [verts] + sub
        
        return None

    for k in range(lb_global, n + 1):
        if time.time() - start_time > deadline:
            break
        
        memo.clear()
        if can_partition(full_mask, 0, k):
            partition = find_partition(full_mask, 0, k)
            if partition is not None:
                result = []
                for part in partition:
                    result.append([int(v) for v in part])
                return result
    
    # Fallback: greedy
    return _greedy_construct()

def _greedy_construct():
    """Fallback greedy constructor."""
    n = P['n']
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u, v), max(u, v)))
    
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            is_cli = True
            is_ind = True
            for u in part:
                e = (min(u, v), max(u, v))
                if e in edge_set:
                    is_ind = False
                else:
                    is_cli = False
            if is_cli or is_ind:
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    
    return parts

# EVOLVE-BLOCK-END
