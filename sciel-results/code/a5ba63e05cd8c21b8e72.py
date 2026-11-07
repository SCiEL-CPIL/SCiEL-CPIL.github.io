"""Cochromatic partition solver with iterative deepening and memoization."""
import time

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Precompute validity for all subsets (2^14 = 16384 subsets, very fast)
    # A subset is valid if it's a clique or an independent set
    valid_part = [False] * (1 << n)
    for mask in range(1, 1 << n):
        # Check if clique: for each vertex in mask, all other vertices in mask must be adjacent
        is_cl = True
        temp = mask
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            if (mask ^ lsb) & ~adj_mask[v]:
                is_cl = False
                break
            temp ^= lsb
        if is_cl:
            valid_part[mask] = True
            continue
        # Check if independent: no edges within mask
        is_indep = True
        temp = mask
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            if mask & adj_mask[v]:
                is_indep = False
                break
            temp ^= lsb
        if is_indep:
            valid_part[mask] = True
    
    # Greedy initial solution
    def greedy_partition():
        parts = []
        for v in range(n):
            vmask = 1 << v
            placed = False
            for i in range(len(parts)):
                trial = parts[i] | vmask
                if valid_part[trial]:
                    parts[i] = trial
                    placed = True
                    break
            if not placed:
                parts.append(vmask)
        return parts
    
    initial = greedy_partition()
    best_k = len(initial)
    best_partition = initial[:]
    
    # Order vertices by degree descending for better pruning
    deg = [bin(adj_mask[v]).count('1') for v in range(n)]
    vertex_order = sorted(range(n), key=lambda v: -deg[v])
    
    # Iterative deepening: try k = 1, 2, 3, ... until we find a solution
    # For each k, use backtracking with memoization
    deadline = time.time() + 85
    
    # Memoization: for a given (assigned_mask, k_used, parts_tuple), whether it's solvable
    # We'll use a simpler approach: memoize (assigned_mask, frozenset of part masks)
    memo = {}
    
    def backtrack(k_max, assigned_mask, parts_tuple, idx):
        """Try to assign vertex_order[idx:] to parts with at most k_max total parts."""
        if time.time() > deadline:
            return None
        
        if idx == n:
            # All vertices assigned
            result = []
            for p in parts_tuple:
                if p:
                    verts = []
                    temp = p
                    while temp:
                        lsb = temp & (-temp)
                        verts.append(lsb.bit_length() - 1)
                        temp ^= lsb
                    result.append(verts)
            return result
        
        v = vertex_order[idx]
        vmask = 1 << v
        
        # Key for memoization: (assigned_mask, sorted parts)
        key = (assigned_mask, parts_tuple)
        if key in memo:
            return memo[key]
        
        parts_list = list(parts_tuple)
        
        # Try placing v in each existing part
        tried_empty = False
        for i in range(len(parts_list)):
            if parts_list[i] == 0:
                if tried_empty:
                    continue
                tried_empty = True
                parts_list[i] = vmask
                new_parts = tuple(parts_list)
                res = backtrack(k_max, assigned_mask | vmask, new_parts, idx + 1)
                if res is not None:
                    memo[key] = res
                    return res
                parts_list[i] = 0
            else:
                trial = parts_list[i] | vmask
                if valid_part[trial]:
                    parts_list[i] = trial
                    new_parts = tuple(parts_list)
                    res = backtrack(k_max, assigned_mask | vmask, new_parts, idx + 1)
                    if res is not None:
                        memo[key] = res
                        return res
                    parts_list[i] = trial ^ vmask
        
        # Try creating a new part (if we haven't used all k_max parts)
        num_used = sum(1 for p in parts_list if p != 0)
        if num_used < k_max:
            # Add a new empty part
            parts_list.append(vmask)
            new_parts = tuple(parts_list)
            res = backtrack(k_max, assigned_mask | vmask, new_parts, idx + 1)
            if res is not None:
                memo[key] = res
                return res
            parts_list.pop()
        
        memo[key] = None
        return None
    
    # Try increasing k from 1 upward
    for k in range(1, best_k + 1):
        if time.time() > deadline:
            break
        memo.clear()
        parts_init = [0] * k
        result = backtrack(k, 0, tuple(parts_init), 0)
        if result is not None:
            best_partition = result
            best_k = k
            break
    
    # Convert best_partition to list of lists of ints
    if best_partition and isinstance(best_partition[0], int):
        # It's a list of bitmasks
        final_parts = []
        for p in best_partition:
            if p:
                verts = []
                temp = p
                while temp:
                    lsb = temp & (-temp)
                    verts.append(lsb.bit_length() - 1)
                    temp ^= lsb
                final_parts.append(verts)
    else:
        # It's already a list of lists
        final_parts = best_partition
    
    # Ensure types are correct: Python int, list
    final_result = []
    for part in final_parts:
        final_result.append([int(v) for v in part])
    
    return final_result

# EVOLVE-BLOCK-END
