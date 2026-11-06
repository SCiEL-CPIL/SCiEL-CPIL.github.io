import math
from math import gcd

def construct():
    MAX_N = 256
    N = 255
    values = list(range(2, 257))
    weights = [1.0 / v for v in values]
    
    # 1. Precompute incompatibility masks
    # incompat[i] has bit j set if lcm(values[i], values[j]) <= 256
    incompat = [0] * N
    for i in range(N):
        vi = values[i]
        for j in range(N):
            if i == j:
                continue
            vj = values[j]
            g = gcd(vi, vj)
            if (vi // g) * vj <= 256:
                incompat[i] |= (1 << j)
                
    all_mask = (1 << N) - 1
    compat = [0] * N
    for i in range(N):
        # Compatible with self and all j not in incompat
        compat[i] = all_mask ^ incompat[i]

    # 2. Order nodes by weight (descending)
    # We want to pick high weight nodes first to maximize score and prune early
    order = sorted(range(N), key=lambda i: weights[i], reverse=True)
    
    # Precompute suffix masks for the order
    # suffix_mask[i] is the bitmask of nodes in order[i], order[i+1], ... order[N-1]
    suffix_mask = [0] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_mask[i] = suffix_mask[i + 1] | (1 << order[i])
        
    # Helper to sum weights of bits in a mask
    # We can precompute weight sums for small chunks? Or just iterate.
    # Iterating set bits is fast enough for N=255 if pruned well.
    def sum_weights(mask):
        s = 0.0
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            s += weights[idx]
            mask ^= lsb
        return s

    # 3. Initial Greedy to set a good lower bound (best_score)
    # This is a heuristic initialization, not the core solver logic.
    best_score = 0.0
    best_set_mask = 0
    
    # Simple greedy: iterate order, add if compatible
    cur_mask = 0
    cur_sum = 0.0
    cur_compat = all_mask
    for node in order:
        bit = 1 << node
        if bit & cur_compat:
            cur_mask |= bit
            cur_sum += weights[node]
            cur_compat &= compat[node]
            
    if cur_sum > best_score:
        best_score = cur_sum
        best_set_mask = cur_mask

    # 4. Branch and Bound
    import sys
    sys.setrecursionlimit(5000)
    
    def dfs(pos, selected_mask, current_compat_mask, current_sum):
        nonlocal best_score, best_set_mask
        
        if pos == N:
            if current_sum > best_score:
                best_score = current_sum
                best_set_mask = selected_mask
            return
            
        # Upper Bound
        # Nodes remaining in order are in suffix_mask[pos]
        # Available nodes are those in suffix_mask[pos] AND current_compat_mask
        avail_mask = suffix_mask[pos] & current_compat_mask
        
        # Quick check: if no available nodes, we can just check if current_sum is best
        if avail_mask == 0:
            if current_sum > best_score:
                best_score = current_sum
                best_set_mask = selected_mask
            return

        ub = current_sum + sum_weights(avail_mask)
        
        if ub <= best_score:
            return
        
        node = order[pos]
        bit = 1 << node
        
        # Branch 1: Include node (if compatible)
        if bit & current_compat_mask:
            new_selected = selected_mask | bit
            new_compat = current_compat_mask & compat[node]
            new_sum = current_sum + weights[node]
            dfs(pos + 1, new_selected, new_compat, new_sum)
            
        # Branch 2: Exclude node
        dfs(pos + 1, selected_mask, current_compat_mask, current_sum)

    dfs(0, 0, all_mask, 0.0)
    
    # Extract result
    result = []
    for i in range(N):
        if best_set_mask & (1 << i):
            result.append(values[i])
            
    # Ensure non-empty (problem guarantees non-empty subset exists, e.g., {256})
    if not result:
        result = [256]
        
    return result
