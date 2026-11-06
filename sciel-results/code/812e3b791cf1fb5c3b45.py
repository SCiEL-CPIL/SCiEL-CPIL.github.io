"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    N = 256
    
    # Build conflict sets
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1
    
    # conflict[a] = set of b in [1,N] such that a+b is a perfect square
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for sq in squares:
            b = sq - a
            if 1 <= b <= N:
                conflict[a].add(b)
    
    # Numbers where 2a is a perfect square are forbidden (self-conflict)
    forbidden = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            forbidden.add(a)
    
    valid_numbers = [x for x in range(1, N + 1) if x not in forbidden]
    M = len(valid_numbers)
    num_to_idx = {x: i for i, x in enumerate(valid_numbers)}
    idx_to_num = valid_numbers
    
    # Build adjacency as bitmasks (Python ints as arbitrary-precision bitsets)
    adj = [0] * M
    for i, a in enumerate(valid_numbers):
        for b in conflict[a]:
            if b in num_to_idx:
                j = num_to_idx[b]
                if j != i:  # exclude self
                    adj[i] |= (1 << j)
    
    # Compute degrees
    degrees = [bin(adj[i]).count('1') for i in range(M)]
    
    # Branch and bound for maximum independent set
    # State: (candidates_mask, selected_mask, current_size)
    # We use a recursive approach with pruning
    
    best_size = 0
    best_mask = 0
    node_count = 0
    
    # Sort vertices by degree for better branching
    # Lower degree vertices first (fewer conflicts, more likely to be in MIS)
    degree_order = sorted(range(M), key=lambda i: degrees[i])
    
    # Map: position in degree_order -> original index
    pos_to_orig = degree_order
    orig_to_pos = {orig: pos for pos, orig in enumerate(degree_order)}
    
    # Rebuild adjacency in degree order
    adj_ordered = [0] * M
    for pos, orig in enumerate(degree_order):
        mask = 0
        for j in range(M):
            if (adj[orig] >> j) & 1:
                mask |= (1 << orig_to_pos[j])
        adj_ordered[pos] = mask
    
    def popcount(x):
        return bin(x).count('1')
    
    def branch_and_bound(candidates, selected, size, depth):
        nonlocal best_size, best_mask, node_count
        
        if time.time() > deadline:
            return
        
        node_count += 1
        
        if size + popcount(candidates) <= best_size:
            return
        
        if candidates == 0:
            if size > best_size:
                best_size = size
                best_mask = selected
            return
        
        # Pick vertex with minimum degree in the candidate subgraph
        # For speed, just pick the first candidate
        # Actually, let's pick the vertex with min degree among candidates
        cand_list = []
        temp = candidates
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            cand_list.append(idx)
            temp ^= lsb
        
        if not cand_list:
            return
        
        # Find vertex with minimum degree in candidate subgraph
        min_deg = M + 1
        min_idx = cand_list[0]
        for idx in cand_list:
            deg_in_cand = popcount(adj_ordered[idx] & candidates)
            if deg_in_cand < min_deg:
                min_deg = deg_in_cand
                min_idx = idx
        
        # Branch 1: Include min_idx
        # Remove min_idx and its neighbors from candidates
        new_candidates = candidates & ~(1 << min_idx) & ~adj_ordered[min_idx]
        new_selected = selected | (1 << min_idx)
        branch_and_bound(new_candidates, new_selected, size + 1, depth + 1)
        
        # Branch 2: Exclude min_idx
        new_candidates = candidates & ~(1 << min_idx)
        branch_and_bound(new_candidates, selected, size, depth + 1)
    
    # Initial call
    all_candidates = (1 << M) - 1
    branch_and_bound(all_candidates, 0, 0, 0)
    
    # Convert best_mask back to original indices
    result = []
    for pos in range(M):
        if (best_mask >> pos) & 1:
            result.append(idx_to_num[pos_to_orig[pos]])
    
    if not result:
        # Fallback: use numbers ≡ 1 mod 3
        result = [x for x in range(1, N + 1, 3) if x not in forbidden]
    
    return sorted(result)
# EVOLVE-BLOCK-END
