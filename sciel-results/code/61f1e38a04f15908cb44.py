import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    p = 101
    max_den = 10
    
    # Precompute inverses
    inv = [0] * (max_den + 1)
    for i in range(1, max_den + 1):
        inv[i] = pow(i, -1, p)
    
    # DP: mask_info[mask] = (sum_mod_p, size)
    mask_info = [None] * (1 << max_den)
    mask_info[0] = (0, 0)
    
    for mask in range(1, 1 << max_den):
        lsb = mask & -mask
        i = lsb.bit_length() - 1  # 0-indexed bit position
        prev_mask = mask ^ lsb
        prev_sum, prev_size = mask_info[prev_mask]
        denom = i + 1
        new_sum = (prev_sum + inv[denom]) % p
        new_size = prev_size + 1
        mask_info[mask] = (new_sum, new_size)
    
    # For each residue, find the minimum size subset that maps to it
    min_size_for_residue = [10**9] * p
    best_mask_for_residue = [0] * p
    
    for mask in range(1, 1 << max_den):
        s, sz = mask_info[mask]
        if sz < min_size_for_residue[s]:
            min_size_for_residue[s] = sz
            best_mask_for_residue[s] = mask
    
    # Time check
    if time.time() > deadline:
        # Should not happen, but safety net
        pass
    
    # Construct the result: 101 lists, row r contains the denominators of best_mask_for_residue[r]
    result = []
    for r in range(p):
        mask = best_mask_for_residue[r]
        terms = []
        for i in range(max_den):
            if mask & (1 << i):
                terms.append(i + 1)
        result.append(terms)
    
    return result
