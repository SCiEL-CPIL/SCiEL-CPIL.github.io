import time

_START = time.time()
_DEADLINE = 80.0

def _elapsed():
    return time.time() - _START

def construct():
    MOD = 101
    N = 10
    
    if _elapsed() > _DEADLINE:
        return [list(range(1, N + 1))] * MOD
    
    # Precompute inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, -1, MOD)
    
    # Precompute all non-empty subsets: (sum_mod, size, mask)
    # Use DP over masks for efficiency
    subset_sum = [0] * (1 << N)
    subset_size = [0] * (1 << N)
    for mask in range(1, 1 << N):
        # Get lowest set bit
        lsb = mask & (-mask)
        bit_idx = lsb.bit_length() - 1  # 0-indexed position
        prev_mask = mask ^ lsb
        subset_sum[mask] = (subset_sum[prev_mask] + inv[bit_idx + 1]) % MOD
        subset_size[mask] = subset_size[prev_mask] + 1
    
    # Group by sum residue: for each residue, list of (size, mask)
    by_residue = [[] for _ in range(MOD)]
    for mask in range(1, 1 << N):
        by_residue[subset_sum[mask]].append((subset_size[mask], mask))
    
    # Sort each residue's list by size (ascending)
    for r in range(MOD):
        by_residue[r].sort(key=lambda x: x[0])
    
    # Find minimum k such that every residue has at least one subset of size <= k
    best_k = N
    for k in range(1, N + 1):
        if all(by_residue[r] and by_residue[r][0][0] <= k for r in range(MOD)):
            best_k = k
            break
    
    # For each residue, pick the smallest-size subset (with size <= best_k)
    result = []
    for r in range(MOD):
        chosen_mask = None
        for size, mask in by_residue[r]:
            if size <= best_k:
                chosen_mask = mask
                break
        if chosen_mask is None:
            chosen_mask = by_residue[r][0][1] if by_residue[r] else 0
        
        terms = []
        for x in range(1, N + 1):
            if chosen_mask & (1 << (x - 1)):
                terms.append(x)
        result.append(terms)
        
    return result
