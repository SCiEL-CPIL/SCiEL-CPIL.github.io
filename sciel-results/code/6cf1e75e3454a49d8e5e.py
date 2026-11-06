def construct():
    n = 256
    best_size = 0
    best_set = []
    
    # Precompute masks for divisibility
    # mask_div[x] has bit i set if i % x == 0
    # Max sum is 2 * 256 = 512
    max_sum = 2 * n
    mask_div = [0] * (n + 1)
    for x in range(1, n + 1):
        mask = 0
        for multiple in range(x, max_sum + 1, x):
            mask |= (1 << multiple)
        mask_div[x] = mask

    def solve(idx, S, sums_mask):
        nonlocal best_size, best_set
        
        # Pruning 1: Upper bound
        # We can add at most idx more elements (idx, idx-1, ..., 1)
        if len(S) + idx <= best_size:
            return
            
        if idx == 0:
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
            return
            
        x = idx
        
        # Option 1: Try to add x
        # Check if x divides any existing sum
        if (sums_mask & mask_div[x]) == 0:
            # Can add
            # Optimization: Create new list only if needed?
            # S + [x] is fine for N=256
            new_S = S + [x]
            new_sums_mask = sums_mask
            # Add new sums: 2*x and x+y for y in S
            new_sums_mask |= (1 << (2 * x))
            for y in S:
                new_sums_mask |= (1 << (x + y))
            
            solve(idx - 1, new_S, new_sums_mask)
            
        # Option 2: Skip x
        # Pruning: If even adding all remaining elements (idx-1 of them) doesn't beat best
        if len(S) + (idx - 1) > best_size:
            solve(idx - 1, S, sums_mask)

    solve(n, [], 0)
    
    return best_set
