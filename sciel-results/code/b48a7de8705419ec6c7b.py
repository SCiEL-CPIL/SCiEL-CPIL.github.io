import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 12
    K = 1
    for i in range(1, n + 1):
        K = K * i // math.gcd(K, i)
    # K = 27720
    
    max_L = 144
    
    # Precompute for each i and each rem in 0..i-1, the bitmask of j in 1..max_L with j % i == rem
    # bitmask has bits 1..max_L set (bit j is 1 if j is a candidate)
    valid_bitmasks = [[0] * i for i in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, max_L + 1):
            rem = j % i
            valid_bitmasks[i][rem] |= (1 << j)
            
    def has_matching(r, L):
        # Check if there's a perfect matching using augmenting paths (Hopcroft-Karp style)
        # Left side: i in 1..n, Right side: j in 1..L
        # Edge (i,j) exists if (r+j) % i == 0 and j <= L
        
        # Build adjacency as bitmasks
        adj = [0] * (n + 1)  # adj[i] = bitmask of valid j's for i
        for i in range(1, n + 1):
            rem = (-r) % i
            adj[i] = valid_bitmasks[i][rem] & ((1 << (L + 1)) - 2)  # mask to j in 1..L
        
        # Check if any i has no candidates
        for i in range(1, n + 1):
            if adj[i] == 0:
                return False
        
        # Simple augmenting path algorithm
        # match[j] = i that is matched to j, 0 if unmatched
        match = [0] * (L + 1)
        
        def try_augment(i, visited):
            # Try to find an augmenting path starting from left node i
            # Get all valid j's for i
            mask = adj[i]
            while mask:
                # Get lowest set bit
                lowbit = mask & (-mask)
                j = lowbit.bit_length() - 1
                mask ^= lowbit
                if visited[j]:
                    continue
                visited[j] = True
                # If j is unmatched, or we can reassign j's current match
                if match[j] == 0 or try_augment(match[j], visited):
                    match[j] = i
                    return True
            return False
        
        for i in range(1, n + 1):
            visited = [False] * (L + 1)
            if not try_augment(i, visited):
                return False
        
        return True

    def compute_f_for_residue(r):
        # Binary search for the least L such that has_matching(r, L) is True
        lo, hi = 1, max_L
        # First check if even max_L works
        if not has_matching(r, max_L):
            return 0
        while lo < hi:
            mid = (lo + hi) // 2
            if has_matching(r, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    # Compute L(r) for all r
    L_vals = [0] * K
    for r in range(K):
        if time.time() > deadline:
            # If we run out of time, fill remaining with 0 and break
            break
        L_vals[r] = compute_f_for_residue(r)

    # Find max L
    max_L_found = max(L_vals) if any(L_vals) else 0
    
    # Find residues with this max L
    best_residues = [r for r in range(K) if L_vals[r] == max_L_found]
    
    # Find smallest m in [0, 1000000] with m % K in best_residues
    best_m = min(best_residues) if best_residues else 0
    
    return int(best_m)
