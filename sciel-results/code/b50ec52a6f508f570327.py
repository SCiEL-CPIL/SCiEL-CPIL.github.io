"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time

def construct():
    """Find largest subset of {1..128} with no three distinct elements a,b,c
    such that a+b, a+c, b+c are all in the set."""
    
    # Based on analysis, the optimal structure is S = L U H
    # where H = {K, ..., 128} and L is a sum-free subset of {1, ..., K-1}
    # such that max(L) + max(L) < K.
    
    best_set = set()
    best_size = 0
    
    # We know K=64 gives size 81.
    # Let's verify and check nearby K values.
    
    for K in range(64, 40, -1):
        H = set(range(K, 129))
        
        # Max element in L must satisfy 2*m < K => m < K/2
        max_L = (K - 1) // 2
        
        if max_L < 1:
            continue
            
        # Largest sum-free subset of {1, ..., max_L} is {floor(max_L/2)+1, ..., max_L}
        start_L = max_L // 2 + 1
        L = set(range(start_L, max_L + 1))
        
        S = L | H
        
        # Verify validity
        if is_valid(S):
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
    
    # Fallback to known good if something went wrong
    if not best_set:
        best_set = set(range(16, 32)) | set(range(64, 129))
        
    return list(best_set)

def is_valid(S):
    """Check if set S has no violating triples."""
    S_set = set(S)
    S_list = sorted(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i+1, n):
            b = S_list[j]
            ab = a + b
            if ab not in S_set:
                continue
            for k in range(j+1, n):
                c = S_list[k]
                ac = a + c
                bc = b + c
                if ac in S_set and bc in S_set:
                    return False
    return True

# EVOLVE-BLOCK-END
