"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time

def construct():
    """Find largest subset of {1..128} with no three distinct elements a,b,c
    such that a+b, a+c, b+c are all in the set."""
    
    deadline = time.time() + 85  # safety margin before 90s kill
    
    best_set = set()
    best_size = 0
    
    # Strategy: S = L ∪ H where H = {K, ..., 128} and L ⊆ {1, ..., K-1}
    # Key constraint: for any a,b,c ∈ S (distinct), not all of a+b, a+c, b+c ∈ S
    # 
    # If H = {K,...,128}, then for a,b,c all in H: sums ≥ 2K+2 > 128 (if K ≥ 64), safe.
    # For mixed triples (some in L, some in H): need to check carefully.
    #
    # Simplest safe construction: L is sum-free (no x+y in L for x,y in L, x≠y)
    # AND max(L) + max(H) ≤ 128 is NOT needed since sums can exceed 128.
    # The real constraint: if a,b ∈ L and c ∈ H, then a+b, a+c, b+c must not ALL be in S.
    # Since a+c ≥ a+K and b+c ≥ b+K, these are in H if ≤ 128.
    # a+b could be in L or in H or > 128.
    
    # Try multiple K values and different L constructions
    
    for K in range(64, 30, -1):
        if time.time() > deadline:
            break
        
        H = set(range(K, 129))
        max_L = K - 1
        
        if max_L < 1:
            continue
        
        # Construction 1: L = {floor(max_L/2)+1, ..., max_L} (largest sum-free in [1,max_L])
        start_L = max_L // 2 + 1
        L1 = set(range(start_L, max_L + 1))
        S1 = L1 | H
        if is_valid_fast(S1):
            if len(S1) > best_size:
                best_size = len(S1)
                best_set = set(S1)
        
        # Construction 2: L = {1, ..., floor(max_L/2)} (also sum-free? No, 1+2=3 might be in L)
        # Actually {1,...,m} is NOT sum-free. Skip.
        
        # Construction 3: Try L = odd numbers in [1, max_L]
        # Odds + Odds = Even, so if L is all odds, no sum of two distinct odds is in L (it's even).
        # But we need max(L) + max(L) < K? No, we need to check cross terms.
        L3 = set(range(1, max_L + 1, 2))
        S3 = L3 | H
        if is_valid_fast(S3):
            if len(S3) > best_size:
                best_size = len(S3)
                best_set = set(S3)
        
        # Construction 4: L = numbers in (max_L/3, max_L] that are sum-free
        # More generally, try L = {x : x > max_L/2} which is sum-free
        # This is same as Construction 1.
        
        # Construction 5: Try slightly different K offsets
        # What if H = {K, ..., 128} but we also include some elements between max_L+1 and K-1?
        # Those would be in the "gap" - call it M.
        # For simplicity, stick with L ∪ H for now.
    
    # Fallback to known good
    if not best_set:
        best_set = set(range(16, 32)) | set(range(64, 129))
    
    result = list(best_set)
    # Ensure all are Python int
    return [int(x) for x in result]

def is_valid_fast(S):
    """Fast check: no three distinct a,b,c in S with a+b, a+c, b+c all in S."""
    S_set = set(S)
    S_list = sorted(S)
    n = len(S_list)
    
    # For each pair (a,b) with a<b, check if a+b is in S.
    # If so, for each c > b, check if both a+c and b+c are in S.
    # Optimization: precompute for each pair (a,b) whether a+b is in S.
    
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            ab = a + b
            if ab not in S_set:
                continue
            # Now check all c > b
            for k in range(j + 1, n):
                c = S_list[k]
                ac = a + c
                if ac not in S_set:
                    continue
                bc = b + c
                if bc in S_set:
                    return False
    return True

# EVOLVE-BLOCK-END
