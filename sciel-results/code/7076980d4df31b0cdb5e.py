import random

def construct():
    max_n = 512
    max_size = 48
    offset = 25000 # Sufficient for sum range
    
    def check_validity(C, x):
        # Check if x is mean of any subset of C (size >= 2)
        # Sum(t - x) = 0 for T subset C, |T| >= 2
        if len(C) < 2:
            # Can't form subset of size >= 2 from C if C is empty or size 1?
            # If C is empty, no subset.
            # If C has 1 elem, no subset of size 2.
            pass 
        else:
            values = [t - x for t in C]
            # DP for sum 0
            # dp[k] bitset
            M = len(C)
            dp = [0] * (M + 1)
            dp[0] = 1 << offset
            
            for v in values:
                # Iterate backwards to avoid using same element twice
                # We only care up to size M
                for k in range(M - 1, -1, -1):
                    if dp[k]:
                        if v >= 0:
                            dp[k+1] |= (dp[k] << v)
                        else:
                            dp[k+1] |= (dp[k] >> (-v))
            
            # Check if 0 is achievable with k >= 2
            # Target sum 0 corresponds to bit at offset
            target_bit = 1 << offset
            for k in range(2, M + 1):
                if dp[k] & target_bit:
                    return False
        
        # Check if any existing a in C is now mean of subset involving x
        # For each a in C:
        # Need T' subset C \ {a} such that sum(t' - a) = a - x, |T'| >= 1
        for a in C:
            others = [t for t in C if t != a]
            if not others:
                continue
            values = [t - a for t in others]
            target = a - x
            M_o = len(others)
            dp = [0] * (M_o + 1)
            dp[0] = 1 << offset
            
            for v in values:
                for k in range(M_o - 1, -1, -1):
                    if dp[k]:
                        if v >= 0:
                            dp[k+1] |= (dp[k] << v)
                        else:
                            dp[k+1] |= (dp[k] >> (-v))
            
            # Check if target is achievable with k >= 1
            target_bit = 1 << (offset + target)
            # Handle bounds
            if target_bit < 0 or target_bit.bit_length() > 50000: # Rough bound
                 # Actually just check if index is valid
                 pass
            
            for k in range(1, M_o + 1):
                if dp[k] & target_bit:
                    return False
        
        return True

    def greedy(order):
        C = []
        for x in order:
            if len(C) >= max_size:
                break
            if check_validity(C, x):
                C.append(x)
        return C

    # Try different orders
    best_set = []
    
    # Order 1: Ascending
    C1 = greedy(range(1, max_n + 1))
    if len(C1) > len(best_set):
        best_set = C1
        
    # Order 2: Descending
    C2 = greedy(range(max_n, 0, -1))
    if len(C2) > len(best_set):
        best_set = C2
        
    # Order 3: Random
    rand_order = list(range(1, max_n + 1))
    random.seed(42)
    random.shuffle(rand_order)
    C3 = greedy(rand_order)
    if len(C3) > len(best_set):
        best_set = C3
        
    return best_set
