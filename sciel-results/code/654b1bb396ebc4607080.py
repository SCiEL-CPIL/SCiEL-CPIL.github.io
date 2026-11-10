def construct():
    from math import gcd, lcm
    import numpy as np
    
    n = 512
    # Create compatibility matrix
    # compatible[i][j] = True if lcm(i+1, j+1) <= 512
    
    # For efficiency, compute pairwise
    # We want to find the largest subset where all pairs are compatible
    
    # Greedy approach: start with all, iteratively remove
    S = set(range(1, n+1))
    
    # Compute compatibility
    def is_compatible(a, b):
        return (a * b) // gcd(a, b) <= 512
    
    # Iterative removal
    changed = True
    while changed:
        changed = False
        # Find a violating pair
        lst = list(S)
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                if not is_compatible(lst[i], lst[j]):
                    # Remove the one with fewer compatible partners in S
                    # Count compatible partners for each
                    cnt_i = sum(1 for x in S if x != lst[i] and is_compatible(lst[i], x))
                    cnt_j = sum(1 for x in S if x != lst[j] and is_compatible(lst[j], x))
                    if cnt_i <= cnt_j:
                        S.discard(lst[i])
                    else:
                        S.discard(lst[j])
                    changed = True
                    break
            if changed:
                break
    
    return list(S)
