import time

def construct():
    # Deadline guard for safety, though computation is O(N) and very fast
    deadline = time.time() + 85
    
    # 255 = 3 * 5 * 17
    # To maximize the subset size containing 255 with pairwise gcd > 1,
    # we should choose the set of multiples of the smallest prime factor of 255, which is 3.
    # The set of multiples of 3 in [1, 255] has size 255 // 3 = 85.
    # Any two distinct multiples of 3 have gcd at least 3.
    # This is the maximum possible cardinality for this constraint.
    
    result = list(range(3, 256, 3))
    
    # Verify 255 is in the set (it is, 255 % 3 == 0)
    # Verify pairwise gcd > 1 (all are multiples of 3)
    
    return result
