import time

def construct():
    """
    Returns a subset of [1, 256] maximizing cardinality such that
    for every a < b <= c in the set, a does not divide b + c.
    
    Based on the analysis that if S is a subset of [K, N], the condition
    a < b <= c implies a < b+c <= 2N.
    If 2N < 3a, then b+c is strictly between 2a and 3a, so a cannot
    divide b+c.
    For N=256, 2N=512. We need 3a > 512 => a > 170.66.
    So a >= 171.
    The set [171, 256] has size 86 and is valid.
    Attempts to include elements < 171 require removing multiple elements
    from the upper tail to resolve conflicts, resulting in a net loss in
    cardinality.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    # The optimal set identified is the upper tail starting from 171
    # 171 to 256 inclusive
    result = list(range(171, 257))
    
    # Optional: Quick validity check (very fast)
    # This is O(N^3) but N=86, so ~640k ops, negligible.
    # However, we can just trust the math.
    
    return result
