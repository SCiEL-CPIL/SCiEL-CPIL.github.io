import time

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, 
    with gcd(a, b) > 1 for all distinct pairs. 
    Maximizes cardinality.
    """
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin before 60s limit
    
    N = 255
    # The smallest prime factor of 255 is 3.
    # The set of all multiples of 3 in [1, 255] contains 255.
    # Any two distinct multiples of 3 have gcd at least 3, which is > 1.
    # This set has size floor(255/3) = 85.
    # This is the maximum possible size for a non-coprime set containing 255.
    
    result = list(range(3, N + 1, 3))
    
    # Simple validity check (not strictly necessary for performance but good for safety)
    # if len(result) > 0 and N in result:
    #    pass
    
    return result
