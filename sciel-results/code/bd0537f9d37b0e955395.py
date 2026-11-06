import math
import time

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255,
    with gcd(a, b) > 1 for all distinct pairs, maximizing cardinality.
    
    The maximum size is achieved by taking all multiples of the smallest prime factor of 255.
    255 = 3 * 5 * 17. Smallest prime factor is 3.
    The set of multiples of 3 in [1, 255] has size floor(255/3) = 85.
    Any two distinct multiples of 3 have gcd at least 3.
    """
    start_time = time.time()
    deadline = start_time + 60.0  # 60s limit, keep margin
    
    max_val = 255
    # The smallest prime factor of 255 is 3.
    # The set of all multiples of 3 in [1, 255] is a valid solution.
    # Size is 85.
    # This is proven to be the maximum possible size for this problem.
    
    # Generate the set of multiples of 3
    # range(3, 256, 3) generates 3, 6, ..., 255
    candidate_set = set(range(3, max_val + 1, 3))
    
    # Ensure 255 is in the set (it is, since 255 is divisible by 3)
    # 255 % 3 == 0
    
    # Convert to list
    result = list(candidate_set)
    
    # Safety check: ensure 255 is present
    if 255 not in result:
        result.append(255)
        
    # Time check (should be instantaneous)
    if time.time() > deadline:
        # Fallback: return a small valid set if somehow we hit the limit (unlikely)
        return [255]
        
    return result
