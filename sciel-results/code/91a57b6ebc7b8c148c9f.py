import time
import math

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255,
    with gcd(a, b) > 1 for all distinct pairs.
    Maximizes cardinality.
    
    For n = 255 = 3 * 5 * 17, the largest pairwise non-coprime set
    containing 255 is the set of all multiples of the smallest prime factor, 3.
    Any two distinct multiples of 3 have a gcd of at least 3.
    Size of this set is floor(255 / 3) = 85.
    
    This is optimal because:
    - Every element in the set must share a factor with 255 (i.e., 3, 5, or 17)
    - Every pair must have gcd > 1
    - The set of all multiples of 3 has size 85
    - Any element not divisible by 3 must be divisible by 5 or 17
    - But such elements would need to share a factor with ALL multiples of 3 in the set
    - This forces a restriction that reduces the total size below 85
    """
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    n = 255
    smallest_prime = 3
    
    # Construct the set of all multiples of 3 in [1, 255]
    result = list(range(smallest_prime, n + 1, smallest_prime))
    
    # Verify: check that all pairs have gcd > 1
    # This is O(n^2) but n=85, so ~3600 checks, instant
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) <= 1:
                # Fallback: return just [255] if verification fails (shouldn't happen)
                return [255]
    
    # Check deadline (though this is instant)
    if time.time() > deadline:
        return [n]
        
    return result
