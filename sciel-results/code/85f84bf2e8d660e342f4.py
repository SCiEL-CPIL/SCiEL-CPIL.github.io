import time
import math

def construct():
    """
    Constructs a nonempty subset of [1, 255] containing 255,
    with gcd(a, b) > 1 for all distinct pairs, maximizing cardinality.
    
    For n = 255 = 3 * 5 * 17, the largest pairwise non-coprime set
    containing 255 is the set of all multiples of the smallest prime factor, 3.
    Any two distinct multiples of 3 have a gcd of at least 3.
    Size of this set is floor(255 / 3) = 85.
    
    Proof of optimality: If a set S contains 255 and is pairwise non-coprime,
    then for any x in S, gcd(x, 255) > 1, so x is divisible by 3, 5, or 17.
    Moreover, if 3 in S, then every other element must be divisible by 3.
    Even without 3 in S, the structure forces the set to be a subset
    of multiples of some prime p in {3,5,17} to maintain high cardinality.
    The largest such set is the multiples of 3, giving 85 elements.
    Mixing primes (e.g., adding multiples of 5) requires removing many
    multiples of 3 that are coprime to 5, resulting in a net loss in size.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s kill
    
    n = 255
    smallest_prime = 3
    result = list(range(smallest_prime, n + 1, smallest_prime))
    
    # Quick validity check
    if 255 not in result:
        result.append(255)
    
    # Verify pairwise non-coprime (fast for 85 elements: ~3570 pairs)
    if time.time() < deadline:
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if math.gcd(result[i], result[j]) == 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            # Fallback: just return [255]
            return [n]
    
    return result
