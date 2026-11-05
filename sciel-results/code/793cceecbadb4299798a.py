"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, 
    with gcd(a,b)>1 for all distinct pairs. Maximize cardinality.
    
    Strategy:
    255 = 3 * 5 * 17.
    Any element a in the set must share a factor with 255.
    The largest set of pairwise non-coprime numbers containing 255 
    is typically the set of all multiples of the smallest prime factor of 255, 
    which is 3.
    
    Set S = {x in [1, 255] | x % 3 == 0}.
    Size = floor(255/3) = 85.
    All elements are divisible by 3, so gcd(a,b) >= 3 > 1.
    255 is in S.
    
    This is significantly better than multiples of 5 (size 51) or 17 (size 15).
    """
    # Explicit deadline guard (though this solution is instantaneous)
    start_time = time.time()
    deadline = 60.0 # seconds
    
    # Generate multiples of 3 up to 255
    # range(start, stop, step)
    # 3, 6, ..., 255
    candidates = list(range(3, 256, 3))
    
    # Verify 255 is in the set
    if 255 not in candidates:
        candidates.append(255)
        candidates.sort()
        
    # Verify pairwise gcd > 1
    # Since all are multiples of 3, this is guaranteed.
    # But we can do a quick sanity check if time permits (it will)
    if time.time() - start_time < deadline - 0.1:
        import math
        valid = True
        n = len(candidates)
        for i in range(n):
            for j in range(i+1, n):
                if math.gcd(candidates[i], candidates[j]) == 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            # Fallback to a known valid smaller set if logic fails (shouldn't happen)
            return [255]
            
    return candidates
# EVOLVE-BLOCK-END
