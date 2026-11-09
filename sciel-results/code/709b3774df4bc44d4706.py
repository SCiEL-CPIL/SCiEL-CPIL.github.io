import time
import math

def construct():
    """
    Return a nonempty subset of [1, 255] containing 255, with gcd(a,b)>1 for all distinct pairs.
    Maximize cardinality.
    
    The maximum cardinality is 85, achieved by the set of all multiples of 3.
    """
    start_time = time.time()
    deadline = start_time + 60.0  # 60 second limit
    
    # The prime factors of 255 are 3, 5, 17.
    # The smallest prime factor is 3.
    # The set of multiples of 3 is the largest pairwise non-coprime subset containing 255.
    
    # Generate all multiples of 3 in [1, 255]
    # range(start, stop, step)
    # 3, 6, ..., 255
    result = list(range(3, 256, 3))
    
    # Ensure 255 is included (it is, as 255 is divisible by 3)
    # Ensure non-empty (it is, size 85)
    
    # Optional: Verification step to ensure validity, kept short to respect deadline
    # This is a sanity check for the "model-and-optimize" framework
    if time.time() < deadline - 0.1:
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if math.gcd(result[i], result[j]) == 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            # Fallback to a smaller valid set if the main one fails (should never happen)
            # Multiples of 5
            result = list(range(5, 256, 5))
            
    return result
