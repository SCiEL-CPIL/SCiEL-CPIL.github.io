import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    n = 255
    # 255 = 3 * 5 * 17
    # The largest pairwise non-coprime set containing 255 is the set of all
    # multiples of the smallest prime factor (3), giving 85 elements.
    # Proof: Any element x in a valid set must share a factor with 255, so x
    # is divisible by 3, 5, or 17. If we take all multiples of 3, every pair
    # has gcd >= 3. We cannot add any element not divisible by 3 because
    # gcd(x, 3) = 1 for any x not divisible by 3, and 3 is in our set.
    # Thus the maximum size is floor(255/3) = 85.
    
    # Check time, though this calculation is trivial
    if time.time() > deadline:
        # Fallback to a smaller valid set if somehow delayed (unlikely)
        return [255]
        
    return list(range(3, n + 1, 3))
