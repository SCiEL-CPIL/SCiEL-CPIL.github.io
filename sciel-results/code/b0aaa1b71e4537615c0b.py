import math
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the largest subset of [1, 255] containing 255 such that 
    gcd(a, b) > 1 for all distinct pairs a, b in the subset.
    
    The optimal set is the set of all multiples of the smallest prime factor of 255.
    255 = 3 * 5 * 17. Smallest prime factor is 3.
    The set is {3, 6, ..., 255}.
    Size = 255 // 3 = 85.
    """
    start_time = time.time()
    deadline = start_time + 85  # 85 seconds safety margin (limit is 90s)
    
    # The set of multiples of 3 in [1, 255]
    # This ensures any pair (a, b) has gcd >= 3 > 1.
    # It contains 255.
    result = [x for x in range(1, 256) if x % 3 == 0]
    
    # Optional: Verification (very fast)
    # for i in range(len(result)):
    #     for j in range(i+1, len(result)):
    #         if math.gcd(result[i], result[j]) == 1:
    #             raise ValueError("Invalid set found")
    
    return result
# EVOLVE-BLOCK-END
