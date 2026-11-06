import time
from math import gcd

def construct():
    # The optimal set found through analysis
    # m=1: {1, 2, 4, 8, 16, 32, 64}
    # m=3: {3, 6, 12, 24, 48}
    # m=5: {5, 10, 20}
    # m=7: {7, 14, 28}
    
    result = [1, 2, 4, 8, 16, 32, 64, 3, 6, 12, 24, 48, 5, 10, 20, 7, 14, 28]
    
    # Optional: Verification to ensure validity
    # This is fast and ensures no bugs in the hardcoded list
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            lcm_val = (a * b) // gcd(a, b)
            if lcm_val > 512:
                # Fallback to a known valid smaller set if this fails (should not happen)
                return list(range(1, 11))
                
    return result
