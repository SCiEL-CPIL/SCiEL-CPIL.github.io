import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 90  # seconds

    # 255 = 3 * 5 * 17
    # The largest pairwise non-coprime subset containing 255 is the set of multiples
    # of the smallest prime factor of 255, which is 3.
    # 
    # Let S = { x in [1, 255] : 3 divides x }.
    # Size of S = floor(255 / 3) = 85.
    # For any a, b in S, gcd(a, b) is at least 3, so gcd(a, b) > 1.
    # 255 is a multiple of 3, so 255 is in S.
    #
    # Proof of maximality:
    # Any element in a valid set must share a factor with 255, so it must be
    # divisible by 3, 5, or 17.
    # If the set contains 3, then every other element must be divisible by 3.
    # Thus S is a subset of multiples of 3, so |S| <= 85.
    # If the set does not contain 3, we can potentially include numbers not
    # divisible by 3. However, adding a number x not divisible by 3 (e.g., 5)
    # requires removing all elements y in the set such that gcd(x, y) == 1.
    # If we start with the full set of multiples of 3 (size 85) and try to add 5,
    # we must remove all multiples of 3 that are not multiples of 5.
    # Count(multiples of 3) = 85.
    # Count(multiples of 15) = floor(255/15) = 17.
    # Multiples of 3 not div by 5 = 85 - 17 = 68.
    # So adding 5 forces removal of 68 elements. Net change: +1 - 68 = -67.
    # Similar analysis for 17 shows even worse trade-offs.
    # Thus, the set of multiples of 3 is optimal.

    # Generate the set of multiples of 3 in [1, 255]
    S = list(range(3, 256, 3))
    
    # Ensure 255 is in the set (it is, but just in case)
    if 255 not in S:
        S.append(255)
        
    # Quick verification (optional but safe)
    # Check that all pairs have gcd > 1
    # This is O(N^2) which is 85^2 ~ 7000 operations, very fast.
    # We can skip this for performance, but it's negligible.
    # Let's just return the set.
    
    return S
