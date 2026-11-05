import math

def construct():
    n = 255
    # The prime factors of 255 are 3, 5, 17.
    # To maximize the size of a subset S containing 255 with gcd(a,b)>1 for all pairs,
    # we should pick the multiples of the smallest prime factor of 255, which is 3.
    # This set is S = {3, 6, ..., 255}.
    # Size is floor(255/3) = 85.
    # Any other number x not in S is not divisible by 3.
    # Since 3 is in S, gcd(x, 3) = 1, so x cannot be added.
    # Thus, this is the maximum size set containing 3.
    # Sets not containing 3 are significantly smaller (e.g., using factor 5 gives 51, factor 2 with constraints gives 64).
    
    p = 3
    return list(range(p, n + 1, p))
