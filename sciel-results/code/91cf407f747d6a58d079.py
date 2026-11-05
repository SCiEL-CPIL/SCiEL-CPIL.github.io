import time
from math import gcd

def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 55
    
    N = 255
    
    # The optimal strategy for this specific N=255 (3*5*17) is to take all multiples of the smallest prime factor, 3.
    # This yields a clique of size floor(255/3) = 85.
    # Any attempt to mix multiples of 5 or 17 will reduce the set size because elements must share factors with ALL other elements.
    # Specifically, if we include a multiple of 3 that is not a multiple of 5, we exclude all multiples of 5 that are not multiples of 3.
    # The set of all multiples of 3 is a valid clique because gcd(a,b) >= 3 for all a,b in the set.
    
    # Construct the set of multiples of 3 in [1, 255]
    # This set automatically contains 255 since 255 % 3 == 0.
    S = list(range(3, N + 1, 3))
    
    # Verify validity (optional but good for robustness)
    # Check if 255 is in S
    if 255 not in S:
        S.append(255)
        S.sort()
        
    # Check pairwise gcd > 1 (for the first few to ensure logic holds, full check is O(n^2) but n=85 is small)
    # Actually, we know it's valid by construction.
    # But let's ensure we don't return an empty set (though it won't be).
    
    if not S:
        return [255]
        
    return S
