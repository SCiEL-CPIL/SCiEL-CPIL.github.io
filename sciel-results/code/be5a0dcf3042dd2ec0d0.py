"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from collections import defaultdict

def construct():
    # We need the largest subset S of [1,255] containing 255 such that
    # gcd(a,b) > 1 for all distinct pairs in S.
    
    # 255 = 3 * 5 * 17
    # Any element in S must share a factor with 255, so must be divisible by 3, 5, or 17.
    # But more strongly, any two elements must share a common factor.
    
    # Key insight: The set of all multiples of the smallest prime factor (3) of 255
    # is always a valid solution: any two multiples of 3 have gcd >= 3 > 1.
    # Size = floor(255/3) = 85.
    
    # Can we do better by mixing numbers divisible by different primes?
    # If we include a number divisible by 5 but not 3, it must share a factor with
    # every multiple of 3 in the set. A multiple of 3 like 3*7=21 (not div by 5 or 17)
    # would be coprime to a number like 5*11=55 unless they share another factor.
    # So mixing is very restrictive.
    
    # The optimal answer for "largest pairwise non-coprime subset containing n"
    # where n has smallest prime factor p is floor(n/p), achieved by all multiples of p.
    # This is because the multiples of p form a clique in the "gcd>1" graph,
    # and you cannot add any element not divisible by p without removing many
    # multiples of p that are coprime to it.
    
    # For 255: smallest prime factor is 3, so answer is floor(255/3) = 85.
    result = list(range(3, 256, 3))
    
    # Verify: 255 is in the set (255 = 3*85, and 85 <= 85)
    assert 255 in result
    
    # Verify pairwise gcd > 1: any two elements are multiples of 3, so gcd >= 3
    return result
# EVOLVE-BLOCK-END
