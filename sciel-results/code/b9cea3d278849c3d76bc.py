"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    # The problem asks for a nonempty subset of [1, 255] containing 255,
    # with gcd(a, b) > 1 for all distinct pairs, maximizing cardinality.
    
    # 255 = 3 * 5 * 17.
    # Any element a in the set must satisfy gcd(a, 255) > 1, so a must be divisible by 3, 5, or 17.
    # Additionally, for any pair a, b in the set, gcd(a, b) > 1.
    
    # A sufficient condition for gcd(a, b) > 1 for all pairs is that all elements share a common prime factor p.
    # Since 255 must be in the set, p must divide 255.
    # The possible primes are 3, 5, 17.
    
    # If we choose p=3, the set is all multiples of 3 in [1, 255].
    # Size: floor(255 / 3) = 85.
    # If we choose p=5, the set is all multiples of 5 in [1, 255].
    # Size: floor(255 / 5) = 51.
    # If we choose p=17, the set is all multiples of 17 in [1, 255].
    # Size: floor(255 / 17) = 15.
    
    # Can we do better than 85 by mixing?
    # If we include any number not divisible by 3, it must be divisible by 5 or 17.
    # Let x be such a number. For any y in the set, gcd(x, y) > 1.
    # If the set contains any y not divisible by 5 (and assuming x is not divisible by other common primes),
    # we run into constraints that force us to remove many multiples of 3.
    # It is a known result that for n = p1*p2*... with p1 < p2 < ...,
    # the largest non-coprime subset containing n is the set of multiples of p1.
    
    # Therefore, the optimal strategy is to return all multiples of 3.
    # This set is valid because any two multiples of 3 have gcd >= 3 > 1.
    # It contains 255 because 255 is divisible by 3.
    
    result = [i for i in range(1, 256) if i % 3 == 0]
    return result
# EVOLVE-BLOCK-END
