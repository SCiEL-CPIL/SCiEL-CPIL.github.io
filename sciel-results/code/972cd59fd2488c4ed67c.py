import math
import time

def construct():
    # The problem asks for the largest subset S of [1, 255] containing 255
    # such that gcd(a, b) > 1 for all distinct a, b in S.
    
    # The prime factorization of 255 is 3 * 5 * 17.
    # The smallest prime factor is 3.
    # The set of all multiples of 3 in [1, 255] is a valid pairwise non-coprime set
    # because any two multiples of 3 share the factor 3.
    # It contains 255 (since 255 is divisible by 3).
    # The size of this set is floor(255 / 3) = 85.
    
    # It is a known result in combinatorial number theory that for an integer N,
    # the largest pairwise non-coprime subset of [1, N] containing N has size
    # floor(N / p), where p is the smallest prime factor of N.
    # Thus, 85 is the maximum possible cardinality.
    
    # We return this set.
    return list(range(3, 256, 3))
