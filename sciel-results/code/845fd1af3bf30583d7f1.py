"""
Constructs an integer n <= 10^18 that maximizes the number of adjacent 
coprime pairs in its sorted divisor list.

Strategy:
The score tau_perp(n) is maximized by maximizing the number of distinct 
prime factors of n, as this increases the size of the divisor lattice 
(hypercube structure for square-free n) and the opportunities for 
disjoint prime supports to appear in adjacent sorted divisors.

The product of the first k primes, P_k, is the smallest square-free 
integer with k distinct prime factors. We choose the largest k such that 
P_k <= 10^18.

P_15 = 2 * 3 * 5 * ... * 47 = 614,889,782,588,491,410
P_16 = P_15 * 53 > 10^18

Thus, n = P_15 is the optimal candidate under the assumption that 
square-free numbers with maximum distinct prime factors yield the highest 
score.
"""

# EVOLVE-BLOCK-START
def construct():
    # Product of the first 15 primes
    # 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n = 1
    for p in primes:
        n *= p
    return n
# EVOLVE-BLOCK-END
