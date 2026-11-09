"""
Solved by direct derivation:
We seek the smallest n such that phi(n) is divisible by 120.
120 = 2^3 * 3 * 5.
Analysis of phi(n) = 120 (since phi(n) < n for n > 2, and we check n < 143):
1. If n is odd, n must be a product of primes p where p-1 | 120.
   Primes: 2, 3, 5, 7, 11, 13, 31, 41, 61.
   Smallest odd n with phi(n)=120 is 11*13 = 143.
   (Checked combinations: 7*25=175, 5*31=155, 3*61=183, etc. all > 143).
2. If n is even, n=2m with m odd. phi(n)=phi(m).
   Smallest m is 143, so smallest even n is 286.
3. Thus 143 is the global minimum.
"""

def construct():
    return 143
