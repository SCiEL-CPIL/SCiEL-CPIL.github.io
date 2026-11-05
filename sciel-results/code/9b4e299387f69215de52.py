"""
Find the smallest n in [1, 1000000] such that phi(n) is divisible by 120.
The minimum such n is 143, since phi(143) = phi(11*13) = 10*12 = 120.
Any n < 143 has phi(n) < 143, and the only multiple of 120 less than 143 is 120.
Verification shows no n < 143 has phi(n) = 120.
"""

# EVOLVE-BLOCK-START
def construct():
    return 143
# EVOLVE-BLOCK-END
