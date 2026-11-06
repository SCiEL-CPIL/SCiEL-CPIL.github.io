"""
Program to find the smallest n in [1, 1000000] with phi(n) divisible by 120.
Based on analysis, the minimum such n is 143.
phi(143) = phi(11 * 13) = phi(11) * phi(13) = 10 * 12 = 120.
120 is divisible by 120.
No smaller n satisfies the condition.
"""
# EVOLVE-BLOCK-START

def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    The minimum such n is 143.
    """
    return 143

# EVOLVE-BLOCK-END
