"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Analytical Solution:
    We require phi(n) % 120 == 0.
    For n < 240, phi(n) < 240. The only multiple of 120 in this range is 120.
    So we look for the smallest n such that phi(n) = 120.
    
    Checking candidates:
    n=143: 143 = 11 * 13. phi(143) = phi(11)*phi(13) = 10 * 12 = 120.
    120 is divisible by 120.
    
    Verification for n < 143:
    We checked all n < 143 and none have phi(n) = 120.
    Thus, 143 is the minimum.
    """
    return 143
# EVOLVE-BLOCK-END
