"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Analysis:
    phi(n) must be a multiple of 120.
    Since phi(n) < n for n > 1, we must have n > 120.
    We check integers n starting from 121.
    
    n=121: phi(121) = 110
    n=122: phi(122) = 60
    ...
    n=143: phi(143) = phi(11*13) = 10*12 = 120. 120 % 120 == 0.
    
    No n < 143 has phi(n) divisible by 120.
    Therefore, 143 is the minimum.
    """
    return 143
# EVOLVE-BLOCK-END
