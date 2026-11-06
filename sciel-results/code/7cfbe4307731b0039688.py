"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Analysis shows that 143 is the minimal such integer.
    phi(143) = phi(11*13) = 10*12 = 120, which is divisible by 120.
    No n < 143 satisfies the condition.
    """
    return 143
# EVOLVE-BLOCK-END
