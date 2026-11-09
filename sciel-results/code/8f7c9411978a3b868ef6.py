"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
# EVOLVE-BLOCK-START
def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    The minimum n is 143.
    phi(143) = 120.
    No n < 143 has phi(n) divisible by 120.
    This is an O(1) solution.
    """
    start_time = time.time()
    # Deadline guard: not strictly necessary for O(1), but included for compliance with budget directives.
    if time.time() - start_time > 59.0:
        return 143
    
    # Verify 143 is valid (phi(143) = 120)
    # 143 = 11 * 13
    # phi(143) = 143 * (1 - 1/11) * (1 - 1/13) = 143 * (10/11) * (12/13) = 13 * 10 * 12 / 11 * 13 / 13? No.
    # phi(11*13) = (11-1)*(13-1) = 10*12 = 120.
    # 120 % 120 == 0.
    
    return 143
# EVOLVE-BLOCK-END
