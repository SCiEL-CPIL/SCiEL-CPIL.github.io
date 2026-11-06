"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    # The smallest integer n such that phi(n) is divisible by 120 is 143.
    # phi(143) = phi(11 * 13) = (11-1)*(13-1) = 10 * 12 = 120.
    # 120 is divisible by 120.
    # Any smaller n does not satisfy the condition.
    # Deadline guard: return immediately since this is a known constant.
    _start = time.time()
    _deadline = _start + 58.0  # safety margin before 60s limit
    
    n = 143
    return n
# EVOLVE-BLOCK-END
