"""Find the smallest n in [1, 1000000] with phi(n) divisible by 120."""

import math

def construct():
    # The smallest integer n such that phi(n) is divisible by 120 is 143.
    # phi(143) = phi(11 * 13) = (11-1) * (13-1) = 10 * 12 = 120.
    # 120 is divisible by 120.
    # No n < 143 has phi(n) divisible by 120.
    return 143
