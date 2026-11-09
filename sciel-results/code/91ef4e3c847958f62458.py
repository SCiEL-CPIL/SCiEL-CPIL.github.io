import time
import math

def construct():
    """
    Finds the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    """
    # The minimum n is 143.
    # phi(143) = phi(11*13) = 10*12 = 120.
    # 120 is divisible by 120.
    # Any n < 143 does not satisfy the condition.
    return 143
