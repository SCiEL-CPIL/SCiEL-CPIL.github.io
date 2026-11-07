"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    # 210 = 2 * 3 * 5 * 7
    # For a single number n: 210 * n must be a perfect square
    # n must be 210 * s^2 for integer s
    # n > 210 requires s >= 2, so n >= 210 * 4 = 840
    # Minimize n - 210, so use s = 2, n = 840
    n = 210 * 4  # = 840
    # Verify: 210 * 840 = 176400 = 420^2
    assert 210 * n == 420 * 420
    assert n > 210
    assert n <= 10210
    return [n]
# EVOLVE-BLOCK-END
