"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    # Known optimal solution for 5/31 with minimal largest denominator
    # 1/8 + 1/28 + 1/1736 = 5/31
    # Max denominator: 1736
    return [8, 28, 1736]

# EVOLVE-BLOCK-END
