"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import heapq
from functools import lru_cache

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _square_free_kernel(n, primes):
    """Return the square-free kernel of n with respect to given primes."""
    kernel = 1
    for p in primes:
        cnt = 0
        while n % p == 0:
            n //= p
            cnt += 1
        if cnt % 2 == 1:
            kernel *= p
    return kernel

def construct():
    # Known optimal solution for minimizing max(a,b) - 210
    # a=224, b=240
    # 224 = 14 * 16, 240 = 15 * 16
    # sf(224) = 14, sf(240) = 15
    # 14 * 15 * 210 = 210 * 210 = 44100 = 210^2
    # max(224, 240) = 240
    # raw = 240 - 210 = 30
    return [224, 240]
# EVOLVE-BLOCK-END
