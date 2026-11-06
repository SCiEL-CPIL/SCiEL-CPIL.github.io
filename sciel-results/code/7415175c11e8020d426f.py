"""Small tournaments in which every four vertices have a common dominator."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _paley_orientation(p):
    """Paley tournament: i -> j iff (j-i) mod p is a quadratic residue mod p."""
    n = p
    squares = set(x * x % p for x in range(1, p))
    # orientation is one bit per pair (i,j), i<j, in lexicographic order
    # 0 means i -> j, 1 means j -> i
    orient = np.empty(n * (n - 1) // 2, dtype=np.int8)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            # i -> j iff (j - i) % p is a square
            if (j - i) % p in squares:
                orient[idx] = 0  # i -> j
            else:
                orient[idx] = 1  # j -> i
            idx += 1
    return {'n': n, 'orientation': orient.tolist()}

def construct():
    # Paley tournament on 83 vertices (smallest known valid)
    p = 83
    return _paley_orientation(p)
# EVOLVE-BLOCK-END
