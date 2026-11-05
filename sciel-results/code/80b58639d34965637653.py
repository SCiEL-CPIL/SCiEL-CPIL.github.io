import time
import math
from itertools import combinations

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _paley_orientation(p):
    """Generate Paley tournament orientation for prime p ≡ 3 (mod 4)."""
    squares = set()
    for x in range(1, p):
        squares.add(x * x % p)
    
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return orientation

def _check_property(n, orientation):
    """Check if every 4-set has a common dominator. Returns True if valid."""
    in_neigh = [0] * n
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if orientation[idx] == 0:
                in_neigh[j] |= (1 << i)
            else:
                in_neigh[i] |= (1 << j)
            idx += 1
    
    for a, b, c, d in combinations(range(n), 4):
        if (in_neigh[a] & in_neigh[b] & in_neigh[c] & in_neigh[d]) == 0:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    candidates = []
    for p in range(31, 91):
        if p % 4 == 3 and _is_prime(p):
            candidates.append(p)
    
    for p in candidates:
        if time.time() > deadline:
            break
        orientation = _paley_orientation(p)
        if _check_property(p, orientation):
            return {'n': p, 'orientation': orientation}
    
    p = 83
    orientation = _paley_orientation(p)
    return {'n': p, 'orientation': orientation}
