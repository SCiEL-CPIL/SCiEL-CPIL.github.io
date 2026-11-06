import numpy as np
from itertools import combinations
import time

def construct():
    """
    Find the smallest n (5 <= n <= 90) for which a tournament exists where every
    set of 4 vertices has a common dominator.
    """
    start_time = time.time()
    time_limit = 110  # Safety margin before 120s hard limit

    def is_paley_valid(p):
        """Check if Paley tournament on p vertices is valid."""
        if p < 5:
            return False

        # Build squares set
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)

        # Build in_mask: in_mask[v] has bit u set if u -> v
        in_mask = [0] * p
        for i in range(p):
            for j in range(p):
                if i == j:
                    continue
                diff = (j - i) % p
                if diff in squares:
                    # i -> j, so i is in in_mask[j]
                    in_mask[j] |= (1 << i)

        # Check all 4-subsets
        # Optimize: use local variables
        im = in_mask
        for a, b, c, d in combinations(range(p), 4):
            if (im[a] & im[b] & im[c] & im[d]) == 0:
                return False
        return True

    def make_paley_orientation(p):
        """Construct orientation bits for Paley tournament on p vertices."""
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)

        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                diff = (j - i) % p
                if diff in squares:
                    orientation.append(0)  # i -> j
                else:
                    orientation.append(1)  # j -> i
        return orientation

    # Strategy: Check Paley primes in ascending order
    # If any is valid, return it
    # If none valid, return 83 (known valid)
    candidates = [31, 43, 47, 59, 67]

    for p in candidates:
        if time.time() - start_time > time_limit - 15:
            break

        if is_paley_valid(p):
            orientation = make_paley_orientation(p)
            return {'n': p, 'orientation': orientation}

    # Fallback: Return the seed solution (83)
    p = 83
    orientation = make_paley_orientation(p)
    return {'n': p, 'orientation': orientation}
