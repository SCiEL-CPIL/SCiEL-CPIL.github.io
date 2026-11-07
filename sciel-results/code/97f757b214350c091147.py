"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _is_latin(sq, n):
    for i in range(n):
        if sorted(sq[i]) != list(range(n)):
            return False
    for j in range(n):
        col = [sq[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True

def _are_orthogonal(sq1, sq2, n):
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (sq1[i][j], sq2[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return True

def _build_square(c2, c5, n=10):
    """Build Latin square using Z_2 x Z_5 linear construction."""
    sq = []
    for i in range(n):
        i2 = i // 5
        i5 = i % 5
        row = []
        for j in range(n):
            j2 = j // 5
            j5 = j % 5
            a_out = (c2 * i2 + j2) % 2
            b_out = (c5 * i5 + j5) % 5
            row.append(a_out * 5 + b_out)
        sq.append(row)
    return sq

def construct():
    n = 10
    start = time.time()
    deadline = start + 50.0  # safety margin before 60s limit

    # Generate all candidate squares from Z_2 x Z_5 linear construction
    candidates = []
    for c2 in range(2):
        for c5 in range(5):
            sq = _build_square(c2, c5, n)
            if _is_latin(sq, n):
                candidates.append((c2, c5, sq))

    # Greedily find maximum set of mutually orthogonal squares
    # Sort candidates to prefer diverse parameters
    best_set = []
    for cand in candidates:
        c2, c5, sq = cand
        ok = True
        for _, _, sq2 in best_set:
            if not _are_orthogonal(sq, sq2, n):
                ok = False
                break
        if ok:
            best_set.append(cand)

    if best_set:
        result = [sq for _, _, sq in best_set]
    else:
        # Fallback: single trivial Latin square
        result = [[(i + j) % n for j in range(n)] for i in range(n)]

    # Ensure proper types: all values are Python int, all sequences are lists
    result = [[list(row) for row in sq] for sq in result]
    for sq in result:
        for row in sq:
            for k in range(len(row)):
                row[k] = int(row[k])

    return result
# EVOLVE-BLOCK-END
