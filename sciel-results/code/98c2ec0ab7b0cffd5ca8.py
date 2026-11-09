"""Optimal per-residue minimum-cardinality covering of residues mod 101.

Rows are independent (denominators may recur across rows), so the max row
length equals max_r (min cardinality of a subset of {1..10} whose modular
inverse-sum is r). Enumerating subsets by increasing size yields that
provably-optimal per-residue minimum.
"""
# EVOLVE-BLOCK-START
from itertools import combinations

def construct():
    P = 101
    inv = [0] + [pow(x, -1, P) for x in range(1, 11)]
    best = [None] * P
    pool = list(range(1, 11))
    # Fill residues by increasing subset size -> per-residue minimum cardinality.
    for size in range(1, 11):
        for combo in combinations(pool, size):
            s = 0
            for x in combo:
                s += inv[x]
            s %= P
            if best[s] is None:
                best[s] = list(combo)
        # Early exit once every residue has a (minimum) representation.
        if all(b is not None for b in best):
            break
    # Safety fallback (should not trigger): full set covers a residue.
    for r in range(P):
        if best[r] is None:
            best[r] = list(pool)
    return [best[r] for r in range(P)]
# EVOLVE-BLOCK-END
