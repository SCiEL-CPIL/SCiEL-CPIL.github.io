"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

_START = time.time()
_DEADLINE = 80.0  # safety margin under the 90s kill limit


def _elapsed():
    return time.time() - _START


def construct():
    MOD = 101
    N = 10

    # Precompute inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, -1, MOD)

    # Precompute subset sums for all subsets, grouped by size
    # subset_sums[size] = list of (residue, terms) for all subsets of that size
    subset_sums = [[] for _ in range(N + 1)]

    for size in range(1, N + 1):
        for terms in combinations(range(1, N + 1), size):
            residue = sum(inv[x] for x in terms) % MOD
            subset_sums[size].append((residue, list(terms)))

    # Greedy: assign smallest subsets first
    best = [None] * MOD  # best[r] = terms for residue r
    max_size_used = 0

    for size in range(1, N + 1):
        if _elapsed() > _DEADLINE:
            break
        for residue, terms in subset_sums[size]:
            if best[residue] is None:
                best[residue] = terms
                max_size_used = size

        # Check if all covered
        if all(b is not None for b in best):
            break

    # If deadline hit and not all covered, try to fill remaining with largest available
    if _elapsed() > _DEADLINE or any(b is None for b in best):
        # Fill any uncovered residues with any subset
        for size in range(1, N + 1):
            if _elapsed() > _DEADLINE:
                break
            for residue, terms in subset_sums[size]:
                if best[residue] is None:
                    best[residue] = terms
                    if size > max_size_used:
                        max_size_used = size

    return [best[r] for r in range(MOD)]
# EVOLVE-BLOCK-END
