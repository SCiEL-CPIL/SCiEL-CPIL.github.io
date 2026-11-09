"""Optimal solver: minimize max row length by picking min-size subset per residue."""
from functools import lru_cache

def _precompute():
    # Compute residue and popcount for every non-empty subset of {1..10}
    inv = [pow(x, -1, 101) for x in range(11)]
    # For each residue, track the minimum popcount subset (as mask)
    best = {}  # residue -> (popcount, mask)
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        pc = len(terms)
        residue = sum(inv[x] for x in terms) % 101
        if residue not in best or pc < best[residue][0]:
            best[residue] = (pc, mask)
    # Build rows: for each residue, the terms of its best (min-size) subset
    rows = []
    for r in range(101):
        if r in best:
            pc, mask = best[r]
            terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
            rows.append(terms)
        else:
            rows.append([])
    return rows

# Precompute once at module load
_ROWS = _precompute()

def construct():
    return [list(row) for row in _ROWS]
