"""Solves for the earliest run of 5 consecutive quadratic residues modulo 1009.
Uses precomputation at module load with bitset-based search for O(1) construct()."""

def _precompute():
    p = 1009
    m = 5
    
    # Precompute the smallest root for each non-zero quadratic residue modulo p.
    root_of = [0] * p
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and root_of[r] == 0:
            root_of[r] = x

    # Build a bitset where bit i is set if i is a non-zero quadratic residue.
    bitset = 0
    for i in range(1, p):
        if root_of[i] != 0:
            bitset |= (1 << i)

    # Find the smallest s in [1, p-m] such that bits s, s+1, ..., s+m-1 are all set.
    mask = (1 << m) - 1
    for s in range(1, p - m + 1):
        if ((bitset >> s) & mask) == mask:
            return {'start': s, 'roots': [root_of[s + i] for i in range(m)]}

    raise RuntimeError('No run found')


# Precompute once at module load
_CACHED_RESULT = _precompute()


def construct():
    return _CACHED_RESULT
