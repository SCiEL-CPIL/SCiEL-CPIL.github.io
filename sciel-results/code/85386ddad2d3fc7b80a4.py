"""Bitset-accelerated greedy covering C(24,7,3).

Each 3-subset of the v points is assigned an integer id.  The set of
currently-uncovered 3-subsets is kept as a Python int bitmask (bit i set <=>
subset i still uncovered).  For every candidate k-block we precompute the
bitmask of the t-subsets it covers.  A greedy set-cover step is then just a
bitwise AND + popcount per candidate, which is extremely fast and lets us
place as many blocks as the time budget allows.
"""
# EVOLVE-BLOCK-START
import itertools
import time


def _trunc(n, r):
    """Number of r-combinations of n elements (for id bookkeeping)."""
    if r < 0 or r > n:
        return 0
    r = min(r, n - r)
    res = 1
    for i in range(1, r + 1):
        res = res * (n - r + i) // i
    return res


def _combo_id(combo, n, r):
    """Lexicographic rank (0-based) of an r-combination of range(n)."""
    # combo must be a tuple/list of r sorted distinct ints in 0..n-1
    rank = 0
    for i in range(r):
        # number of combinations with i-th element smaller than combo[i]
        # while keeping the previous i elements fixed and the last r-1-i
        # elements free.
        lo = combo[i - 1] + 1 if i > 0 else 0
        # count choices for positions i..r-1 all > combo[i-1] and < combo[i]
        # = C(combo[i] - lo, r - i)
        rank += _trunc(combo[i] - lo, r - i)
    return rank


def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0  # keep a safety margin under the 120 s limit

    # --- build id table for all t-subsets -------------------------------
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    id_of = {}
    for idx, combo in enumerate(all_t):
        id_of[combo] = idx

    # --- precompute coverage bitmask for every candidate k-block ---------
    # For a k-block, the t-subsets it covers are all t-combinations of it.
    # We build the mask by OR-ing 1 << id_of(subset) over those subsets.
    masks = {}
    for block in itertools.combinations(range(v), k):
        m = 0
        for sub in itertools.combinations(block, t):
            m |= 1 << id_of[sub]
        masks[block] = m

    # --- greedy set cover using bitmasks --------------------------------
    uncovered = (1 << n_t) - 1  # all t-subsets uncovered
    blocks = []
    used = set()

    while uncovered:
        if time.time() >= deadline:
            break
        # find candidate with maximum marginal gain = popcount(uncovered & mask)
        best_gain = -1
        best_block = None
        for block, mask in masks.items():
            if block in used:
                continue
            gain = (uncovered & mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block = block
                if best_gain == n_t:  # cannot do better than covering everything
                    break
        if best_block is None or best_gain <= 0:
            break
        used.add(best_block)
        blocks.append(list(best_block))
        uncovered &= ~masks[best_block]

    # Fallback: if we ran out of time and still have uncovered subsets, fill
    # with a simple (slower) greedy to guarantee validity, but only if there
    # is remaining time.  Otherwise return what we have (may be invalid).
    if uncovered and time.time() < deadline - 5.0:
        remaining = [c for c in all_t if (uncovered >> id_of[c]) & 1]
        # quick fill: for each uncovered subset, add a block containing it
        for sub in remaining:
            if time.time() >= deadline - 2.0:
                break
            # find a k-block containing this t-subset with max new coverage
            best_gain = -1
            best_block = None
            for block in itertools.combinations(range(v), k):
                if set(sub).issubset(set(block)) and block not in used:
                    g = (uncovered & masks[block]).bit_count()
                    if g > best_gain:
                        best_gain = g
                        best_block = block
            if best_block is not None:
                used.add(best_block)
                blocks.append(list(best_block))
                uncovered &= ~masks[best_block]

    return blocks
# EVOLVE-BLOCK-END
