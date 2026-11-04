"""Football pool / ternary covering code K_3(8,1).
Strategy: start from the direct-sum construction (729 words), then iteratively
remove redundant codewords (a word is redundant if every point it covers is
covered by another word).  This greedy pruning is fast and keeps the code valid.
"""
# EVOLVE-BLOCK-START
import time
import random

# Precompute the ball of radius 1 around each codeword index once.
# A codeword c covers c itself and all words at Hamming distance 1.


def _neighbors(c, n):
    """Return list of indices (as ints) of c and all words at distance 1 from c."""
    idx = 0
    for i, v in enumerate(c):
        idx = idx * 3 + v
    out = [idx]
    for i in range(n):
        base = c[:i]
        rest = c[i + 1:]
        for v in range(3):
            if v == c[i]:
                continue
            word = base + [v] + rest
            j = 0
            for x in word:
                j = j * 3 + x
            out.append(j)
    return out


def construct(q, n, radius):
    start = time.time()
    deadline = start + 115.0  # safety margin before 120s kill

    if q != 3 or n != 8 or radius != 1:
        # Fallback: direct-sum construction generalized
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        if n >= 4:
            return [h + list(tail) for h in tetracode
                    for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            # Just return all words (brute force)
            import itertools
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Build initial code: direct sum of tetracode with all 3^4 tails = 9*81 = 729 words
    import itertools
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode
            for tail in itertools.product(range(3), repeat=4)]

    N = 3 ** n  # 6561

    # Map each codeword to its index
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * 3 + v
        return idx

    # For each codeword, store its covered set as a frozenset of point-indices
    # We'll maintain: cover_count[p] = number of codewords covering point p
    # and for each codeword i, the list of points it covers.

    cover_count = [0] * N
    covers = []  # covers[i] = list of point indices covered by code[i]
    for w in code:
        pts = _neighbors(w, n)
        covers.append(pts)
        for p in pts:
            cover_count[p] += 1

    # Greedy removal: repeatedly find a redundant codeword and remove it.
    # A codeword i is redundant if for every p in covers[i], cover_count[p] > 1
    #   (i.e., some other codeword also covers p).
    # To be efficient, we only need to check if there exists a p in covers[i]
    # with cover_count[p] == 1; if so, i is essential.

    active = [True] * len(code)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(code)):
            if not active[i]:
                continue
            # Check if codeword i is essential
            essential = False
            for p in covers[i]:
                if cover_count[p] == 1:
                    essential = True
                    break
            if not essential:
                # Remove codeword i
                active[i] = False
                for p in covers[i]:
                    cover_count[p] -= 1
                improved = True

    # Build result
    result = [code[i] for i in range(len(code)) if active[i]]
    return result
# EVOLVE-BLOCK-END
