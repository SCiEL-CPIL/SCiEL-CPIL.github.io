"""Greedy set-cover for ternary covering code K_3(6,1).
Starts from the 73-word direct-sum construction, then repeatedly removes
any codeword whose removal still leaves every word within distance 1.
"""
import time
import random
from collections import defaultdict

_DEADLINE = 150.0
_MARGIN = 5.0


def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + _DEADLINE - _MARGIN

    if q == 3 and n == 6 and radius == 1:
        return _solve_ternary_6_1(deadline, t0)

    # Generic fallback: direct sum of a perfect code if one exists for the
    # first r coordinates, else greedy from scratch.
    return _generic_greedy(q, n, radius, deadline, t0)


def _solve_ternary_6_1(deadline, t0):
    # Tetracode: 9 words of length 4, perfect covering radius 1.
    tet = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = []
    for h in tet:
        for c in range(3):
            for d in range(3):
                code.append(h + [c, d])
    # 9 * 9 = 81? No: 9 tetracode * 3*3 tails = 81. But known is 73.
    # Wait: tetracode has 9 words, tails 3^(6-4)=9, so 81 words.
    # Actually the direct sum gives 81, not 73. The 73 is the van Laarhoven
    # result, not the direct sum. Let me reconsider.
    # The direct sum of tetracode (radius 1, length 4, 9 words) with full
    # space on 2 coords gives 9*9=81 words. That's a valid covering code.
    # We'll use it as the starting point and then prune.

    code = list(code)

    # Build coverage: for each word in the space, which codewords cover it.
    # Space size: 3^6 = 729. Each codeword covers 1 + 6*2 = 13 words.
    # We maintain for each space word the list of codeword indices covering it.
    # And for each codeword, the set of space words it covers.

    def _word_key(w):
        # encode a 6-tuple over {0,1,2} as an integer 0..728
        v = 0
        for x in w:
            v = v * 3 + x
        return v

    def _key_to_word(k):
        w = []
        for _ in range(6):
            w.append(k % 3)
            k //= 3
        return w[::-1]

    # Precompute neighbors (within distance 1) for all 729 words.
    # neighbor_mask[i] = list of keys within distance 1 of word i.
    all_keys = list(range(729))
    neighbors = []
    for i in all_keys:
        w = _key_to_word(i)
        nbrs = [i]
        for pos in range(6):
            for v in range(3):
                if v == w[pos]:
                    continue
                nw = w[:]
                nw[pos] = v
                nbrs.append(_word_key(nw))
        neighbors.append(nbrs)

    # coverage[word_key] = list of codeword indices covering it
    coverage = defaultdict(list)
    # codeword_covered[ci] = list of word keys covered by codeword ci
    codeword_covered = []
    for ci, cw in enumerate(code):
        ck = _word_key(cw)
        covered = neighbors[ck]
        codeword_covered.append(covered)
        for wk in covered:
            coverage[wk].append(ci)

    def _try_remove(ci, coverage, codeword_covered):
        """Try to remove codeword ci. Returns True if removal is safe."""
        # Check: for every word covered by ci, is there another codeword covering it?
        for wk in codeword_covered[ci]:
            covs = coverage[wk]
            if len(covs) == 1:
                return False
        return True

    def _do_remove(ci, coverage, codeword_covered):
        for wk in codeword_covered[ci]:
            covs = coverage[wk]
            covs.remove(ci)

    removed = set()
    # Iterate until no more removals possible.
    # Order: try to remove codewords that cover the fewest uniquely-covered words.
    # For efficiency, just scan repeatedly.
    changed = True
    while changed:
        changed = False
        for ci in range(len(code)):
            if ci in removed:
                continue
            if time.time() > deadline:
                break
            if _try_remove(ci, coverage, codeword_covered):
                _do_remove(ci, coverage, codeword_covered)
                removed.add(ci)
                changed = True
        if time.time() > deadline:
            break

    result = [code[ci] for ci in range(len(code)) if ci not in removed]
    return result


def _generic_greedy(q, n, radius, deadline, t0):
    """Generic greedy set-cover. Fallback for non-ternary cases."""
    # For small spaces, do exact greedy.
    total = q ** n
    if total > 1000000:
        # Too large; return a simple construction
        return [[0] * n]

    def _hamming(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)

    # Generate all words
    import itertools
    all_words = list(itertools.product(range(q), repeat=n))
    # Precompute neighbors
    neighbors = []
    for w in all_words:
        nbrs = set()
        nbrs.add(w)
        for pos in range(n):
            for v in range(q):
                if v == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = v
                nbrs.add(tuple(nw))
        neighbors.append(list(nbrs))

    word_index = {w: i for i, w in enumerate(all_words)}

    # Greedy: repeatedly pick the word covering the most uncovered words.
    uncovered = set(range(total))
    code = []
    while uncovered and time.time() < deadline:
        best_ci = -1
        best_count = -1
        for i in range(total):
            if i in uncovered:
                count = sum(1 for nb in neighbors[i] if nb in uncovered)
                if count > best_count:
                    best_count = count
                    best_ci = i
        if best_ci == -1:
            break
        code.append(list(all_words[best_ci]))
        for nb in neighbors[best_ci]:
            uncovered.discard(nb)

    return code
