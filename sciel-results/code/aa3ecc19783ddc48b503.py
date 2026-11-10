# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def construct(q, n, radius):
    """Construct a covering code of length n over alphabet q with covering radius
    radius.  For the football-pool task this is (q, n, radius) = (3, 6, 1).

    Strategy:
      * Build an initial seed set using a product construction from a known
        perfect code on the first 4 coordinates (the ternary tetracode).
      * Greedily try to remove redundant codewords while preserving the covering
        property.
      * Greedily try to add a new codeword only if it covers at least 2 words
        not already covered by the current code (i.e., it is useful).
    """
    # --- helpers ---------------------------------------------------------------
    def encode(word, base=q):
        """Encode a tuple/list of symbols into a single integer."""
        v = 0
        for sym in word:
            v = v * base + sym
        return v

    def decode(v, base=q, length=n):
        """Decode an integer into a tuple of symbols (length n)."""
        out = []
        for _ in range(length):
            v, r = divmod(v, base)
            out.append(r)
        return tuple(reversed(out))

    # Precompute all words and their neighbours (within Hamming distance <= radius)
    total = q ** n
    # For each word, list of words within distance <= radius
    # We'll build this lazily or precompute if feasible
    # For q=3, n=6: total = 729, neighbourhood size is 1 + n*(q-1) + C(n,2)*(q-1)^2 ...
    # For radius=1: 1 + 6*2 = 13 neighbours
    # Precompute neighbour lists
    neighbours = [None] * total
    for w in range(total):
        word = decode(w)
        neigh = set([w])
        if radius >= 1:
            for i in range(n):
                orig = word[i]
                for sym in range(q):
                    if sym != orig:
                        new_word = list(word)
                        new_word[i] = sym
                        neigh.add(encode(new_word))
        if radius >= 2:
            for i in range(n):
                for j in range(i + 1, n):
                    orig_i = word[i]
                    orig_j = word[j]
                    for sym_i in range(q):
                        for sym_j in range(q):
                            if sym_i == orig_i and sym_j == orig_j:
                                continue
                            new_word = list(word)
                            new_word[i] = sym_i
                            new_word[j] = sym_j
                            neigh.add(encode(new_word))
        neighbours[w] = list(neigh)

    # Build reverse mapping: for each word, which codeword positions cover it
    # We'll maintain a coverage count array
    covered_by = [0] * total  # count of codewords covering each word

    def add_codeword(cw):
        """Add a codeword (list of ints) and update coverage."""
        cw_int = encode(cw)
        for nb in neighbours[cw_int]:
            covered_by[nb] += 1

    def remove_codeword(cw):
        """Remove a codeword (list of ints) and update coverage."""
        cw_int = encode(cw)
        for nb in neighbours[cw_int]:
            covered_by[nb] -= 1

    def is_covered():
        """Check if all words are covered."""
        return all(c > 0 for c in covered_by)

    # --- initial seed: tetracode product --------------------------------------
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]

    import itertools
    tails = list(itertools.product(range(q), repeat=n - 4)) if n > 4 else [[]]

    codewords = [h + list(tail) for h in tetracode for tail in tails]

    # Initialize coverage
    for cw in codewords:
        add_codeword(cw)

    # --- greedy removal of redundant codewords --------------------------------
    # Try to remove each codeword; if after removal all words are still covered,
    # keep it removed.
    for idx in range(len(codewords)):
        cw = codewords[idx]
        remove_codeword(cw)
        if is_covered():
            # Keep it removed
            codewords.pop(idx)
            idx -= 1
        else:
            add_codeword(cw)

    # --- greedy addition: try to find a codeword that covers many uncovered-ish words
    # Actually, since we already have a full cover, we can try to replace multiple
    # codewords with one that covers their union. This is complex, so instead:
    # Try to add a codeword and then see if we can remove more.

    # Better approach: local search / simulated annealing style
    # Try random perturbations: remove a random codeword, add a random one,
    # if coverage is maintained and size decreased or same, accept.

    import random
    random.seed(42)

    best_code = codewords[:]
    best_size = len(best_code)

    # Try a few iterations of: remove one, add one
    for iteration in range(200):
        if not is_covered():
            break
        # Try removing a random codeword
        idx = random.randint(0, len(codewords) - 1)
        cw_removed = codewords[idx]
        remove_codeword(cw_removed)
        codewords.pop(idx)

        if is_covered():
            # We can keep it removed
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_code = codewords[:]
            continue

        # Try to add a new codeword that covers the uncovered words
        # Find words that are now uncovered
        uncovered = [w for w in range(total) if covered_by[w] == 0]
        if not uncovered:
            add_codeword(cw_removed)
            codewords.insert(idx, cw_removed)
            continue

        # Try to find a codeword that covers many uncovered words
        # Sample some random positions
        best_gain = 0
        best_cw = None
        for _ in range(50):
            cand = [random.randint(0, q - 1) for _ in range(n)]
            cand_int = encode(cand)
            # Count how many uncovered words it covers
            gain = sum(1 for nb in neighbours[cand_int] if covered_by[nb] == 0)
            if gain > best_gain:
                best_gain = gain
                best_cw = cand

        if best_cw is not None and best_gain > 0:
            add_codeword(best_cw)
            codewords.append(best_cw)
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_code = codewords[:]
        else:
            # Revert
            add_codeword(cw_removed)
            codewords.insert(idx, cw_removed)

    # Return the best found
    return best_code
# EVOLVE-BLOCK-END
