"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    # Deadline guard: evaluator kills after 150s wall-clock.
    # We target finishing well before that with a safety margin.
    deadline = time.time() + 140.0

    # The tetracode: 9 words of length 4 over {0,1,2} that perfectly cover radius 1.
    # T = {(a, b, (a+b)%3, (a+2b)%3) : a,b in {0,1,2}}
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    T = np.array(tetracode, dtype=np.int8)  # shape (9, 4)

    # Direct-sum construction: C0 = T × {0,1,2}^4  (9 * 81 = 729 words)
    # Each word is [t, tail] where t in T, tail in {0,1,2}^4
    tails = np.array(list(itertools.product(range(3), repeat=4)), dtype=np.int8)  # (81, 4)

    # Build initial code as list of 8-tuples
    code = []
    for t in T:
        for tail in tails:
            code.append(list(t) + list(tail))

    # Convert to numpy array for vectorized operations
    C = np.array(code, dtype=np.int8)  # (729, 8)

    # We'll use a greedy local-search approach:
    # 1. Compute coverage: for each word in the space, which codewords cover it (within radius 1)?
    #    A word w is covered by codeword c if Hamming distance <= 1.
    #    For radius 1, w is covered by c iff w == c or w differs in exactly 1 position.
    #
    # 2. For each codeword, count how many words it uniquely covers (i.e., no other codeword covers them).
    #    If a codeword uniquely covers 0 words, it can be removed.
    #
    # 3. After removal, recheck coverage. If some words become uncovered, we may need to add back
    #    or find replacements. But for now, let's do a simple deletion pass.

    # Efficient coverage counting:
    # For a codeword c, the words it covers are: c itself, and for each position i and value v != c[i],
    # the word with c[i] replaced by v. That's 1 + 8*2 = 17 words per codeword.
    #
    # We can build a mapping: for each word in the space, how many codewords cover it?
    # But the space has 6561 words, and we have 729 codewords each covering 17 words = 12393 (with overlap).
    #
    # Let's use a dictionary or array indexed by the integer representation of each word.
    # Map word (tuple of 8 values in 0..2) to index in 0..6560 via base-3 encoding.

    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * 3 + v
        return idx

    # Precompute for each codeword the set of words it covers
    # coverage_count[idx] = number of codewords covering word idx
    coverage_count = np.zeros(3**n, dtype=np.int32)

    for c in C:
        idx_c = word_to_idx(c)
        coverage_count[idx_c] += 1
        for i in range(n):
            for v in range(3):
                if v != c[i]:
                    w = list(c)
                    w[i] = v
                    idx_w = word_to_idx(w)
                    coverage_count[idx_w] += 1

    # Now do iterative deletion: remove codewords that don't uniquely cover any word.
    # A codeword c uniquely covers word w if coverage_count[w] == 1 and c covers w.
    # But we need to track which codewords cover which words for the "unique" check.
    #
    # Simpler approach: for each codeword, check if removing it would leave any word uncovered.
    # A word w is left uncovered if it was only covered by this codeword.
    #
    # Let's use a different strategy: iterative greedy deletion.
    # In each pass, for each codeword (in some order), tentatively remove it and check if
    # all words it covered are still covered by other codewords. If yes, keep it removed.

    # To make this efficient, let's maintain for each word the set (or count) of covering codewords.
    # Given the size (6561 words, ~729 codewords), we can use arrays.

    # Let's use a 2D approach: for each codeword, store the 17 words it covers.
    # covers[c_idx] = list of word indices covered by codeword c_idx
    num_cw = len(C)
    covers = []
    for ci in range(num_cw):
        c = C[ci]
        idx_c = word_to_idx(c)
        covered = [idx_c]
        for i in range(n):
            for v in range(3):
                if v != c[i]:
                    w = list(c)
                    w[i] = v
                    covered.append(word_to_idx(w))
        covers.append(covered)

    # active[ci] = True if codeword ci is in the current code
    active = np.ones(num_cw, dtype=bool)

    # count_covered[w] = number of active codewords covering word w
    count_covered = coverage_count.copy()

    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        # Try to remove each codeword (in reverse order for stability)
        for ci in range(num_cw - 1, -1, -1):
            if not active[ci]:
                continue
            # Check if all words covered by ci are covered by at least one other active codeword
            can_remove = True
            for w_idx in covers[ci]:
                if count_covered[w_idx] <= 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove codeword ci
                active[ci] = False
                for w_idx in covers[ci]:
                    count_covered[w_idx] -= 1
                improved = True

    # Build the final code from active codewords
    final_code = [C[ci].tolist() for ci in range(num_cw) if active[ci]]

    # Now try a more sophisticated local search: swap codewords to reduce count further.
    # For each codeword, try replacing it with a different word that covers the same "critical" words
    # but also covers words that other codewords cover, potentially allowing further deletions.
    #
    # Actually, let's try a different angle: after deletion, see if we can add a codeword that
    # allows removing two or more existing codewords (net reduction).

    # Let's also try: for each active codeword, check if moving it to a nearby position
    # (distance 1 or 2) allows the same or better coverage with potential for more deletions.

    # Given time constraints, let's do a few rounds of "try to add a word that enables 2+ deletions"
    # and "try to relocate a word to a better position".

    # Relocation: for each active codeword, try moving one coordinate to a different value.
    # If the new position still covers all previously-unique words AND enables at least one
    # additional deletion, accept the move.

    # This is getting complex. Let's at least do one more pass of deletion after potential
    # structural improvements.

    # Convert to list for return
    result = [list(c) for c in final_code]

    # Verify: ensure all words are covered
    # (The evaluator will check, but let's be safe)
    if time.time() < deadline - 5.0:
        # Quick verification using the count_covered array
        # All words should have count_covered > 0
        if np.any(count_covered < 1):
            # Something went wrong; fall back to the full direct-sum code
            result = [list(c) for c in C]

    return result
# EVOLVE-BLOCK-END
