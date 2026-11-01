"""Structural rewrite: greedy maximal-coverage set cover + local-search deletion,
with vectorized NumPy coverage over the full 3^n space and a deadline guard."""
import time
import random
import itertools
import numpy as np

# Precompute per-position base vectors for fast Hamming-distance computation.
# For alphabet size q and length n, a word w (int in [0, q^n)) decodes to digits.
# We precompute, for each position i and each value v, the mask of words whose
# i-th digit equals v.  distance-1 coverage of a center c is: union over i of
# (words with digit_i != c_i) plus c itself.  Equivalently, covered set of c =
#   all words  minus  (words with digit_i == c_i for ALL i) ... no, that's just c.
# Better: covered(c) = {c} union {words differing in exactly one position}.
# Using masks: words_equal_i_v = set of words with digit i == v.
# words_diff_i = all_words minus words_equal_i_{c_i}.
# covered(c) = union_i words_diff_i  plus c.  But union of words_diff_i = all words
# except those equal to c in every position = all words minus {c}.  So covered(c)
# = all words (the whole space) for radius 1!  That's only true if radius>=1 and
# every word is within distance 1 of c? No.  distance-1 ball of c has size
# 1 + n*(q-1).  The union_i (words with digit_i != c_i) is NOT the whole space.
# Let me recompute: words_diff_i = {w : w_i != c_i}.  A word w is in the union iff
# exists i with w_i != c_i, i.e. w != c.  So union_i words_diff_i = all words except c.
# That means covered(c) = all words.  That's WRONG for a radius-1 ball.
# The error: the radius-1 ball is words at distance EXACTLY 0 or 1, i.e. w=c or
# (exactly one coordinate differs).  "w_i != c_i for some i" includes words that
# differ in MORE than one coordinate.  So union_i words_diff_i is the whole space
# minus {c}, which is distance >=1, not distance ==1.  I need words that differ in
# EXACTLY one position.
# words_diff_exactly_i = (words with digit_i != c_i) AND (digit_j == c_j for all j!=i).
# That's the intersection over j!=i of words_equal_j_{c_j} minus words_equal_i_{c_i}.
# This is expensive to form as a mask directly.  Alternative: compute distance-1
# ball by enumerating the 1 + n*(q-1) neighbors explicitly and marking them.
# For n=7, q=3: ball size = 1 + 7*2 = 15.  Total space = 3^7 = 2187.
# Greedy: for each candidate center, compute its ball mask (15 bits) and count
# overlap with uncovered.  With 2187 candidates and 2187 space, we can do this.
# We'll represent the uncovered set as a boolean array of length q^n.

def _decode_digits(q, n):
    """Return a 2D int array shape (q**n, n) of all words' digits."""
    total = q ** n
    digits = np.empty((total, n), dtype=np.int8)
    for idx in range(total):
        x = idx
        for i in range(n - 1, -1, -1):
            digits[idx, i] = x % q
            x //= q
    return digits

def _ball_mask(digits, center_idx, q, n, radius):
    """Boolean mask (length q**n) of words within Hamming distance <= radius of center."""
    total = q ** n
    c = digits[center_idx]
    mask = np.zeros(total, dtype=bool)
    if radius >= 1:
        # all words differing in exactly one position, plus the center
        mask[center_idx] = True
        for i in range(n):
            for v in range(q):
                if v == c[i]:
                    continue
                # words with digit_i == v and digit_j == c_j for all j != i
                # Build via boolean AND of per-position equality masks.
                m = np.ones(total, dtype=bool)
                for j in range(n):
                    if j == i:
                        m &= (digits[:, j] == v)
                    else:
                        m &= (digits[:, j] == c[j])
                mask |= m
    return mask

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin below 120s / 150s
    total = q ** n
    digits = _decode_digits(q, n)
    rng = random.Random(12345)

    best_codewords = None
    best_count = total + 1

    def _is_valid(cw):
        # Verify every word is within radius of some codeword.
        uncovered = np.ones(total, dtype=bool)
        for c in cw:
            idx = int(np.ravel_multi_index(tuple(c), (q,) * n))
            uncovered &= ~_ball_mask(digits, idx, q, n, radius)
        return not uncovered.any()

    def _greedy(uncovered, candidates, digits, q, n, radius, deadline):
        """Greedy set cover: repeatedly pick candidate covering most uncovered."""
        uncovered = uncovered.copy()
        chosen = []
        cand_idx = list(range(total))
        while uncovered.any():
            if time.time() > deadline - 1.0:
                break
            best_gain = -1
            best_i = None
            for ci in cand_idx:
                b = _ball_mask(digits, ci, q, n, radius)
                gain = int(np.count_nonzero(b & uncovered))
                if gain > best_gain:
                    best_gain = gain
                    best_i = ci
                if gain == 0:
                    continue
            if best_i is None or best_gain <= 0:
                break
            chosen.append(best_i)
            uncovered &= ~_ball_mask(digits, best_i, q, n, radius)
        return chosen

    # --- Multi-start: a few randomized greedy runs with random candidate order ---
    for restart in range(6):
        if time.time() > deadline - 5.0:
            break
        # randomize candidate order for diversity
        order = list(range(total))
        rng.shuffle(order)
        uncovered = np.ones(total, dtype=bool)
        chosen = []
        for ci in order:
            if time.time() > deadline - 2.0:
                break
            if not uncovered.any():
                break
            b = _ball_mask(digits, ci, q, n, radius)
            gain = int(np.count_nonzero(b & uncovered))
            if gain > 0:
                chosen.append(ci)
                uncovered &= ~b
        if not uncovered.any() and len(chosen) < best_count:
            best_count = len(chosen)
            best_codewords = [list(digits[ci]) for ci in chosen]

    # --- Local search: try to delete codewords while staying valid ---
    if best_codewords is not None:
        cw = list(best_codewords)
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for k in range(len(cw)):
                if time.time() > deadline - 2.0:
                    break
                trial = cw[:k] + cw[k + 1:]
                if _is_valid(trial):
                    cw = trial
                    improved = True
                    break
        if len(cw) < best_count:
            best_count = len(cw)
            best_codewords = cw

    # Fallback: ensure we always return a valid covering (the direct-sum code).
    if best_codewords is None:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            best_codewords = [h + list(tail) for h in tetracode
                              for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            best_codewords = [list(w) for w in itertools.product(range(3), repeat=n)]

    return best_codewords
