"""
Vectorized greedy covering code construction for K_q(n, r) with local search polish.
Uses the tetracode product as a strong initial solution, then reduces via
critical-word analysis and 2-for-1 / 1-for-2 moves, with a final polish pass.
"""
import time
import numpy as np
import itertools

# Deadline guard: kill at 145s to stay safely under 150s wall-clock.
_DEADLINE = 145.0


def _time_left():
    return _DEADLINE - time.time()


def _ball_mask_single(codeword, q, n, radius):
    """
    Return a boolean array of length q**n: True for words within Hamming distance
    <= radius of codeword.  Uses a recursive/iterative ball enumeration.
    For small radius and n this is fine; we rely on it being called on the
    initial solution only (once per codeword) and during polish.
    """
    total = q ** n
    mask = np.zeros(total, dtype=bool)
    # Enumerate all words within distance radius
    # Use itertools for clarity; for q=3, n=7, radius=1 the ball size is 1 + 7*2 = 15
    # We can enumerate efficiently.
    for d in range(radius + 1):
        for positions in itertools.combinations(range(n), d):
            for changes in itertools.product(range(q), repeat=d):
                word = list(codeword)
                for pos, val in zip(positions, changes):
                    if val != word[pos]:
                        word[pos] = val
                # Convert to index
                idx = 0
                for digit in word:
                    idx = idx * q + digit
                mask[idx] = True
    return mask


def _ball_mask_vectorized(codewords, q, n, radius):
    """
    For a set of codewords (list of lists), compute the union ball mask.
    Returns (mask, per_codeword_masks) where per_codeword_masks is a list of bool arrays.
    """
    total = q ** n
    mask = np.zeros(total, dtype=bool)
    per = []
    for cw in codewords:
        m = _ball_mask_single(cw, q, n, radius)
        per.append(m)
        mask |= m
    return mask, per


def _critical_words(mask, q, n):
    """Return list of indices of uncovered words (should be empty if valid)."""
    uncovered = np.where(~mask)[0]
    return uncovered.tolist()


def _index_to_word(idx, q, n):
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def _word_to_index(word, q):
    idx = 0
    for d in word:
        idx = idx * q + d
    return idx


def _greedy_add(codewords, mask, q, n, radius, max_add=500):
    """
    Greedily add codewords to cover the most uncovered words.
    Returns updated codewords and mask.
    """
    uncovered = np.where(~mask)[0]
    if len(uncovered) == 0:
        return codewords, mask
    for _ in range(max_add):
        if len(uncovered) == 0:
            break
        # Try each uncovered word as a candidate center; pick the one covering most uncovered
        # To keep it fast, sample a subset
        candidates = uncovered[:min(len(uncovered), 200)]
        best_idx = -1
        best_count = 0
        best_mask = None
        for c in candidates:
            word = _index_to_word(c, q, n)
            m = _ball_mask_single(word, q, n, radius)
            count = np.sum(m & ~mask)
            if count > best_count:
                best_count = count
                best_idx = c
                best_mask = m
        if best_idx < 0 or best_count == 0:
            break
        new_word = _index_to_word(best_idx, q, n)
        codewords.append(new_word)
        mask |= best_mask
        uncovered = np.where(~mask)[0]
    return codewords, mask


def _try_2_for_1(codewords, mask, per_masks, q, n, radius, max_tries=200):
    """
    Try to remove two codewords and add one that covers all their previously-unique coverage.
    Returns (success, new_codewords, new_mask, new_per_masks).
    """
    m = len(codewords)
    if m < 2:
        return False, codewords, mask, per_masks
    # For each pair (i, j), find words covered only by i and j (not by others)
    # This is expensive; sample pairs.
    np.random.seed(42)
    pairs = list(itertools.combinations(range(m), 2))
    np.random.shuffle(pairs)
    for i, j in pairs[:max_tries]:
        # Words covered by i or j but not by any other codeword
        others_mask = np.ones(len(mask), dtype=bool)
        for k in range(m):
            if k != i and k != j:
                others_mask &= ~per_masks[k]
        # Words in (mask_i | mask_j) & others_mask
        unique_to_pair = (per_masks[i] | per_masks[j]) & others_mask
        if not np.any(unique_to_pair):
            continue
        # Find a single word whose ball covers all of unique_to_pair
        # Try centers from the unique words themselves and from i, j
        candidate_centers = set()
        for idx in np.where(unique_to_pair)[0][:50]:
            candidate_centers.add(_index_to_word(idx, q, n))
        candidate_centers.add(tuple(codewords[i]))
        candidate_centers.add(tuple(codewords[j]))
        for cw_tuple in candidate_centers:
            cw = list(cw_tuple)
            m_new = _ball_mask_single(cw, q, n, radius)
            if np.all(m_new[unique_to_pair]):
                # Check: after removing i and j and adding cw, is everything still covered?
                new_mask = mask.copy()
                # Remove i and j contributions: but we need to recompute from scratch or track
                # Simpler: build new mask from all codewords except i,j, plus cw
                new_mask = np.zeros(len(mask), dtype=bool)
                new_per = []
                new_cws = []
                for k in range(m):
                    if k == i or k == j:
                        continue
                    new_cws.append(codewords[k])
                    new_per.append(per_masks[k])
                    new_mask |= per_masks[k]
                new_cws.append(cw)
                new_per.append(m_new)
                new_mask |= m_new
                if np.all(new_mask):
                    return True, new_cws, new_mask, new_per
    return False, codewords, mask, per_masks


def _try_1_for_2(codewords, mask, per_masks, q, n, radius, max_tries=100):
    """
    Try to replace one codeword with two that cover more / allow future reduction.
    Actually, 1-for-2 increases size; skip unless it enables a subsequent 2-for-1.
    For simplicity, skip this move.
    """
    return False, codewords, mask, per_masks


def _polish(codewords, q, n, radius):
    """
    Final polish: try to remove codewords one by one if coverage remains.
    """
    mask, per_masks = _ball_mask_vectorized(codewords, q, n, radius)
    if not np.all(mask):
        return codewords
    m = len(codewords)
    changed = True
    while changed and _time_left() > 5:
        changed = False
        for i in range(len(codewords)):
            if _time_left() < 5:
                break
            # Try removing codeword i
            test_mask = np.ones(len(mask), dtype=bool)
            for k in range(len(codewords)):
                if k != i:
                    test_mask &= per_masks[k]
            if np.all(test_mask):
                codewords.pop(i)
                per_masks.pop(i)
                mask = test_mask
                changed = True
                break
    return codewords


def construct(q, n, radius):
    t_start = time.time()

    # Initial solution: tetracode product (if n >= 4 and q == 3)
    if q == 3 and n >= 4 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # Fallback: greedy from scratch
        codewords = []
        mask = np.zeros(q ** n, dtype=bool)
        codewords, mask = _greedy_add(codewords, mask, q, n, radius, max_add=5000)
        if not np.all(mask):
            # Still not covered; add more
            while not np.all(mask) and _time_left() > 10:
                codewords, mask = _greedy_add(codewords, mask, q, n, radius, max_add=100)

    # Ensure validity
    mask, per_masks = _ball_mask_vectorized(codewords, q, n, radius)
    if not np.all(mask):
        codewords, mask = _greedy_add(codewords, mask, q, n, radius, max_add=1000)
        mask, per_masks = _ball_mask_vectorized(codewords, q, n, radius)

    # Local search: 2-for-1 moves
    best_cw = codewords
    best_size = len(codewords)
    for iteration in range(50):
        if _time_left() < 10:
            break
        success, new_cw, new_mask, new_per = _try_2_for_1(codewords, mask, per_masks, q, n, radius, max_tries=300)
        if success:
            codewords, mask, per_masks = new_cw, new_mask, new_per
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_cw = codewords[:]
        else:
            # No improvement; break early
            break

    # Final polish
    if _time_left() > 5:
        codewords = _polish(codewords, q, n, radius)
        if len(codewords) < best_size:
            best_cw = codewords[:]

    # Safety: ensure validity of best_cw
    if _time_left() > 2:
        final_mask, _ = _ball_mask_vectorized(best_cw, q, n, radius)
        if not np.all(final_mask):
            # Fall back to the last known valid
            best_cw = codewords
            final_mask, _ = _ball_mask_vectorized(best_cw, q, n, radius)
            if not np.all(final_mask):
                # Emergency: use the initial tetracode product
                if q == 3 and n >= 4 and radius == 1:
                    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                    best_cw = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return best_cw
