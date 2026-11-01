"""Football pool K_3(7,1): greedy set-cover with vectorized coverage + local 1-for-1 swap repair."""
import time
import numpy as np

_DEADLINE = 150.0  # evaluator kills at 150s wall-clock; keep a safety margin


def _time_left():
    return _DEADLINE - time.time()


def construct(q, n, radius):
    t0 = time.time()
    if radius != 1 or q != 3 or n != 7:
        # Fallback: trivial cover (only correct for the specific task, but keep valid interface)
        return [[0] * n]

    # Precompute all words of length 7 over {0,1,2} as a flat int64 array in base-3.
    # 2187 words.
    words = np.zeros(3 ** n, dtype=np.int64)
    for i in range(3 ** n):
        v = i
        for pos in range(n):
            words[i] += (v % q) * (q ** pos)
            v //= q

    # Precompute for each position and each digit value, the mask of words having that digit at that position.
    # This lets us quickly compute the ball of a codeword: a word is within distance 1 of c iff
    # for some position p, word[p] == c[p], OR for some position p, word[p] != c[p] but we allow one mismatch.
    # Actually distance <= 1 means: either equal, or differs in exactly one position.
    # So ball(c) = union over p of {words that match c in all positions except possibly p}.
    # = union over p of (intersection over p' != p of {words with word[p'] == c[p']}).

    # Precompute digit masks: digit_mask[p][d] = boolean array of words where position p has digit d.
    digit_mask = np.zeros((n, q, 3 ** n), dtype=bool)
    for p in range(n):
        for d in range(q):
            digit_mask[p][d] = (words // (q ** p)) % q == d

    # Greedy set cover:
    # uncovered = boolean array of words not yet covered
    uncovered = np.ones(3 ** n, dtype=bool)
    uncovered_count = 3 ** n
    selected = []  # list of codeword tuples (as base-3 int)

    # Precompute for each candidate codeword (which is any word in the space) its ball size and ball mask.
    # But 2187 candidates is small. We can compute ball masks on the fly or precompute all.
    # Precompute ball_mask for all 2187 words: ball_mask[i] = boolean array of words within dist 1 of word i.
    # This is 2187 * 2187 booleans = ~4.8M booleans = ~4.8 MB. Fine.
    ball_mask = np.zeros((3 ** n, 3 ** n), dtype=bool)
    for i in range(3 ** n):
        c = words[i]
        # Ball: words within Hamming distance <= 1 of c.
        # Equivalent: for each position p, take words matching c in all positions except possibly p.
        # Compute via intersection of digit masks for all positions except p, for each p, then OR.
        bm = np.zeros(3 ** n, dtype=bool)
        for p in range(n):
            # Words matching c in all positions except p
            # Start with all True, then AND with digit_mask[p'][c[p']] for p' != p
            mask = np.ones(3 ** n, dtype=bool)
            cp = (c // (q ** p)) % q
            for pp in range(n):
                if pp == p:
                    continue
                cpp = (c // (q ** pp)) % q
                mask &= digit_mask[pp][cpp]
            bm |= mask
        ball_mask[i] = bm

    # Precompute ball sizes
    ball_sizes = ball_mask.sum(axis=1)

    # Greedy: repeatedly pick the word whose ball covers the most uncovered words.
    # To speed up, we can maintain uncovered and for each candidate compute overlap = ball_mask[i] & uncovered.
    # But that's 2187 * 2187 per step. With ~200 steps, that's ~100M ops. Might be slow in Python loop.
    # Vectorize: compute overlaps for all candidates at once using matrix multiply?
    # overlap[i] = sum(ball_mask[i] & uncovered) = dot(ball_mask[i], uncovered.astype(int))
    # We can do: overlaps = (ball_mask @ uncovered.astype(np.int8))
    # That's a matrix-vector product: 2187x2187 @ 2187. Fast with NumPy.

    while uncovered_count > 0 and _time_left() > 30:
        unc_int = uncovered.astype(np.int8)
        overlaps = ball_mask @ unc_int
        best_idx = int(np.argmax(overlaps))
        if overlaps[best_idx] == 0:
            break
        selected.append(best_idx)
        uncovered &= ~ball_mask[best_idx]
        uncovered_count = int(uncovered.sum())
        # Remove selected from candidates to avoid re-picking
        ball_mask[best_idx] = False  # zero out so it won't be picked again

    # Convert selected indices to codeword lists
    code = [list((words[i] // (q ** p)) % q for p in range(n)) for i in selected]

    # Local search: try 1-for-1 swaps to reduce code size while maintaining coverage.
    # A swap: remove one codeword c_out, add one new codeword c_in (not in code) such that
    # coverage is maintained. We want to find swaps that allow eventual removal.
    # Simpler: try removing each codeword; if coverage breaks, find a replacement that covers the lost words.
    
    # For local search, use a coverage count array: cov_count[w] = number of selected codewords covering w.
    # We can maintain this incrementally.
    
    # Build cov_count from current code
    cov_count = np.zeros(3 ** n, dtype=np.int32)
    for i in selected:
        cov_count += ball_mask[i].astype(np.int32)
    
    # Track selected as a set for O(1) lookup
    sel_set = set(selected)
    
    # Local search: repeatedly try to remove a codeword and add a better one.
    # Strategy: for each codeword in code, check if removing it leaves any uncovered words.
    # If yes, find a codeword (not in code) that covers the most of those uncovered words, add it, then remove the old one.
    # This is a 1-for-1 swap that doesn't increase size.
    
    improved = True
    while improved and _time_left() > 10:
        improved = False
        for idx_out in list(sel_set):
            # Temporarily remove
            lost = ball_mask[idx_out] & (cov_count == 1)
            if not lost.any():
                # Can remove without replacement
                cov_count -= ball_mask[idx_out].astype(np.int32)
                sel_set.remove(idx_out)
                selected.remove(idx_out)
                code.remove(list((words[idx_out] // (q ** p)) % q for p in range(n)))
                improved = True
                break
            else:
                # Find best replacement: a codeword not in sel_set that covers the most lost words
                # lost_int = lost.astype(np.int8)
                # gains = ball_mask @ lost_int  but only for non-selected
                # We need to exclude already selected
                # Create a mask of non-selected
                non_sel = np.ones(3 ** n, dtype=bool)
                for s in sel_set:
                    non_sel[s] = False
                # gains for non-selected only
                gains = np.zeros(3 ** n, dtype=np.int32)
                lost_int = lost.astype(np.int8)
                # Only compute for non-selected to save time? Or compute all and mask.
                all_gains = ball_mask @ lost_int
                all_gains[~non_sel] = -1  # disallow selected
                best_in = int(np.argmax(all_gains))
                if all_gains[best_in] < lost.sum():
                    # Can't fully cover lost words, skip this swap
                    continue
                # Perform swap
                cov_count -= ball_mask[idx_out].astype(np.int32)
                cov_count += ball_mask[best_in].astype(np.int32)
                sel_set.remove(idx_out)
                sel_set.add(best_in)
                selected.remove(idx_out)
                selected.append(best_in)
                # Update code list
                code.remove(list((words[idx_out] // (q ** p)) % q for p in range(n)))
                code.append(list((words[best_in] // (q ** p)) % q for p in range(n)))
                improved = True
                break
        if _time_left() < 10:
            break

    # Final: try pure removals again
    for _ in range(3):
        removed_any = False
        for idx_out in list(sel_set):
            lost = ball_mask[idx_out] & (cov_count == 1)
            if not lost.any():
                cov_count -= ball_mask[idx_out].astype(np.int32)
                sel_set.remove(idx_out)
                selected.remove(idx_out)
                code.remove(list((words[idx_out] // (q ** p)) % q for p in range(n)))
                removed_any = True
        if not removed_any:
            break

    return code
