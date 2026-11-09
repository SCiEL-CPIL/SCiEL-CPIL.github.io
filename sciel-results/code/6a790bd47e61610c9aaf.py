# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
import random


def _base_cover(q, n, radius):
    # Direct sum of the ternary tetracode (9 words of length 4, perfect covering radius 1)
    # with every word on the remaining n - 4 coordinates: 9 * 3**(n - 4) words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(q) for b in range(q)]
    if n <= 4:
        return [list(h[:n]) for h in tetracode]
    return [h + list(tail) for h in tetracode
            for tail in itertools.product(range(q), repeat=n - 4)]


def _is_cover(codewords, q, n, radius):
    if not codewords:
        return n == 0
    C = np.array(codewords, dtype=np.int32).reshape(-1, n)
    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    # word i is covered by codeword j iff sum_{k} (w_i,k != c_j,k) <= radius
    diff = (words[:, None, :] != C[None, :, :])
    dist = diff.sum(axis=2)
    return bool(np.all(dist.min(axis=1) <= radius))


def _greedy_cover(q, n, radius, deadline, rng=None, init_order='max'):
    """Greedy set-cover with configurable initialization ordering."""
    if n == 0:
        return [[]]

    if rng is None:
        rng = random.Random(42)

    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    M = words.shape[0]

    # Build ball-membership matrix: ball[i][j] = True if word j is within radius of word i
    diff = (words[:, None, :] != words[None, :, :])
    dist = diff.sum(axis=2)
    del diff
    ball = dist <= radius
    del dist
    del words

    col_sums = ball.sum(axis=1).astype(np.int64)  # size of each ball

    covered = np.zeros(M, dtype=bool)
    code_idx = []
    remaining = M

    while remaining > 0:
        if time.time() > deadline:
            break
        cand = ~covered
        if not cand.any():
            break

        if init_order == 'max':
            gains = np.where(cand, col_sums, -1)
            j = int(np.argmax(gains))
        elif init_order == 'random':
            cand_indices = np.where(cand)[0]
            if len(cand_indices) == 0:
                break
            j = int(cand_indices[rng.randrange(len(cand_indices))])
        elif init_order == 'weighted_random':
            # Weight by ball size, pick randomly among candidates
            gains = np.where(cand, col_sums, 0).astype(np.float64)
            total = gains.sum()
            if total <= 0:
                cand_indices = np.where(cand)[0]
                j = int(cand_indices[rng.randrange(len(cand_indices))])
            else:
                probs = gains / total
                j = int(rng.choice(M, p=probs))
        else:
            gains = np.where(cand, col_sums, -1)
            j = int(np.argmax(gains))

        code_idx.append(j)
        newcov = ball[j] & ~covered
        if newcov.any():
            covered |= newcov
            remaining -= int(newcov.sum())

    return [list(words_orig) for words_orig in _words_from_indices(code_idx, q, n)]


def _words_from_indices(indices, q, n):
    """Convert integer indices back to word tuples."""
    result = []
    for idx in indices:
        word = []
        rem = int(idx)
        for _ in range(n):
            word.append(rem % q)
            rem //= q
        result.append(word)
    return result


def _local_polish(codewords, q, n, radius, deadline):
    """Try to remove redundant codewords and swap in better ones."""
    if not codewords:
        return codewords

    C = np.array(codewords, dtype=np.int32).reshape(-1, n)
    M_all = q ** n
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)

    # Compute coverage: for each word, which codewords cover it
    diff = (all_words[:, None, :] != C[None, :, :])
    dist = diff.sum(axis=2)
    del diff
    cover_matrix = dist <= radius  # M_all x len(C)
    del dist

    # A codeword is redundant if every word it covers is also covered by others
    # i.e., if removing it doesn't leave any word uncovered
    current_code = list(range(len(C)))
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
        for i in current_code:
            if time.time() > deadline:
                break
            # Check if codeword i is redundant
            temp_cover = cover_matrix[:, [c for c in current_code if c != i]]
            if np.all(temp_cover.any(axis=1)):
                # Codeword i is redundant
                current_code.remove(i)
                changed = True
                break

    result = [list(C[i]) for i in current_code]

    # Try swaps: for each codeword, see if replacing it with an uncovered-neighbor
    # or a better-positioned word reduces the total or maintains coverage with fewer
    if time.time() > deadline:
        return result

    C2 = np.array(result, dtype=np.int32).reshape(-1, n)
    diff2 = (all_words[:, None, :] != C2[None, :, :])
    dist2 = diff2.sum(axis=2)
    del diff2
    cover2 = dist2 <= radius
    del dist2

    # For each codeword, try removing it and see if we can add one that covers
    # more uncovered words (or same coverage with a "better" position)
    for i in range(len(C2)):
        if time.time() > deadline:
            break
        without_i = cover2[:, [c for c in range(len(C2)) if c != i]]
        uncovered_by_others = ~without_i.any(axis=1)
        if not uncovered_by_others.any():
            # Redundant, remove it
            result.pop(i)
            break

    return result


def _greedy_cover_v2(q, n, radius, deadline):
    """Improved greedy with multiple strategies and polish."""
    if n == 0:
        return [[]]

    M = q ** n
    if M > 100000:
        # Too large for full matrix approach; fall back to simpler method
        return _base_cover(q, n, radius)

    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)

    diff = (words[:, None, :] != words[None, :, :])
    dist = diff.sum(axis=2)
    del diff
    ball = dist <= radius
    del dist
    del words

    col_sums = ball.sum(axis=1).astype(np.int64)

    best_code = None
    best_size = M + 1

    # Strategy 1: Standard greedy (max gain)
    covered = np.zeros(M, dtype=bool)
    code_idx = []
    remaining = M
    while remaining > 0 and time.time() < deadline:
        cand = ~covered
        gains = np.where(cand, col_sums, -1)
        j = int(np.argmax(gains))
        if gains[j] <= 0:
            j = int(np.argmax(cand))
        code_idx.append(j)
        newcov = ball[j] & ~covered
        if newcov.any():
            covered |= newcov
            remaining -= int(newcov.sum())

    if remaining == 0:
        cand_code = _words_from_indices(code_idx, q, n)
        if len(cand_code) < best_size:
            best_size = len(cand_code)
            best_code = cand_code

    # Strategy 2: Greedy with tie-breaking by minimum overlap with already-chosen
    if time.time() < deadline - 5:
        covered = np.zeros(M, dtype=bool)
        code_idx = []
        remaining = M
        chosen_mask = np.zeros(M, dtype=bool)
        while remaining > 0 and time.time() < deadline:
            cand = ~covered
            gains = np.where(cand, col_sums, -1)
            # Among max-gain candidates, pick one with least overlap with chosen
            max_gain = gains.max()
            if max_gain <= 0:
                j = int(np.argmax(cand))
            else:
                max_cands = np.where((gains == max_gain) & cand)[0]
                if len(max_cands) == 1:
                    j = int(max_cands[0])
                else:
                    # Pick the one with least overlap with already-chosen balls
                    overlaps = ball[max_cands][:, chosen_mask].sum(axis=1)
                    j = int(max_cands[np.argmin(overlaps)])
            code_idx.append(j)
            chosen_mask[j] = True
            newcov = ball[j] & ~covered
            if newcov.any():
                covered |= newcov
                remaining -= int(newcov.sum())

        if remaining == 0:
            cand_code = _words_from_indices(code_idx, q, n)
            if len(cand_code) < best_size:
                best_size = len(cand_code)
                best_code = cand_code

    # Strategy 3: Greedy with random restarts
    rng = random.Random(12345)
    for trial in range(3):
        if time.time() > deadline - 3:
            break
        covered = np.zeros(M, dtype=bool)
        code_idx = []
        remaining = M
        while remaining > 0 and time.time() < deadline:
            cand = ~covered
            gains = np.where(cand, col_sums, -1)
            max_gain = gains.max()
            if max_gain <= 0:
                cand_indices = np.where(cand)[0]
                j = int(cand_indices[rng.randrange(len(cand_indices))])
            else:
                max_cands = np.where((gains == max_gain) & cand)[0]
                j = int(max_cands[rng.randrange(len(max_cands))])
            code_idx.append(j)
            newcov = ball[j] & ~covered
            if newcov.any():
                covered |= newcov
                remaining -= int(newcov.sum())

        if remaining == 0:
            cand_code = _words_from_indices(code_idx, q, n)
            if len(cand_code) < best_size:
                best_size = len(cand_code)
                best_code = cand_code

    if best_code is None:
        return _base_cover(q, n, radius)

    # Polish: try to remove redundant codewords
    if time.time() < deadline - 2:
        polished = _local_polish(best_code, q, n, radius, deadline)
        if len(polished) < len(best_code):
            best_code = polished

    return best_code


def construct(q, n, radius):
    start = time.time()
    deadline = start + 110.0  # safety margin vs 120s wall-clock

    if n == 0:
        return [[]]

    # Baseline that is guaranteed valid and always available to return.
    base = _base_cover(q, n, radius)
    base = base[:1000000]
    if not _is_cover(base, q, n, radius):
        base = [list(w) for w in itertools.product(range(q), repeat=n)]
    best = base

    # Try improved greedy with multiple strategies
    try:
        cand = _greedy_cover_v2(q, n, radius, deadline)
        if len(cand) < len(best) and _is_cover(cand, q, n, radius):
            best = cand
    except Exception:
        pass

    if time.time() > deadline:
        return best
    return best
# EVOLVE-BLOCK-END
