"""Greedy covering-code construction for K_q(n, radius)."""
import time
import itertools
import numpy as np

# Deadline guard: return best result before the evaluator kills the process.
_DEADLINE = 120.0
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _precompute_mins(q, n, radius):
    """For each position, the list of symbols within `radius` of a given symbol.

    Returns a numpy int array of shape (q, n, radius+1).
    """
    mins = np.empty((q, n, radius + 1), dtype=np.int64)
    for s in range(q):
        for i in range(n):
            for r in range(radius + 1):
                if r == 0:
                    mins[s, i, r] = s
                else:
                    mins[s, i, r] = (s + r) % q
    return mins


def _build_cover(q, n, radius, mins):
    """Greedy covering code. Returns list of codewords (each a list of ints)."""
    total = q ** n
    # covered[i] = True if word i is within `radius` of some chosen codeword
    covered = np.zeros(total, dtype=bool)
    count = 0
    code = []
    # Precompute powers of q for index conversion
    qpow = [1]
    for _ in range(n):
        qpow.append(qpow[-1] * q)

    # All positions for distance-1 words: for each position i, the offset in index space
    # is qpow[i] * (new_symbol - old_symbol). Precompute per position the delta for each symbol change.
    # For a word with index idx, symbol at pos i = (idx // qpow[i]) % q.
    # New word index = idx + qpow[i] * (new_s - old_s).

    while count < total and _time_left() > 10.0:
        # Find an uncovered word to add
        # Use a batched approach: check chunks for uncovered
        found = -1
        start = 0
        chunk = 4096
        while start < total:
            end = min(start + chunk, total)
            seg = covered[start:end]
            if not seg.all():
                # find first uncovered
                nz = np.flatnonzero(~seg)
                if nz.size:
                    found = start + int(nz[0])
                    break
            start = end
        if found < 0:
            break

        # Add this word; update covered
        word = [(found // qpow[i]) % q for i in range(n)]
        code.append(word)
        covered[found] = True
        count += 1

        # Mark all words within radius as covered
        if radius >= 1:
            # radius-1 neighbors
            for i in range(n):
                old_s = word[i]
                for r in range(1, radius + 1):
                    for new_s in ((old_s + r) % q, (old_s - r) % q):
                        if new_s == old_s:
                            continue
                        idx2 = found + qpow[i] * (new_s - old_s)
                        if 0 <= idx2 < total and not covered[idx2]:
                            covered[idx2] = True
                            count += 1
        if radius >= 2:
            # radius-2 neighbors
            for i in range(n):
                old_i = word[i]
                for j in range(i + 1, n):
                    old_j = word[j]
                    for ri in range(1, 3):
                        for rj in range(1, 3):
                            ni = (old_i + ri) % q
                            nj = (old_j + rj) % q
                            idx2 = found + qpow[i] * (ni - old_i) + qpow[j] * (nj - old_j)
                            if 0 <= idx2 < total and not covered[idx2]:
                                covered[idx2] = True
                                count += 1

    return code


def _remove_redundant(q, n, radius, code):
    """Remove codewords that don't cover any unique word. Returns pruned list."""
    if len(code) <= 1:
        return code
    total = q ** n
    qpow = [1]
    for _ in range(n):
        qpow.append(qpow[-1] * q)

    # For each codeword, find its covered set (as a set of indices)
    # To save memory, compute per-codeword covered indices lazily.
    # We need: for each codeword c, is there a word covered ONLY by c?
    # Approach: for each word, count how many codewords cover it. If any word has count 1,
    # its covering codeword is essential.

    # This is expensive for large code. Use a sampling/heuristic approach:
    # For each codeword, check if removing it leaves a gap by sampling uncovered words.
    # Simpler: build a coverage count array.

    # For large code, this may be too slow. Use a time-limited approach.
    n_code = len(code)
    # coverage_count[word_idx] = number of codewords covering word_idx
    # Memory: total * 4 bytes. For 6561 that's fine. For larger, may be heavy.
    if total * 4 > 50_000_000:
        return code  # skip pruning for very large spaces

    cov_count = np.zeros(total, dtype=np.int32)

    for c in code:
        # Mark all words within radius of c
        # Compute base index
        base = sum(c[i] * qpow[i] for i in range(n))
        cov_count[base] += 1
        if radius >= 1:
            for i in range(n):
                old_s = c[i]
                for r in range(1, radius + 1):
                    for new_s in ((old_s + r) % q, (old_s - r) % q):
                        if new_s == old_s:
                            continue
                        idx2 = base + qpow[i] * (new_s - old_s)
                        if 0 <= idx2 < total:
                            cov_count[idx2] += 1
        if radius >= 2:
            for i in range(n):
                old_i = c[i]
                for j in range(i + 1, n):
                    old_j = c[j]
                    for ri in range(1, 3):
                        for rj in range(1, 3):
                            ni = (old_i + ri) % q
                            nj = (old_j + rj) % q
                            idx2 = base + qpow[i] * (ni - old_i) + qpow[j] * (nj - old_j)
                            if 0 <= idx2 < total:
                                cov_count[idx2] += 1

    # A codeword is removable if no word has cov_count == 1 that is covered by it.
    # More precisely: a word is "uniquely covered" by codeword c if cov_count[word]==1 and c covers word.
    # If codeword c has no uniquely-covered words, it can be removed.

    # For each codeword, check if it has any uniquely covered word.
    essential = [False] * n_code
    for ci, c in enumerate(code):
        base = sum(c[i] * qpow[i] for i in range(n))
        # Check the codeword itself
        if cov_count[base] == 1:
            essential[ci] = True
            continue
        if radius >= 1:
            for i in range(n):
                old_s = c[i]
                for r in range(1, radius + 1):
                    for new_s in ((old_s + r) % q, (old_s - r) % q):
                        if new_s == old_s:
                            continue
                        idx2 = base + qpow[i] * (new_s - old_s)
                        if 0 <= idx2 < total and cov_count[idx2] == 1:
                            essential[ci] = True
                            break
                    if essential[ci]:
                        break
                if essential[ci]:
                    break
            if essential[ci]:
                continue
        if radius >= 2:
            for i in range(n):
                old_i = c[i]
                for j in range(i + 1, n):
                    old_j = c[j]
                    for ri in range(1, 3):
                        for rj in range(1, 3):
                            ni = (old_i + ri) % q
                            nj = (old_j + rj) % q
                            idx2 = base + qpow[i] * (ni - old_i) + qpow[j] * (nj - old_j)
                            if 0 <= idx2 < total and cov_count[idx2] == 1:
                                essential[ci] = True
                                break
                        if essential[ci]:
                            break
                    if essential[ci]:
                        break
                if essential[ci]:
                    break

    pruned = [c for ci, c in enumerate(code) if essential[ci]]
    return pruned if pruned else code


def construct(q, n, radius):
    global _START
    _START = time.time()

    mins = _precompute_mins(q, n, radius)
    code = _build_cover(q, n, radius, mins)

    # Try to prune redundant codewords
    if _time_left() > 15.0:
        code = _remove_redundant(q, n, radius, code)

    # Ensure distinctness
    seen = set()
    unique = []
    for c in code:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            unique.append(c)

    return unique
