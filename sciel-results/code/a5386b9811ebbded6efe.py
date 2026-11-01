"""Stronger design for K_3(6,1): greedy set-cover over the 729-word space,
with a time-budget guard and a fast NumPy validity/coverage check."""
import time
import itertools
import numpy as np

# Deadline guard: the evaluator kills after 150s; leave a safety margin.
_DEADLINE = 150.0
_START = None


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE - 0.5 - _elapsed()


def _tetracode():
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _direct_sum_code(q, n, radius):
    # Direct sum of ternary tetracode with all tails (9 * 3**(n-4) words).
    tetracode = _tetracode()
    if n <= 4:
        return [h[:n] for h in tetracode]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]


def _is_covering(codewords, q, n, radius):
    """Return True if every word in {0..q-1}^n is within Hamming distance <= radius of some codeword."""
    if not codewords:
        return q == 1 and n == 0
    cw = np.array(codewords, dtype=np.int8)
    # All words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    # Compute min Hamming distance to any codeword efficiently in chunks.
    covered = np.zeros(len(all_words), dtype=bool)
    chunk = 4096
    for s in range(0, len(all_words), chunk):
        block = all_words[s:s + chunk]
        # distances: (chunk, len(cw))
        diff = (block[:, None, :] != cw[None, :, :]).sum(axis=2)
        covered[s:s + chunk] = (diff.min(axis=1) <= radius)
    return bool(covered.all())


def _greedy_cover(q, n, radius):
    """Greedy set-cover: repeatedly add the word that covers the most currently-uncovered words.
    Uses precomputed coverage bitsets via Hamming balls."""
    # Precompute all words as integer indices for compact bitsets.
    total = q ** n
    # Map word tuple -> index
    words = list(itertools.product(range(q), repeat=n))
    # Precompute Hamming ball (set of indices within radius) for each candidate center.
    # For q=3,n=6: 729 balls of size 1 + 6*2 = 13 (radius 1). Cheap.
    ball = []
    for w in words:
        w_arr = np.array(w, dtype=np.int8)
        idx = set()
        idx.add(words.index(w))
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                v = list(w)
                v[pos] = val
                idx.add(words.index(tuple(v)))
        # radius 1 only for this task; generalize below if needed
        if radius > 1:
            # fallback: brute force membership for higher radius
            idx = set()
            for w2 in words:
                if sum(a != b for a, b in zip(w, w2)) <= radius:
                    idx.add(words.index(w2))
        ball.append(idx)

    uncovered = set(range(total))
    code = []
    # Precompute ball sizes for tie-breaking / ordering
    # Greedy: pick center maximizing |ball(center) & uncovered|
    while uncovered and _remaining() > 0.2:
        best = None
        best_gain = -1
        # Iterate candidates; to speed up, we can stop early if we find full coverage possible
        for i in range(total):
            if not uncovered:
                break
            gain = len(ball[i] & uncovered)
            if gain > best_gain:
                best_gain = gain
                best = i
                if best_gain == 0:
                    break
        if best is None or best_gain <= 0:
            break
        code.append(words[best])
        uncovered -= ball[best]
    return code


def _remove_redundant(codewords, q, n, radius):
    """Try to remove codewords that are redundant (their ball is covered by others)."""
    if not codewords:
        return codewords
    words = list(itertools.product(range(q), repeat=n))
    total = q ** n
    # Precompute balls
    ball = []
    for w in codewords:
        idx = set()
        for w2 in words:
            if sum(a != b for a, b in zip(w, w2)) <= radius:
                idx.add(w2)
        ball.append(idx)
    # Union of all balls must cover all words
    union = set()
    for b in ball:
        union |= b
    # Remove one by one if union still covers all
    keep = [True] * len(codewords)
    for i in range(len(codewords)):
        if not keep[i]:
            continue
        if _remaining() < 0.2:
            break
        # Check if removing i still covers all
        # Compute union of all kept except i
        u = set()
        for j in range(len(codewords)):
            if j == i or not keep[j]:
                continue
            u |= ball[j]
        if u == union:
            keep[i] = False
    return [cw for cw, k in zip(codewords, keep) if k]


def construct(q, n, radius):
    global _START
    _START = time.time()

    # Fast baseline: direct sum construction
    baseline = _direct_sum_code(q, n, radius)
    # Validate baseline
    if not _is_covering(baseline, q, n, radius):
        # Should not happen, but fall back to brute force if needed
        return baseline

    best = baseline
    best_len = len(baseline)

    # Try greedy set-cover to potentially find fewer codewords
    if _remaining() > 5.0 and q ** n <= 20000:
        greedy = _greedy_cover(q, n, radius)
        if greedy and _is_covering(greedy, q, n, radius):
            if len(greedy) < best_len:
                best = greedy
                best_len = len(greedy)

    # Try to trim redundant codewords from the best
    if _remaining() > 2.0:
        trimmed = _remove_redundant(best, q, n, radius)
        if trimmed and _is_covering(trimmed, q, n, radius):
            if len(trimmed) < best_len:
                best = trimmed
                best_len = len(trimmed)

    return best
