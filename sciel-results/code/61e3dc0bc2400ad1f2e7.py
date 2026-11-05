"""Shortest superpermutation construction with fast block reordering + verification."""
import time
from itertools import permutations

import numpy as np

# Deadline guard: evaluator kills at 150s wall-clock.
_DEADLINE = 145.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _overlaps(a, b, max_ov):
    """Maximal overlap of suffix of a with prefix of b (ints)."""
    if max_ov <= 0:
        return 0
    for o in range(max_ov, 0, -1):
        if a[-o:] == b[:o]:
            return o
    return 0


def construct(n):
    """Construct a superpermutation on n symbols."""
    global _start_time
    _start_time = time.time()

    if n <= 1:
        return "1"

    word = [1]

    for m in range(2, n + 1):
        # Extract all distinct (m-1)-permutations in order of appearance.
        perms, seen = [], set()
        L = len(word)
        for i in range(L - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)

        perms = _reorder(perms, m)

        # Build new word by concatenating blocks with maximal overlap.
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            ov = _overlaps(new, block, min(len(new), len(block)))
            new += block[ov:]
        word = new

        if _elapsed() > _DEADLINE - 5:
            break

    # Final safety: verify the word actually contains every permutation of 1..n.
    if _elapsed() < _DEADLINE - 3:
        word = _verify_or_fallback(word, n)

    return "".join(map(str, word))


def _verify_or_fallback(word, n):
    """Check that every permutation of 1..n appears as an n-window; if not, rebuild."""
    try:
        a = np.asarray(word, dtype=np.int8)
        if a.size < n:
            raise RuntimeError("too short")
        expected = set(permutations(range(1, n + 1)))
        seen = set()
        # Slide windows.
        for i in range(a.size - n + 1):
            seen.add(tuple(a[i:i + n].tolist()))
            if len(seen) == len(expected):
                return word
        if seen == expected:
            return word
    except Exception:
        pass

    # Fallback: plain recursive construction (always valid).
    w = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(w) - m + 2):
            window = tuple(w[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            ov = _overlaps(new, block, min(len(new), len(block)))
            new += block[ov:]
        w = new
    return w


def _reorder(perms, m):
    """Reorder permutation blocks to maximize overlap (fast, vectorized)."""
    n_perms = len(perms)
    if n_perms <= 1:
        return perms

    # For small counts, try multiple starts with best-overlap greedy.
    # For large counts (n=7 -> 720), do a few fast greedy starts.
    # Build blocks once.
    blocks = [list(p) + [m] + list(p) for p in perms]
    blen = len(blocks[0])

    # Precompute per-block prefix/suffix fingerprints for fast overlap via numpy.
    # Represent each block as int array.
    arr = np.asarray(blocks, dtype=np.int8)  # shape (n_perms, blen)

    # Overlap of suffix of i with prefix of j: for candidate o, arr[i, blen-o:] == arr[j, :o].
    # We compute best overlap for each (i, j) by checking o from max down.
    # max overlap between two length-blen blocks is blen-1 (can't be full).
    max_ov = blen - 1

    # Build overlap matrix using vectorized comparison per o.
    # ov[i,j] = max o such that arr[i, blen-o:] == arr[j, :o]
    ov = np.zeros((n_perms, n_perms), dtype=np.int8)
    for o in range(max_ov, 0, -1):
        # suffix of i: arr[:, blen-o:]  shape (n_perms, o)
        # prefix of j: arr[:, :o]       shape (n_perms, o)
        # match[i,j] = all(arr[i, blen-o:] == arr[j, :o])
        # Use broadcasting: (n_perms, 1, o) == (1, n_perms, o) -> (n_perms, n_perms, o)
        # This is heavy for o up to ~13 and n=720: 720*720*o booleans. o*720^2 ~ 13*518k ~ 6.7M, ok per o, *13 ~ 87M. Acceptable.
        # But memory: 720*720*o bytes for o=13 -> 6.7MB, fine.
        if _elapsed() > _DEADLINE - 10:
            break
        suffix = arr[:, blen - o:]  # (n_perms, o)
        prefix = arr[:, :o]         # (n_perms, o)
        match = np.all(suffix[:, None, :] == prefix[None, :, :], axis=2)  # (n_perms, n_perms)
        ov[match] = o

    # Greedy: pick start, then repeatedly pick unvisited j maximizing ov[last][j]
    # (tie-break by smallest index for determinism).
    # Try a few starts.
    if n_perms <= 60:
        starts = list(range(n_perms))
    else:
        # A handful of starts.
        starts = [0, 1, 2, n_perms // 2, n_perms - 1]

    best_order = None
    best_len = None

    for start in starts:
        if _elapsed() > _DEADLINE - 8:
            break
        visited = np.zeros(n_perms, dtype=bool)
        order = [start]
        visited[start] = True
        cur = start
        cur_len = blen
        for _ in range(n_perms - 1):
            if _elapsed() > _DEADLINE - 8:
                break
            row = ov[cur]
            row = row.copy()
            row[visited] = -1
            j = int(np.argmax(row))
            if row[j] < 0 and j != start and visited[j]:
                break
            if visited[j]:
                break
            visited[j] = True
            order.append(j)
            cur_len += blen - int(ov[cur, j])
            cur = j
        if len(order) != n_perms:
            continue
        if best_len is None or cur_len < best_len:
            best_len = cur_len
            best_order = order

    if best_order is not None:
        return [perms[i] for i in best_order]
    return perms
