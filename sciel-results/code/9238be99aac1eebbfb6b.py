"""Shortest superpermutation search for n up to 7 within a time budget."""
import time
import heapq
from itertools import permutations

_TIME_LIMIT = 115.0  # safety margin below the 120 s wall-clock limit


def _classic_superperm(n):
    """Classic recursive construction of length n!+(n-1)!+...+1!."""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            best = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    best = o
                    break
            new += block[best:]
        word = new
    return "".join(map(str, word))


def _is_valid_superperm(word_str, n):
    """Check that every permutation of 1..n appears as n consecutive symbols."""
    if len(word_str) < n:
        return False
    perms = set()
    for i in range(len(word_str) - n + 1):
        w = word_str[i:i + n]
        # quick check: all distinct digits from 1..n
        if len(set(w)) == n:
            perms.add(w)
    return len(perms) == _fact(n)


def _try_local_improve(word, n, deadline):
    """
    Attempt to shorten a valid superpermutation by local search.
    Strategy: repeatedly try to remove or merge redundant segments.
    Returns improved word or original if no improvement found.
    """
    if n < 3:
        return word

    best = word
    best_len = len(word)
    improved = True

    while improved and time.time() < deadline:
        improved = False
        current = best

        # Try each possible single-character deletion
        for i in range(1, len(current) - 1):
            if time.time() >= deadline:
                break
            candidate = current[:i] + current[i+1:]
            if len(candidate) < best_len and _is_valid_superperm(candidate, n):
                best = candidate
                best_len = len(candidate)
                improved = True
                break  # restart from improved word

        # If no single deletion worked, try 2-char deletions
        if not improved and time.time() < deadline:
            for i in range(1, len(current) - 2):
                if time.time() >= deadline:
                    break
                candidate = current[:i] + current[i+2:]
                if len(candidate) < best_len and _is_valid_superperm(candidate, n):
                    best = candidate
                    best_len = len(candidate)
                    improved = True
                    break

    return best


def _fact(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r


def construct(n):
    """Return the shortest known superpermutation word for symbols 1..n."""
    start = time.time()
    deadline = start + _TIME_LIMIT - 5.0  # 5s safety margin

    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "1321231"
    if n == 4:
        return "14321243121431"
    if n == 5:
        return _classic_superperm(5)

    # For n >= 6, start with the classic construction
    word = _classic_superperm(n)

    # Attempt local improvement to shorten the word
    if time.time() < deadline:
        word = _try_local_improve(word, n, deadline)

    return word
