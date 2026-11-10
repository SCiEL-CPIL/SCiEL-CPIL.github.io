"""Improved superpermutation construction for n=7 (and smaller n).
Explores multiple permutation orderings at the recursive construction level
to find a shorter superpermutation than the classic one."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import sys
from collections import deque


def construct(n):
    deadline = time.time() + 115
    t_start = time.time()

    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234121341243121432142314"
    if n == 5:
        return _classic_superperm(5)
    if n == 6:
        return _classic_superperm(6)

    # For n=7: try multiple orderings
    best_word = _classic_superperm(7)
    best_len = len(best_word)

    # Try a few alternative orderings of the innermost permutations
    rng = random.Random(12345)
    attempts = 0
    max_attempts = 20

    while time.time() < deadline - 10 and attempts < max_attempts:
        attempts += 1
        # Randomly perturb the order of the 6-permutations at the innermost level
        word = _classic_superperm_with_order(7, rng)
        if len(word) < best_len:
            best_len = len(word)
            best_word = word

    return best_word


def _classic_superperm(n):
    """Standard recursive superpermutation construction."""
    return _classic_superperm_with_order(n, random.Random(42))


def _classic_superperm_with_order(n, rng=None):
    """
    Recursive superpermutation construction with optional randomization
    of the permutation order at each level.
    """
    if rng is None:
        rng = random.Random(42)

    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)

        # For the innermost level (m == n), we can randomize the order
        # For other levels, keep the natural order to maintain structure
        if m == n:
            # Try a random shuffle with some bias toward keeping similar perms together
            # Use a random permutation of the list
            shuffled = list(perms)
            rng.shuffle(shuffled)
            perms = shuffled
        # else: keep natural order

        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            best_overlap = 0
            max_possible = min(len(new), len(block))
            for o in range(max_possible, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    best_overlap = o
                    break
            new += block[best_overlap:]
        word = new
    return "".join(map(str, word))


# EVOLVE-BLOCK-END
