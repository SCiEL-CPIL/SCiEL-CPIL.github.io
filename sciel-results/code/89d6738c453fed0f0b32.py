"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import permutations

# EVOLVE-BLOCK-START
def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    if n <= 1:
        return "1"
    if n <= 3:
        return _standard_superperm(n)

    # For n=4, we can use a known optimal or near-optimal
    if n == 4:
        return _standard_superperm(n)

    # For n >= 5, use a randomized/optimized construction
    # The classic construction gives 1!+2!+...+n! = 33 for n=4, 153 for n=5, 873 for n=6, 5913 for n=7
    # But we can do better with overlap optimization

    # Start with the classic construction
    word = _standard_superperm(n)

    # Try to improve via local search / greedy overlap
    # This is a heuristic approach - we'll try to find shorter words
    # by reordering the insertion of new symbols

    best_word = word
    best_len = len(word)

    # For n up to 7, try multiple random orderings and pick the best
    # The key insight: when adding symbol m, the order in which we insert
    # the permutations affects the total length due to overlaps

    # We'll use a greedy approach with random restarts
    max_restarts = 50 if n <= 6 else 20

    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break

        result = _greedy_construct(n, random.Random(restart * 1000 + 42))
        if result is not None and len(result) < best_len:
            best_word = result
            best_len = len(result)

    return best_word


def _standard_superperm(n):
    """Classic recursive construction of length 1! + 2! + ... + n!"""
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1)
                          if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))


def _greedy_construct(n, rng=None):
    """
    Greedy construction with randomized ordering of permutation insertions.
    When adding symbol m, we consider all (m-1)! permutations of the previous symbols
    and choose the order that maximizes overlap.
    """
    if rng is None:
        rng = random.Random()

    word = [1]
    for m in range(2, n + 1):
        # Get all (m-1)-permutations that appear in the current word
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)

        # Now we need to order these perms to maximize overlap
        # Use a greedy nearest-neighbor approach with random tie-breaking
        # Start with a random perm
        remaining = list(perms)
        rng.shuffle(remaining)
        first = remaining.pop(0)

        new = list(first) + [m] + list(first)
        remaining_perms = remaining

        # For each next perm, pick the one that gives maximum overlap
        while remaining_perms:
            best_idx = -1
            best_overlap = -1
            for i, p in enumerate(remaining_perms):
                block = list(p) + [m] + list(p)
                # Find maximum overlap
                max_ov = min(len(new), len(block))
                for o in range(max_ov, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        if o > best_overlap:
                            best_overlap = o
                            best_idx = i
                        break
            if best_idx == -1:
                # No overlap found, just pick first
                best_idx = 0
                best_overlap = 0
            p = remaining_perms.pop(best_idx)
            block = list(p) + [m] + list(p)
            new += block[best_overlap:]

        word = new

    # Verify
    result = "".join(map(str, word))
    if _verify(result, n):
        return result
    return None


def _verify(word_str, n):
    """Check if word contains all permutations of 1..n as consecutive substrings."""
    if len(word_str) < n:
        return False
    seen = set()
    for i in range(len(word_str) - n + 1):
        w = word_str[i:i + n]
        if len(set(w)) == n:
            seen.add(w)
    return len(seen) == _factorial(n)


def _factorial(k):
    result = 1
    for i in range(2, k + 1):
        result *= i
    return result
# EVOLVE-BLOCK-END
