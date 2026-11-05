"""Rewritten program: greedy set-cover + local search for ternary covering code."""

import time
import itertools
import random

_DEADLINE = 115.0  # safety margin before 120s kill
_START = None
_BEST = None


def _elapsed():
    return time.monotonic() - _START


def _hamming(a, b):
    """Hamming distance between two tuples."""
    d = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            d += 1
    return d


def _ball_indices(word, n, q):
    """Return a frozenset of all words within Hamming distance 1 of `word`."""
    # word is a tuple of length n over 0..q-1
    covered = {word}
    for pos in range(n):
        orig = word[pos]
        for v in range(q):
            if v == orig:
                continue
            new_word = list(word)
            new_word[pos] = v
            covered.add(tuple(new_word))
    return frozenset(covered)


def construct(q, n, radius):
    global _START, _BEST
    _START = time.monotonic()
    _BEST = None

    # For this specific task: q=3, n=6, radius=1
    # Total space size: 3^6 = 729
    # Each codeword covers 1 + 6*2 = 13 words (ball of radius 1)
    # Lower bound: 729/13 ≈ 56, but structure matters.
    # Known best: 73.

    if radius != 1:
        # Fallback: direct sum construction for general radius (may not be optimal)
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Brute-force single word covers everything if radius >= n
            if radius >= n:
                return [[0] * n]
            # Greedy fallback
            return _greedy_general(q, n, radius)

    # Specialized for q=3, n=6, radius=1
    all_words = [tuple(w) for w in itertools.product(range(3), repeat=n)]
    total = len(all_words)
    word_index = {w: i for i, w in enumerate(all_words)}

    # Precompute balls for all words
    balls = [_ball_indices(w, n, q) for w in all_words]

    # --- Greedy initialization ---
    def greedy_init(seed_words=None, seed_size=0):
        """Greedy set cover. Returns list of codeword indices."""
        uncovered = set(range(total))
        chosen = []

        # Optional: start with a small seed to break symmetry
        if seed_words:
            for sw in seed_words:
                if sw in uncovered:
                    chosen.append(sw)
                    uncovered -= balls[sw]

        while uncovered and _elapsed() < _DEADLINE - 20:
            # Find word covering most uncovered words
            best_word = -1
            best_count = -1
            for wi in range(total):
                if wi in chosen:
                    continue
                count = len(balls[wi] & uncovered)
                if count > best_count:
                    best_count = count
                    best_word = wi
                    if best_count == len(uncovered):
                        break
            if best_word < 0:
                break
            chosen.append(best_word)
            uncovered -= balls[best_word]

        return chosen

    # --- Local search: try to improve by swaps and deletions ---
    def local_search(codewords):
        """Try to reduce the number of codewords via deletion and swap."""
        codewords = list(codewords)
        improved = True
        while improved and _elapsed() < _DEADLINE - 10:
            improved = False

            # Step 1: Try to delete redundant words
            for i in range(len(codewords)):
                if _elapsed() > _DEADLINE - 10:
                    break
                # Check if removing codewords[i] still covers everything
                remaining = [c for j, c in enumerate(codewords) if j != i]
                # Check coverage
                covered = set()
                for c in remaining:
                    covered |= balls[c]
                if len(covered) == total:
                    codewords.pop(i)
                    improved = True
                    break

            # Step 2: Try swaps - replace one word with another to maintain coverage
            if not improved and _elapsed() < _DEADLINE - 15:
                for i in range(len(codewords)):
                    if _elapsed() > _DEADLINE - 15:
                        break
                    # Find words uniquely covered by codewords[i]
                    unique_to_i = set()
                    for j in range(len(codewords)):
                        if j == i:
                            continue
                        unique_to_i -= balls[codewords[j]]
                    unique_to_i = balls[codewords[i]] - unique_to_i

                    # Try replacing codewords[i] with some other word
                    # The new word must cover all of unique_to_i
                    for new_word in range(total):
                        if new_word in codewords:
                            continue
                        if not unique_to_i.issubset(balls[new_word]):
                            continue
                        # Check if this swap doesn't break coverage elsewhere
                        # Actually, since new_word covers unique_to_i, and the rest
                        # is covered by others, we just need to ensure new_word's
                        # coverage doesn't leave gaps that were covered by codewords[i]
                        # But codewords[i]'s coverage is: unique_to_i (now covered by new_word)
                        # and the rest (covered by others). So the swap is valid.
                        codewords[i] = new_word
                        improved = True
                        break
                    if improved:
                        break

        return codewords

    # --- Main search ---
    # Start with a greedy solution
    best_cw = greedy_init()
    best_cw = local_search(best_cw)
    _BEST = best_cw[:]

    # Try multiple random restarts with different seeds
    rng = random.Random(42)
    restart = 0
    while _elapsed() < _DEADLINE - 5:
        restart += 1
        # Create a small random seed to break symmetry
        seed_size = min(3, total)
        seed_words = rng.sample(range(total), seed_size)
        cw = greedy_init(seed_words=seed_words)
        cw = local_search(cw)

        if len(cw) < len(_BEST):
            _BEST = cw[:]

        # If we've hit the known best, try to go below
        if len(_BEST) <= 73:
            # Do more aggressive local search
            if _elapsed() < _DEADLINE - 10:
                _BEST = local_search(_BEST)

    # Final polish: try more deletions and swaps on the best
    if _elapsed() < _DEADLINE - 3:
        _BEST = local_search(_BEST)

    # Convert to list of lists
    return [list(all_words[i]) for i in _BEST]


def _greedy_general(q, n, radius):
    """General greedy for other parameters."""
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    total = len(all_words)

    def ball(word):
        covered = set()
        # Generate all words within Hamming distance radius
        for delta in itertools.product(range(q), repeat=n):
            d = sum(1 for a, b in zip(word, delta) if a != b)
            if d <= radius:
                covered.add(tuple((a + b) % q for a, b in zip(word, delta)))
        return frozenset(covered)

    balls = [ball(w) for w in all_words]
    uncovered = set(range(total))
    chosen = []

    while uncovered:
        best_word = -1
        best_count = -1
        for wi in range(total):
            if wi in chosen:
                continue
            count = len(balls[wi] & uncovered)
            if count > best_count:
                best_count = count
                best_word = wi
        if best_word < 0:
            break
        chosen.append(best_word)
        uncovered -= balls[best_word]

    return [list(all_words[i]) for i in chosen]
