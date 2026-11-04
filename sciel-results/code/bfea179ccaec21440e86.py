"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(q, n, radius):
    # Deadline guard: return best result before 115s (safety margin under 120s limit)
    deadline = time.time() + 115.0

    # For the specific case q=3, n=6, radius=1, use a greedy set-cover approach
    if q == 3 and n == 6 and radius == 1:
        return _greedy_cover_3_6_1(deadline)

    # Fallback: direct sum construction (same as original)
    if n >= 4 and q == 3 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # Generic fallback: all-zero word covers radius-radius sphere, but for general case
        # use a simple greedy approach
        return _greedy_cover_general(q, n, radius, deadline)


def _neighbors_of_word(word, q, n, radius):
    """Generate all words within Hamming distance <= radius from word."""
    if radius == 0:
        yield tuple(word)
        return
    if radius == 1:
        yield tuple(word)
        for pos in range(n):
            orig = word[pos]
            for val in range(q):
                if val != orig:
                    new_word = list(word)
                    new_word[pos] = val
                    yield tuple(new_word)
        return
    # For larger radii, generate all combinations
    for delta_pos in itertools.combinations(range(n), radius):
        for delta_vals in itertools.product(range(1, q), repeat=radius):
            new_word = list(word)
            for i, pos in enumerate(delta_pos):
                new_word[pos] = (new_word[pos] + delta_vals[i]) % q
            yield tuple(new_word)
    # Also include words at distance < radius (handled by radius==1 case recursively)
    if radius > 1:
        # This is incomplete for general radius; for this task radius=1 is the case
        pass


def _greedy_cover_3_6_1(deadline):
    """Greedy set-cover for q=3, n=6, radius=1."""
    q, n, radius = 3, 6, 1
    all_words = list(itertools.product(range(q), repeat=n))
    total = q ** n  # 729

    # Precompute neighbors for each word
    # neighbors[w] = set of words within distance 1 of w
    # For efficiency, use bitmasks or sets

    # Each codeword covers 1 + n*(q-1) = 1 + 6*2 = 13 words
    # Lower bound: 729/13 ≈ 56.08, so at least 57 words
    # Best known: 73

    # Represent coverage as a set of covered words
    # For speed, use a boolean array indexed by word index
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute the coverage set for each possible codeword
    # There are 729 possible codewords; each covers 13 words
    coverage = {}  # word_tuple -> set of word_tuples it covers
    for w in all_words:
        covered = set()
        covered.add(w)
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val != orig:
                    new_w = list(w)
                    new_w[pos] = val
                    covered.add(tuple(new_w))
        coverage[w] = covered

    # Greedy set cover: repeatedly pick the codeword that covers the most uncovered words
    uncovered = set(all_words)
    codewords = []

    while uncovered:
        if time.time() > deadline:
            break

        best_word = None
        best_count = 0
        for w in all_words:
            cnt = len(coverage[w] & uncovered)
            if cnt > best_count:
                best_count = cnt
                best_word = w
                if cnt == len(uncovered):
                    break

        if best_word is None or best_count == 0:
            break

        codewords.append(list(best_word))
        uncovered -= coverage[best_word]

    # If we didn't cover everything (due to deadline), try to add more
    if uncovered and time.time() < deadline:
        # Add remaining uncovered words as codewords (shouldn't happen normally)
        for w in uncovered:
            if time.time() > deadline:
                break
            codewords.append(list(w))

    return codewords


def _greedy_cover_general(q, n, radius, deadline):
    """General greedy set-cover (slower, for non-specific cases)."""
    all_words = list(itertools.product(range(q), repeat=n))
    total = q ** n

    # For small spaces, do greedy
    if total <= 20000:
        def word_to_idx(w):
            idx = 0
            for v in w:
                idx = idx * q + v
            return idx

        coverage = {}
        for w in all_words:
            covered = set()
            # Generate all words within distance radius
            for delta in itertools.product(range(q), repeat=n):
                dist = sum(1 for i in range(n) if w[i] != delta[i])
                if dist <= radius:
                    covered.add(delta)
            coverage[w] = covered

        uncovered = set(all_words)
        codewords = []

        while uncovered and time.time() < deadline:
            best_word = None
            best_count = 0
            for w in all_words:
                cnt = len(coverage[w] & uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_word = w

            if best_word is None or best_count == 0:
                break

            codewords.append(list(best_word))
            uncovered -= coverage[best_word]

        return codewords
    else:
        # Too large for greedy; use direct sum or trivial
        if n >= 4 and q == 3 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [[0] * n]
# EVOLVE-BLOCK-END
