"""Greedy set-cover for ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # For radius=1, the ball around word w is {w} ∪ {words differing in exactly 1 position}
    # Total words in ball: 1 + n*(q-1)
    # For q=3, n=7: 1 + 7*2 = 15 words per ball

    total_words = q ** n
    ball_size = 1 + n * (q - 1)

    # Convert word index to list of digits
    def idx_to_word(idx):
        word = []
        for i in range(n):
            word.append(idx % q)
            idx //= q
        return word

    # Convert word (list of digits) to index
    def word_to_idx(word):
        idx = 0
        for i in range(n - 1, -1, -1):
            idx = idx * q + word[i]
        return idx

    # Precompute all neighbor indices (ball of radius 1) for each word
    # neighbors[idx] = list of indices in the ball around word idx
    neighbors = []
    for idx in range(total_words):
        word = idx_to_word(idx)
        ball = [idx]
        for pos in range(n):
            for val in range(q):
                if val != word[pos]:
                    new_word = word[:]
                    new_word[pos] = val
                    ball.append(word_to_idx(new_word))
        neighbors.append(ball)

    # Greedy set cover
    # uncovered: set of uncovered word indices
    uncovered = set(range(total_words))
    codewords = []

    # For efficiency, track how many uncovered words each candidate covers
    # We'll use a simple approach: at each step, evaluate candidates
    # But evaluating all candidates each step is O(total_words * ball_size) which is expensive

    # Better approach: maintain coverage counts
    # cover_count[w] = number of uncovered words in the ball around w
    cover_count = np.zeros(total_words, dtype=np.int32)
    for idx in range(total_words):
        cover_count[idx] = ball_size  # initially all covered by their own ball

    # Actually, let's think more carefully.
    # When we pick a word w, we remove all words in ball(w) from uncovered.
    # For each remaining candidate c, cover_count[c] decreases by |ball(c) ∩ ball(w) ∩ uncovered_before|

    # This is complex. Let's use a simpler but still effective approach:
    # At each step, sample a subset of candidates and pick the best one.

    # Actually, for total_words=2187, we can afford to check all candidates each step.
    # 2187 candidates * 15 ball size = ~33K operations per step
    # If we need ~100-200 steps, that's ~3-6M operations, which should be fast.

    while uncovered and time.time() < deadline:
        best_candidate = -1
        best_count = -1

        # Sample candidates if there are too many to check all
        # For 2187 words, checking all is fine
        for cand in uncovered:
            count = 0
            for nb in neighbors[cand]:
                if nb in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_candidate = cand

        if best_candidate < 0:
            break

        codewords.append(best_candidate)
        # Remove all words in ball(best_candidate) from uncovered
        for nb in neighbors[best_candidate]:
            uncovered.discard(nb)

    # Convert indices to words
    result = [idx_to_word(cw) for cw in codewords]

    # If we ran out of time, return what we have (may be incomplete)
    # But we need a valid solution. If incomplete, fall back to direct sum.
    if len(uncovered) > 0:
        # Fallback: use the direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        result = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return result
# EVOLVE-BLOCK-END
