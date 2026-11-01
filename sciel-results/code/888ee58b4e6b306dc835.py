# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # For the specific case of q=3, n=8, radius=1
    # We use a greedy covering code algorithm with vectorized operations

    # First, generate the tetracode (perfect code of length 4, radius 1 over GF(3))
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

    # Initialize with the direct sum construction (baseline: 9 * 3^4 = 729 words)
    # But we'll try to do better with a greedy approach

    # For q=3, n=8, radius=1: each codeword covers 1 + 8*2 = 17 words
    # Total space: 3^8 = 6561
    # Lower bound: ceil(6561/17) = 387
    # Best known: 486

    # Greedy covering code algorithm
    # Represent all words as integers 0..3^8-1
    total_words = 3 ** n
    ball_size = 1 + n * (q - 1)  # For radius 1: 1 + 8*2 = 17

    # Precompute balls for each possible codeword position
    # A ball of radius 1 around word w includes w itself and all words at distance 1
    # To check if a word is in a ball, we need to compute Hamming distance

    # For efficiency, represent words as base-3 digits
    def word_to_int(w):
        result = 0
        for digit in w:
            result = result * q + digit
        return result

    def int_to_word(x, n, q):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = x % q
            x //= q
        return w

    # Precompute all balls (which words are covered by each codeword)
    # This is expensive but we can do it incrementally

    # Better approach: use a greedy algorithm that picks the codeword covering the most uncovered words
    # Track uncovered words with a boolean array

    covered = np.zeros(total_words, dtype=bool)
    codewords = []

    # For the greedy step, we need to evaluate which candidate codeword covers the most uncovered words
    # Candidates: all possible words (3^8 = 6561)

    # Precompute the ball membership for each candidate
    # ball_mask[i] is a boolean array of length total_words indicating which words are in the ball around candidate i

    # This is too memory intensive: 6561 * 6561 booleans ~ 43 MB, which is okay

    # Actually, let's be smarter. For each candidate, the ball consists of:
    # - The candidate itself
    # - For each position j and each value v != candidate[j], the word with that position changed to v

    # Let's precompute for each candidate the list of words in its ball
    # But storing 6561 lists of 17 integers each is fine

    # Convert all words to integers
    all_words_int = list(range(total_words))

    # Precompute balls: for each candidate, which words are in its ball
    # We'll store as a list of sets or use a different approach

    # Alternative: for the greedy step, for each uncovered word, find which candidate covers it
    # and count how many uncovered words each candidate covers

    # Let's use the approach: for each candidate, compute the set of words in its ball
    # Then in each greedy step, count uncovered words in each ball

    # Precompute balls as numpy arrays
    # balls[i] contains the integer indices of words in the ball around candidate i

    balls = []
    for x in range(total_words):
        w = int_to_word(x, n, q)
        ball = [x]
        for j in range(n):
            for v in range(q):
                if v != w[j]:
                    w2 = w[:]
                    w2[j] = v
                    ball.append(word_to_int(w2))
        balls.append(ball)

    # Now greedy: repeatedly pick the candidate that covers the most uncovered words
    remaining = np.ones(total_words, dtype=bool)  # True if uncovered
    codewords = []

    while remaining.any():
        if time.time() > deadline - 5:
            break

        # For each candidate, count how many uncovered words it covers
        # This is O(num_candidates * ball_size) = 6561 * 17 ~ 111K operations per step
        # With ~387+ steps, that's ~43M operations, should be fast enough

        best_candidate = -1
        best_count = 0
        for i in range(total_words):
            if not remaining[i]:
                # Skip if this candidate is already covered (it doesn't add new coverage for itself)
                # Actually we should still consider it if it covers other uncovered words
                pass

            # Count uncovered words in ball i
            count = 0
            for word_idx in balls[i]:
                if remaining[word_idx]:
                    count += 1

            if count > best_count:
                best_count = count
                best_candidate = i

        if best_candidate == -1:
            break

        # Add this codeword
        w = int_to_word(best_candidate, n, q)
        codewords.append(w)

        # Mark all words in its ball as covered
        for word_idx in balls[best_candidate]:
            remaining[word_idx] = False

    # If we timed out and didn't cover everything, fall back to a valid construction
    if remaining.any():
        # Use the direct sum construction as fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return codewords

# EVOLVE-BLOCK-END
