import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # for radius=1

    # Precompute all words as integer indices
    def int_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute balls: for each index, the list of indices in its radius-1 ball
    # ball_indices[i] = np.array of indices covered by choosing codeword i
    ball_indices = np.empty(total_words, dtype=object)
    for i in range(total_words):
        w = int_to_word(i)
        ball = [i]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                w[pos] = val
                # compute index
                idx = 0
                for j in range(n):
                    idx = idx * q + w[j]
                ball.append(idx)
            w[pos] = orig
        ball_indices[i] = np.array(ball, dtype=np.int32)

    uncovered = np.ones(total_words, dtype=np.int8)
    code = []

    # Greedy set cover
    while True:
        if time.time() > deadline:
            break

        remaining = uncovered.sum()
        if remaining == 0:
            break

        # Find the candidate that covers the most uncovered words
        max_cover = 0
        best_candidate = -1

        # Vectorized: compute coverage for all candidates at once
        # For each candidate c, coverage = sum of uncovered[ball_indices[c]]
        # We can do this with a loop but vectorize the inner sum
        # Actually, let's just loop but it's 729 iterations each with 13 elements
        
        # Try to vectorize: create a 2D array of all ball indices
        # all_balls = np.vstack([ball_indices[i] for i in range(total_words)])
        # coverage = uncovered[all_balls].reshape(total_words, ball_size).sum(axis=1)
        # But this requires 729*13 = 9477 elements, which is fine
        
        # Precompute the full ball matrix once
        all_balls = np.empty((total_words, ball_size), dtype=np.int32)
        for i in range(total_words):
            all_balls[i] = ball_indices[i]

        coverage = uncovered[all_balls].sum(axis=1)
        best_candidate = int(np.argmax(coverage))
        max_cover = int(coverage[best_candidate])

        if max_cover == 0:
            break

        code.append(int_to_word(best_candidate))
        uncovered[all_balls[best_candidate]] = 0

    return code
