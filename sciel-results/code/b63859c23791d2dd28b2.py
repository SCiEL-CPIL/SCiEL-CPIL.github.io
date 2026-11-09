"""Improved program for ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random


def construct(q, n, radius):
    deadline = time.time() + 140  # safety margin before 150s kill
    rng = np.random.RandomState(42)

    # Precompute all words as indices 0..3^7-1
    total_words = q ** n  # 2187

    def idx_to_word(idx):
        w = np.zeros(n, dtype=np.int8)
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    def word_to_idx(w):
        idx = 0
        for i in range(n - 1, -1, -1):
            idx = idx * q + w[i]
        return idx

    # Precompute: for each word, the set of words within distance radius
    # For radius=1, that's the word itself + all single-position changes
    # We'll work with numpy arrays for speed

    # Generate all words
    all_words = np.array([idx_to_word(i) for i in range(total_words)], dtype=np.int8)

    # For each word, compute its "ball" (words within distance 1)
    # Ball size = 1 + n*(q-1) = 1 + 7*2 = 15 for q=3, n=7
    ball_size = 1 + n * (q - 1)
    balls = np.zeros((total_words, ball_size), dtype=np.int8)
    for i in range(total_words):
        w = all_words[i]
        balls[i, 0] = w
        pos = 1
        for coord in range(n):
            for val in range(q):
                if val != w[coord]:
                    nw = w.copy()
                    nw[coord] = val
                    balls[i, pos] = nw
                    pos += 1

    # Convert balls to indices for faster set operations
    ball_indices = np.zeros((total_words, ball_size), dtype=np.int32)
    for i in range(total_words):
        for j in range(ball_size):
            ball_indices[i, j] = word_to_idx(balls[i, j])

    # Greedy construction: repeatedly pick the word whose ball covers the most uncovered words
    covered = np.zeros(total_words, dtype=bool)
    codewords = []

    def count_uncovered():
        return (~covered).sum()

    # Greedy: pick word with max uncovered coverage
    # Use numpy for speed
    while count_uncovered() > 0:
        if time.time() > deadline - 30:
            break
        # For each candidate word, count how many uncovered words are in its ball
        # This is expensive, so we do it in batches
        uncovered_mask = ~covered
        best_idx = -1
        best_count = 0
        # Sample candidates if too slow, but for 2187 words it should be fine
        for i in range(total_words):
            if covered[i]:
                continue
            ball = ball_indices[i]
            cnt = uncovered_mask[ball].sum()
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_idx < 0:
            break
        codewords.append(all_words[best_idx].tolist())
        covered[ball_indices[best_idx]] = True

    # Local search: try to remove codewords that don't reduce coverage
    improved = True
    while improved:
        if time.time() > deadline - 20:
            break
        improved = False
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() > deadline - 20:
                break
            # Check if removing codeword i still leaves all words covered
            # Temporarily remove and check
            test_covered = np.zeros(total_words, dtype=bool)
            for j, cw in enumerate(codewords):
                if j == i:
                    continue
                idx = word_to_idx(cw)
                test_covered[ball_indices[idx]] = True
            if test_covered.all():
                codewords.pop(i)
                improved = True
                break

    # Add-Remove local search: try replacing one codeword with another to improve
    # Try swapping: remove one, add one that covers more uncovered
    for _ in range(50):  # limited iterations
        if time.time() > deadline - 10:
            break
        found_improvement = False
        for i in range(len(codewords)):
            if time.time() > deadline - 10:
                break
            # Remove codeword i, find uncovered words
            test_covered = np.zeros(total_words, dtype=bool)
            for j, cw in enumerate(codewords):
                if j == i:
                    continue
                idx = word_to_idx(cw)
                test_covered[ball_indices[idx]] = True
            uncovered_mask = ~test_covered
            unc_count = uncovered_mask.sum()
            if unc_count == 0:
                # Can remove entirely
                codewords.pop(i)
                found_improvement = True
                break
            # Find best replacement
            best_new = -1
            best_gain = 0
            for k in range(total_words):
                ball = ball_indices[k]
                gain = uncovered_mask[ball].sum()
                if gain > best_gain:
                    best_gain = gain
                    best_new = k
            if best_new >= 0 and best_gain > unc_count:
                # Replacing doesn't help if we just add without removing
                # Actually we should check: does adding best_new cover all uncovered?
                # If yes, we can remove i
                new_covered = test_covered.copy()
                new_covered[ball_indices[best_new]] = True
                if new_covered.all():
                    codewords[i] = all_words[best_new].tolist()
                    found_improvement = True
                    break
        if not found_improvement:
            break

    # Final greedy removal pass
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            test_covered = np.zeros(total_words, dtype=bool)
            for j, cw in enumerate(codewords):
                if j == i:
                    continue
                idx = word_to_idx(cw)
                test_covered[ball_indices[idx]] = True
            if test_covered.all():
                codewords.pop(i)
                improved = True
                break

    return codewords
# EVOLVE-BLOCK-END
