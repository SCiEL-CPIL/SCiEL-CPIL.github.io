"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    # Direct sum of the ternary tetracode (9 words of length 4, a perfect code of covering
    # radius 1) with every word on the other n - 4 coordinates: 9 * 3**(n - 4) words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = q ** n

    # Map each word to an index
    def word_to_idx(word):
        idx = 0
        for v in word:
            idx = idx * q + v
        return idx

    # Precompute the ball (covered words) for a codeword as a list of indices
    def ball_indices(cw):
        inds = [word_to_idx(cw)]
        if radius >= 1:
            for pos in range(n):
                orig = cw[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    lst = list(cw)
                    lst[pos] = new_val
                    inds.append(word_to_idx(lst))
        return inds

    # Build coverage count: for each word in the space, how many codewords cover it
    def build_coverage_count(cw_list):
        counts = [0] * total_words
        for cw in cw_list:
            for idx in ball_indices(cw):
                counts[idx] += 1
        return counts

    # Check if removing a codeword keeps coverage valid (vectorized over the ball)
    def is_redundant(cw, counts):
        return bool(np.all(np.asarray(counts, dtype=np.int32)[np.asarray(ball_indices(cw), dtype=np.int64)] > 1))

    # --- Phase 1: fast greedy removal with O(1) delta count updates ---
    counts = build_coverage_count(codewords)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() > deadline:
                return codewords
            if is_redundant(codewords[i], counts):
                # Remove codeword i: decrement counts for its ball
                for idx in ball_indices(codewords[i]):
                    counts[idx] -= 1
                codewords.pop(i)
                improved = True
                break  # Restart the scan from the end

    # --- Phase 2: swap-based local search ---
    # Try to replace a codeword with another that covers more uniquely-covered words
    # while keeping all words covered.
    if time.time() < deadline:
        # Rebuild counts to be safe
        counts = build_coverage_count(codewords)
        # Precompute balls for current codewords
        balls = [ball_indices(cw) for cw in codewords]
        # Fast distinct-word membership set for swap candidates
        cw_set = set(tuple(c) for c in codewords)

        improved_swap = True
        while improved_swap and time.time() < deadline:
            improved_swap = False
            for i in range(len(codewords)):
                if time.time() > deadline:
                    break
                cw_old_t = tuple(codewords[i])
                for pos in range(n):
                    if time.time() > deadline:
                        break
                    orig = codewords[i][pos]
                    for new_val in range(q):
                        if new_val == orig:
                            continue
                        new_cw = list(codewords[i])
                        new_cw[pos] = new_val
                        new_cw_t = tuple(new_cw)
                        if new_cw_t in cw_set:
                            continue
                        ball_i_set = set(balls[i])
                        ball_new = ball_indices(new_cw)
                        ball_new_set = set(ball_new)
                        valid = True
                        for idx in ball_i_set:
                            if idx not in ball_new_set:
                                if counts[idx] <= 1:
                                    valid = False
                                    break
                        if not valid:
                            continue
                        for idx in ball_i_set:
                            counts[idx] -= 1
                        for idx in ball_new_set:
                            counts[idx] += 1
                        codewords[i] = new_cw
                        balls[i] = ball_new
                        cw_set.discard(cw_old_t)
                        cw_set.add(tuple(new_cw))
                        improved_swap = True
                        break
                    if improved_swap:
                        break
                if improved_swap:
                    break
            if not improved_swap:
                # No single-symbol swap helped; try two-symbol swaps
                cw_set = set(tuple(c) for c in codewords)
                for i in range(len(codewords)):
                    if time.time() > deadline:
                        break
                    for pos1 in range(n):
                        if time.time() > deadline:
                            break
                        for pos2 in range(pos1 + 1, n):
                            if time.time() > deadline:
                                break
                            orig1 = codewords[i][pos1]
                            orig2 = codewords[i][pos2]
                            for v1 in range(q):
                                if v1 == orig1:
                                    continue
                                for v2 in range(q):
                                    if v2 == orig2:
                                        continue
                                    new_cw = list(codewords[i])
                                    new_cw[pos1] = v1
                                    new_cw[pos2] = v2
                                    new_cw_t = tuple(new_cw)
                                    if new_cw_t in cw_set:
                                        continue
                                    ball_i_set = set(balls[i])
                                    ball_new = ball_indices(new_cw)
                                    ball_new_set = set(ball_new)
                                    valid = True
                                    for idx in ball_i_set:
                                        if idx not in ball_new_set:
                                            if counts[idx] <= 1:
                                                valid = False
                                                break
                                    if not valid:
                                        continue
                                    for idx in ball_i_set:
                                        counts[idx] -= 1
                                    for idx in ball_new_set:
                                        counts[idx] += 1
                                    codewords[i] = new_cw
                                    balls[i] = ball_new
                                    cw_set.discard(cw_old_t)
                                    cw_set.add(tuple(new_cw))
                                    improved_swap = True
                                    break
                                if improved_swap:
                                    break
                            if improved_swap:
                                break
                        if improved_swap:
                            break
                    if improved_swap:
                        break

    # --- Phase 3: final greedy removal pass ---
    if time.time() < deadline:
        counts = build_coverage_count(codewords)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(codewords) - 1, -1, -1):
                if time.time() > deadline:
                    return codewords
                if is_redundant(codewords[i], counts):
                    for idx in ball_indices(codewords[i]):
                        counts[idx] -= 1
                    codewords.pop(i)
                    improved = True
                    break

    # --- Phase 4: random-restart greedy removal ---
    if time.time() < deadline:
        rng = random.Random(12345)
        best = list(codewords)
        best_len = len(best)
        while time.time() < deadline - 2.0:
            trial = list(codewords)
            rng.shuffle(trial)
            counts = build_coverage_count(trial)
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                for i in range(len(trial) - 1, -1, -1):
                    if is_redundant(trial[i], counts):
                        for idx in ball_indices(trial[i]):
                            counts[idx] -= 1
                        trial.pop(i)
                        improved = True
                        break
            if len(trial) < best_len:
                best = trial
                best_len = len(trial)

    # --- Phase 5: add-then-remove to escape local minima ---
    # Add a random codeword, then greedily remove redundant ones.
    # This can allow the code to "move" to a different basin of attraction.
    if time.time() < deadline - 5.0:
        rng = random.Random(99999)
        best = list(codewords)
        best_len = len(best)
        attempts = 0
        while time.time() < deadline - 5.0 and attempts < 50:
            attempts += 1
            trial = list(best)
            # Add a random word not already in the code
            for _ in range(100):
                rand_word = [rng.randint(0, q - 1) for _ in range(n)]
                if tuple(rand_word) not in set(tuple(c) for c in trial):
                    trial.append(rand_word)
                    break
            else:
                continue
            # Now greedily remove redundant codewords
            counts = build_coverage_count(trial)
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                for i in range(len(trial) - 1, -1, -1):
                    if is_redundant(trial[i], counts):
                        for idx in ball_indices(trial[i]):
                            counts[idx] -= 1
                        trial.pop(i)
                        improved = True
                        break
            if len(trial) < best_len:
                best = trial
                best_len = len(trial)

    return best
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
