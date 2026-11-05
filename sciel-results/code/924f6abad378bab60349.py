"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    total_words = q ** n

    # Precompute all ball index lists for every possible codeword (q^n of them)
    # For q=3, n=6, radius=1: 729 balls of size 1+6*2=13 each -> 729*13 = 9477 ints, tiny
    all_balls = []
    for idx in range(total_words):
        # Convert idx to word
        word = []
        tmp = idx
        for _ in range(n):
            word.append(tmp % q)
            tmp //= q
        word.reverse()
        ball = [idx]
        if radius >= 1:
            for pos in range(n):
                orig = word[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    # Compute new index efficiently
                    # word[pos] changes from orig to new_val
                    # idx = sum(word[j] * q^(n-1-j))
                    # delta = (new_val - orig) * q^(n-1-pos)
                    delta = (new_val - orig) * (q ** (n - 1 - pos))
                    ball.append(idx + delta)
        all_balls.append(tuple(ball))

    # Ball for a given codeword list
    def word_to_idx(word):
        idx = 0
        for v in word:
            idx = idx * q + v
        return idx

    def ball_indices(cw):
        return all_balls[word_to_idx(cw)]

    # Build coverage count array (numpy for speed)
    def build_coverage_count(cw_list):
        counts = np.zeros(total_words, dtype=np.int32)
        for cw in cw_list:
            b = all_balls[word_to_idx(cw)]
            counts[np.asarray(b, dtype=np.int64)] += 1
        return counts

    # Check if removing a codeword keeps coverage valid
    def is_redundant(cw, counts):
        b = all_balls[word_to_idx(cw)]
        return bool(np.all(counts[np.asarray(b, dtype=np.int64)] > 1))

    # Initial code: direct sum of ternary tetracode with all tails
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # --- Phase 1: fast greedy removal ---
    counts = build_coverage_count(codewords)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() > deadline:
                return codewords
            if is_redundant(codewords[i], counts):
                b = np.asarray(all_balls[word_to_idx(codewords[i])], dtype=np.int64)
                counts[b] -= 1
                codewords.pop(i)
                improved = True
                break

    # --- Phase 2: swap-based local search (numpy-vectorized) ---
    if time.time() < deadline:
        counts = build_coverage_count(codewords)
        balls_arr = [np.asarray(all_balls[word_to_idx(cw)], dtype=np.int64) for cw in codewords]
        cw_set = set(tuple(c) for c in codewords)

        improved_swap = True
        while improved_swap and time.time() < deadline:
            improved_swap = False
            for i in range(len(codewords)):
                if time.time() > deadline:
                    break
                cw_old_t = tuple(codewords[i])
                # Single-position swaps
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
                        new_idx = word_to_idx(new_cw)
                        ball_new_arr = np.asarray(all_balls[new_idx], dtype=np.int64)
                        # Check validity: words only covered by old codeword must be in new ball
                        old_ball_arr = balls_arr[i]
                        # Words in old ball not in new ball
                        mask = np.isin(old_ball_arr, ball_new_arr)
                        uncovered = old_ball_arr[~mask]
                        if len(uncovered) > 0 and np.any(counts[uncovered] <= 1):
                            continue
                        # Apply swap
                        counts[old_ball_arr] -= 1
                        counts[ball_new_arr] += 1
                        codewords[i] = new_cw
                        balls_arr[i] = ball_new_arr
                        cw_set.discard(cw_old_t)
                        cw_set.add(new_cw_t)
                        improved_swap = True
                        break
                    if improved_swap:
                        break
                if improved_swap:
                    break
            if not improved_swap:
                # Two-position swaps
                cw_set = set(tuple(c) for c in codewords)
                for i in range(len(codewords)):
                    if time.time() > deadline:
                        break
                    cw_old_t = tuple(codewords[i])
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
                                    new_idx = word_to_idx(new_cw)
                                    ball_new_arr = np.asarray(all_balls[new_idx], dtype=np.int64)
                                    old_ball_arr = balls_arr[i]
                                    mask = np.isin(old_ball_arr, ball_new_arr)
                                    uncovered = old_ball_arr[~mask]
                                    if len(uncovered) > 0 and np.any(counts[uncovered] <= 1):
                                        continue
                                    counts[old_ball_arr] -= 1
                                    counts[ball_new_arr] += 1
                                    codewords[i] = new_cw
                                    balls_arr[i] = ball_new_arr
                                    cw_set.discard(cw_old_t)
                                    cw_set.add(new_cw_t)
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
                    b = np.asarray(all_balls[word_to_idx(codewords[i])], dtype=np.int64)
                    counts[b] -= 1
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
                        b = np.asarray(all_balls[word_to_idx(trial[i])], dtype=np.int64)
                        counts[b] -= 1
                        trial.pop(i)
                        improved = True
                        break
            if len(trial) < best_len:
                best = trial
                best_len = len(trial)

    # --- Phase 5: add-then-remove to escape local minima ---
    if time.time() < deadline - 5.0:
        rng = random.Random(99999)
        best = list(codewords)
        best_len = len(best)
        attempts = 0
        while time.time() < deadline - 5.0 and attempts < 50:
            attempts += 1
            trial = list(best)
            trial_set = set(tuple(c) for c in trial)
            for _ in range(100):
                rand_word = [rng.randint(0, q - 1) for _ in range(n)]
                if tuple(rand_word) not in trial_set:
                    trial.append(rand_word)
                    break
            else:
                continue
            counts = build_coverage_count(trial)
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                for i in range(len(trial) - 1, -1, -1):
                    if is_redundant(trial[i], counts):
                        b = np.asarray(all_balls[word_to_idx(trial[i])], dtype=np.int64)
                        counts[b] -= 1
                        trial.pop(i)
                        improved = True
                        break
            if len(trial) < best_len:
                best = trial
                best_len = len(trial)

    return best
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
