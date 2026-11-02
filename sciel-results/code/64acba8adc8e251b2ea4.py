"""Covering code K_3(8,1) via greedy + local search with numpy optimization."""
import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    total_words = q ** n

    # Precompute all words and their integer indices
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_int = {w: idx for idx, w in enumerate(all_words)}

    # Precompute balls (each word's radius-1 neighborhood) as numpy arrays
    balls = []
    for idx, w in enumerate(all_words):
        ball = [idx]
        for pos in range(n):
            for new_val in range(q):
                if new_val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = new_val
                ball.append(word_to_int[tuple(nw)])
        balls.append(np.array(ball, dtype=np.int32))

    # Precompute ball membership: for each word, which balls contain it
    # This is the transpose of the ball matrix
    # ball_matrix[i, j] = 1 if word j is in ball of word i
    # We can build this efficiently
    # For redundancy check: word w is covered by codeword set C if any c in C has w in ball(c)
    # Equivalent: coverage[w] = sum over c in C of [w in ball(c)]

    # For greedy: at each step, for candidate c, gain = number of uncovered words in ball(c)
    # Use numpy: gain = np.sum(uncovered[balls[c]])

    # Greedy set cover
    uncovered = np.ones(total_words, dtype=bool)
    chosen = []
    chosen_set = set()

    # Speed up: precompute balls as a list, and for the greedy loop,
    # we need to evaluate gain for each candidate. With 6561 candidates and 17-element balls,
    # each gain computation is O(17), so total per step is O(6561*17) ~ 111K ops
    # For ~400 steps, that's ~45M ops, which should be fine in numpy

    for step in range(total_words):
        if time.time() > deadline - 10:
            break
        # Find best candidate
        best_idx = -1
        best_count = -1
        # Use numpy to compute gains for all non-chosen candidates
        # Create a mask of non-chosen
        # For efficiency, just iterate but use numpy for the sum
        for idx in range(total_words):
            if idx in chosen_set:
                continue
            gain = np.sum(uncovered[balls[idx]])
            if gain > best_count:
                best_count = gain
                best_idx = idx
            if best_count == 17:
                break
        if best_idx < 0 or best_count <= 0:
            break
        chosen.append(best_idx)
        chosen_set.add(best_idx)
        uncovered[balls[best_idx]] = False

    def int_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        return list(reversed(w))

    # Redundancy removal using numpy
    # Build a coverage matrix: cov[w] = number of chosen codewords whose ball contains w
    # For each word w, cov[w] = sum over c in chosen of [w in balls[c]]
    # We can compute this by: for each c in chosen, cov[balls[c]] += 1
    cov = np.zeros(total_words, dtype=np.int32)
    for c in chosen:
        cov[balls[c]] += 1

    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        for i in range(len(chosen) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            # Check if removing chosen[i] leaves all words in its ball still covered
            # A word w in balls[chosen[i]] is still covered if cov[w] >= 2 (since removing one)
            # Actually: after removing, cov[w] decreases by 1 for w in balls[chosen[i]]
            # So w is still covered if cov[w] >= 2
            ball_i = balls[chosen[i]]
            if np.all(cov[ball_i] >= 2):
                # Redundant
                cov[ball_i] -= 1
                chosen.pop(i)
                changed = True

    best_code = [int_to_word(idx) for idx in chosen]
    best_size = len(chosen)

    # Randomized greedy restarts
    np.random.seed(42)
    while time.time() < deadline - 5:
        uncovered = np.ones(total_words, dtype=bool)
        chosen_r = []
        chosen_set_r = set()

        for step in range(total_words):
            if time.time() > deadline - 5:
                break
            # Sample random candidates
            candidates = np.random.permutation(total_words)[:300]
            best_idx = -1
            best_count = -1
            for idx in candidates:
                if idx in chosen_set_r:
                    continue
                gain = np.sum(uncovered[balls[idx]])
                if gain > best_count:
                    best_count = gain
                    best_idx = idx
                if best_count == 17:
                    break
            if best_idx < 0 or best_count <= 0:
                break
            chosen_r.append(best_idx)
            chosen_set_r.add(best_idx)
            uncovered[balls[best_idx]] = False

        # Redundancy removal
        cov = np.zeros(total_words, dtype=np.int32)
        for c in chosen_r:
            cov[balls[c]] += 1
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(len(chosen_r) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                ball_i = balls[chosen_r[i]]
                if np.all(cov[ball_i] >= 2):
                    cov[ball_i] -= 1
                    chosen_r.pop(i)
                    changed = True

        if len(chosen_r) < best_size:
            best_size = len(chosen_r)
            best_code = [int_to_word(idx) for idx in chosen_r]

        if best_size <= 403:
            break

    return best_code
