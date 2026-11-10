import time
import numpy as np

def construct(q, n, radius):
    # Deadline guard: 150s limit, aim for 115s safety margin
    start_time = time.time()
    deadline = start_time + 115.0

    # For this task: q=3, n=7, radius=1
    # Total space: 3^7 = 2187
    # Ball size: 1 + n*(q-1) = 15

    total_words = q ** n
    ball_size = 1 + n * (q - 1)

    # Convert int to base-q digits
    def int_to_digits(x):
        d = [0] * n
        for i in range(n):
            d[i] = x % q
            x //= q
        return d

    def digits_to_int(d):
        x = 0
        for i in range(n - 1, -1, -1):
            x = x * q + d[i]
        return x

    # Precompute balls: balls[i] = list of indices within radius of i
    balls = [None] * total_words
    for i in range(total_words):
        d = int_to_digits(i)
        ball = [i]
        if radius >= 1:
            for pos in range(n):
                orig = d[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    d[pos] = new_val
                    ball.append(digits_to_int(d))
                d[pos] = orig
        balls[i] = ball

    # Build adjacency matrix M where M[i,j] = 1 if j in balls[i]
    M = np.zeros((total_words, total_words), dtype=np.uint8)
    for i in range(total_words):
        for j in balls[i]:
            M[i, j] = 1

    # Phase 1: Greedy set cover
    covered = np.zeros(total_words, dtype=bool)
    codewords = []
    uncovered_count = np.ones(total_words, dtype=np.uint8)

    # Use numpy to compute initial scores
    while time.time() < deadline - 30:
        if not np.any(uncovered_count > 0):
            break

        # Score for each candidate: how many uncovered words it covers
        # M @ uncovered_mask where uncovered_mask = (uncovered_count > 0)
        uncovered_mask = (uncovered_count > 0).astype(np.uint8)
        scores = M.dot(uncovered_mask)
        best_idx = int(np.argmax(scores))

        if scores[best_idx] == 0:
            break

        codewords.append(best_idx)
        # Update covered
        ball = balls[best_idx]
        covered[ball] = True
        uncovered_count[ball] -= 1
        uncovered_count[ball] = np.clip(uncovered_count[ball], 0, 255)

    # Phase 2: Iterative pruning of redundant codewords
    # cover_count[j] = number of codewords covering j
    cover_count = np.zeros(total_words, dtype=np.int16)
    for c in codewords:
        cover_count[balls[c]] += 1

    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for idx in range(len(codewords)):
            c = codewords[idx]
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                codewords.pop(idx)
                improved = True
                break

    # Phase 3: Simulated annealing with swaps to reduce size
    # We'll try to replace codewords to enable further pruning
    best_codewords = list(codewords)
    best_size = len(codewords)

    if time.time() < deadline - 5:
        T = 50.0
        T_min = 0.1
        cooling = 0.98
        rng = np.random.default_rng(42)

        current_codewords = list(codewords)
        current_cover_count = cover_count.copy()

        while T > T_min and time.time() < deadline - 5:
            # Try a random swap: pick a codeword to replace
            if len(current_codewords) == 0:
                break

            idx = rng.integers(0, len(current_codewords))
            c_old = current_codewords[idx]

            # Pick a random candidate not in current set
            c_new = rng.integers(0, total_words)
            while c_new in current_codewords:
                c_new = rng.integers(0, total_words)

            # Compute energy change
            # E = size + lambda * uncovered (uncovered should be 0 for valid solutions)
            # We only accept if coverage is maintained
            ball_old = balls[c_old]
            ball_new = balls[c_new]

            # Check if removing c_old would leave any word uncovered
            # i.e., for all w in ball_old, cover_count[w] - 1 + (1 if w in ball_new) >= 1
            # Simplify: for all w in ball_old, if w not in ball_new, then cover_count[w] > 1

            valid_swap = True
            for w in ball_old:
                if w not in ball_new:
                    if current_cover_count[w] <= 1:
                        valid_swap = False
                        break

            if valid_swap:
                # Accept the swap
                current_cover_count[ball_old] -= 1
                current_cover_count[ball_new] += 1
                current_codewords[idx] = c_new

                # Try to prune
                pruned = True
                while pruned:
                    pruned = False
                    for pidx in range(len(current_codewords)):
                        pc = current_codewords[pidx]
                        if np.min(current_cover_count[balls[pc]]) > 1:
                            current_cover_count[balls[pc]] -= 1
                            current_codewords.pop(pidx)
                            pruned = True
                            break

                if len(current_codewords) < best_size:
                    best_size = len(current_codewords)
                    best_codewords = list(current_codewords)
                    # Reset temperature occasionally
                    T = max(T * cooling, T_min)
                else:
                    T *= cooling
            else:
                T *= cooling

        # Final prune on best solution
        final_codewords = list(best_codewords)
        final_cover_count = np.zeros(total_words, dtype=np.int16)
        for c in final_codewords:
            final_cover_count[balls[c]] += 1

        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            for idx in range(len(final_codewords)):
                c = final_codewords[idx]
                if np.min(final_cover_count[balls[c]]) > 1:
                    final_cover_count[balls[c]] -= 1
                    final_codewords.pop(idx)
                    changed = True
                    break

        if len(final_codewords) < len(codewords):
            codewords = final_codewords

    # Convert to list of digit lists
    result = []
    for c in codewords:
        result.append(int_to_digits(c))

    return result
