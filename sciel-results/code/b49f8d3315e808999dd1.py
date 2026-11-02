import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)
    return _construct_general(q, n, radius, deadline)


def _construct_3_7_1(deadline):
    N = 2187

    # Precompute digits
    digits = np.zeros((N, 7), dtype=np.int8)
    for i in range(N):
        tmp = i
        for pos in range(7):
            digits[i, pos] = tmp % 3
            tmp //= 3

    # Build ball matrix as a single 2D int8 array
    # ball_mat[i, j] = 1 if j is within Hamming distance 1 of i
    ball_mat = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        ball_mat[i, i] = 1
        for pos in range(7):
            orig = digits[i, pos]
            for val in range(3):
                if val == orig:
                    continue
                # j has same digits as i except at pos, and digit[pos] = val
                # j = i + (val - orig) * 3**pos
                j = i + (val - orig) * (3 ** pos)
                ball_mat[i, j] = 1

    # For candidate search: we need, for each word w, the set of indices j
    # such that j is within distance 1 of w. That's row ball_mat[w].
    # For a set of words W, candidates = intersection of row ball_mat[w] for w in W.
    # With the matrix, this is: np.all(ball_mat[:, W], axis=1)
    # where W is an array of word indices.

    # Precompute 3**pos for fast index arithmetic
    pow3 = np.array([1, 3, 9, 27, 81, 243, 729], dtype=np.int32)

    # Helper: greedy initialization (vectorized)
    def greedy_init(seed):
        rng = random.Random(seed)
        uncovered = np.ones(N, dtype=bool)
        selected = []

        while np.any(uncovered) and time.time() < deadline - 15.0:
            # Count uncovered words covered by each candidate
            # For each i, count = sum of ball_mat[i] over uncovered positions
            counts = ball_mat[~np.isin(range(N), selected)] @ uncovered.astype(np.int8)
            # Actually, let's just compute for all and mask out selected
            all_counts = ball_mat @ uncovered.astype(np.int8)
            for s in selected:
                all_counts[s] = -1
            best_idx = int(np.argmax(all_counts))
            if all_counts[best_idx] <= 0:
                break
            selected.append(best_idx)
            uncovered &= ~ball_mat[best_idx].astype(bool)

        return selected

    # Helper: local search
    def local_search(code, seed):
        rng = random.Random(seed)
        code_set = set(code)

        # Coverage count: number of codewords covering each word
        cnt = np.zeros(N, dtype=np.int32)
        for c in code:
            cnt += ball_mat[c]

        best_code = list(code)
        best_size = len(code)

        while time.time() < deadline - 3.0:
            improved = False

            # Phase 1: Remove redundant codewords
            changed = True
            while changed:
                changed = False
                new_code = []
                for c in code:
                    # c is essential if it uniquely covers some word
                    if np.any((cnt == 1) & ball_mat[c].astype(bool)):
                        new_code.append(c)
                    else:
                        cnt -= ball_mat[c]
                        code_set.discard(c)
                        changed = True
                code = new_code

                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
                    improved = True

            if not code:
                break

            # Phase 2: Try swaps
            swaps_attempted = 0
            max_swaps = 200

            while swaps_attempted < max_swaps and time.time() < deadline - 3.0:
                swaps_attempted += 1
                c = rng.choice(code)

                # Find words uniquely covered by c
                unique_words = np.where((cnt == 1) & ball_mat[c].astype(bool))[0]

                if len(unique_words) == 0:
                    cnt -= ball_mat[c]
                    code.remove(c)
                    code_set.discard(c)
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = list(code)
                    improved = True
                    break

                # Find candidates that cover all unique_words
                # candidate j must be within distance 1 of every word in unique_words
                # i.e., ball_mat[j, w] = 1 for all w in unique_words
                # That's: np.all(ball_mat[:, unique_words], axis=1)
                candidate_mask = np.all(ball_mat[:, unique_words], axis=1)

                # Remove candidates already in code
                for c2 in code:
                    candidate_mask[c2] = False

                candidates = np.where(candidate_mask)[0]

                if len(candidates) == 0:
                    continue

                # Try a few random candidates
                for _ in range(min(5, len(candidates))):
                    c_prime = int(rng.choice(candidates))

                    # Check if swap maintains coverage
                    cnt_new = cnt - ball_mat[c] + ball_mat[c_prime]
                    if np.all(cnt_new > 0):
                        cnt = cnt_new
                        code.remove(c)
                        code_set.discard(c)
                        code.append(c_prime)
                        code_set.add(c_prime)
                        improved = True
                        break

            # Phase 3: Perturbation if no improvement
            if not improved:
                if len(code) > 2:
                    # Remove 2 random codewords and re-add greedily
                    remove_count = min(2, len(code) - 1)
                    to_remove = rng.sample(code, remove_count)
                    for c in to_remove:
                        cnt -= ball_mat[c]
                        code.remove(c)
                        code_set.discard(c)

                    # Re-add greedily to restore coverage
                    while time.time() < deadline - 3.0:
                        uncovered = np.where(cnt == 0)[0]
                        if len(uncovered) == 0:
                            break
                        # Find best candidate
                        all_counts = ball_mat @ (cnt == 0).astype(np.int8)
                        for s in code_set:
                            all_counts[s] = -1
                        best_idx = int(np.argmax(all_counts))
                        if all_counts[best_idx] <= 0:
                            break
                        cnt += ball_mat[best_idx]
                        code.append(best_idx)
                        code_set.add(best_idx)

                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)

        return best_code

    # Main loop: iterated local search
    best_code = None
    best_size = float('inf')

    # Start with greedy
    seed = 42
    while time.time() < deadline - 5.0:
        if time.time() > deadline - 15.0 and best_code is not None:
            break

        code = greedy_init(seed)
        if len(code) == 0:
            seed += 1
            continue

        # Run local search
        code = local_search(code, seed)

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        seed += 1

    if best_code is None:
        code = greedy_init(0)
        best_code = code

    # Convert indices back to words
    words = []
    for i in best_code:
        w = []
        tmp = i
        for pos in range(7):
            w.append(tmp % 3)
            tmp //= 3
        words.append(w)

    return words


def _construct_general(q, n, radius, deadline):
    import itertools
    N = q ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}

    # Compute balls
    balls = []
    for i, w in enumerate(words):
        ball = set([i])
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                neigh = list(w)
                neigh[pos] = val
                j = word_to_idx[tuple(neigh)]
                ball.add(j)
        balls.append(ball)

    # Greedy
    uncovered = set(range(N))
    code = []
    while uncovered and time.time() < deadline - 5.0:
        best_idx = -1
        best_count = -1
        for i in range(N):
            if i in code:
                continue
            cnt = len(uncovered & balls[i])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_idx == -1 or best_count == 0:
            break
        code.append(best_idx)
        uncovered -= balls[best_idx]

    return [words[i] for i in code]
