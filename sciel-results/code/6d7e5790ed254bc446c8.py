import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s kill

    total_words = q ** n
    ball_size = 1 + n * (q - 1)

    def int_to_tuple(x, nn, qq):
        t = [0] * nn
        for i in range(nn):
            t[i] = x % qq
            x //= qq
        return t

    def tuple_to_int(t, qq):
        x = 0
        for i in range(n - 1, -1, -1):
            x = x * qq + t[i]
        return x

    # Precompute balls
    balls = [None] * total_words
    for i in range(total_words):
        t_i = int_to_tuple(i, n, q)
        ball = [i]
        if radius >= 1:
            for pos in range(n):
                orig_val = t_i[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    t_new = t_i[:]
                    t_new[pos] = new_val
                    ball.append(tuple_to_int(t_new, q))
        balls[i] = ball

    # Precompute inverse balls: inv_balls[w] = list of c such that w in balls[c]
    inv_balls = [[] for _ in range(total_words)]
    for c in range(total_words):
        for w in balls[c]:
            inv_balls[w].append(c)

    # Coverage matrix for fast scoring
    M = np.zeros((total_words, total_words), dtype=np.uint8)
    for i in range(total_words):
        M[i, balls[i]] = 1

    # Greedy cover to get initial upper bound
    def greedy_cover():
        covered = np.zeros(total_words, dtype=bool)
        codewords = []
        for _ in range(total_words):
            if not np.any(~covered):
                break
            scores = M.dot((~covered).astype(np.uint8))
            best_idx = int(np.argmax(scores))
            if scores[best_idx] == 0:
                break
            codewords.append(best_idx)
            covered[balls[best_idx]] = True
        return codewords

    # Local search: remove redundant codewords
    def local_search_remove(codewords):
        if not codewords:
            return codewords
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            cover_count = np.zeros(total_words, dtype=np.int32)
            for c in codewords:
                cover_count[balls[c]] += 1
            for i in range(len(codewords)):
                c = codewords[i]
                if np.min(cover_count[balls[c]]) > 1:
                    codewords.pop(i)
                    changed = True
                    break
        return codewords

    # Get initial solution
    initial = greedy_cover()
    initial = local_search_remove(initial)
    best_size = len(initial)
    best_codewords = initial[:]

    # Randomized backtracking search with constraint propagation
    def backtracking_search(uncovered_mask, chosen, max_size, rng):
        """
        Try to find a cover of size <= max_size using randomized backtracking.
        uncovered_mask: boolean array, True for uncovered words
        chosen: list of codewords already selected
        max_size: maximum allowed total codewords
        rng: random.Random instance
        Returns list of codewords if found, else None.
        """
        # Check timeout
        if time.time() > deadline - 5:
            return None

        # If all covered, we have a valid solution
        if not np.any(uncovered_mask):
            return chosen[:]

        # Lower bound: each codeword covers at most ball_size words
        remaining = int(np.sum(uncovered_mask))
        if remaining == 0:
            return chosen[:]

        # Lower bound on additional codewords needed
        lb = (remaining + ball_size - 1) // ball_size  # ceil(remaining / ball_size)
        if len(chosen) + lb > max_size:
            return None

        # MRV: find uncovered word with fewest candidate codewords
        # Among uncovered words, pick one with minimum |candidates|
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            return chosen[:]

        # Compute candidate counts for each uncovered word
        # For efficiency, sample if too many
        if len(uncovered_indices) > 200:
            sample_idx = rng.sample(list(uncovered_indices), 200)
        else:
            sample_idx = list(uncovered_indices)

        # Find the uncovered word with fewest candidates
        best_word = -1
        best_cand_count = float('inf')
        for w in sample_idx:
            if not uncovered_mask[w]:
                continue
            count = len(inv_balls[w])
            if count < best_cand_count:
                best_cand_count = count
                best_word = w

        if best_word == -1:
            return None

        # Get candidates for best_word, filter to only those not already chosen
        chosen_set = set(chosen)
        candidates = [c for c in inv_balls[best_word] if c not in chosen_set]

        if not candidates:
            return None  # Dead end

        # Score candidates by how many uncovered words they cover
        uncovered_u8 = uncovered_mask.astype(np.uint8)
        cand_scores = {}
        for c in candidates:
            score = int(M[c].dot(uncovered_u8))
            cand_scores[c] = score

        # Sort by score descending, with random tie-breaking
        sorted_cands = sorted(candidates, key=lambda c: (-cand_scores[c], rng.random()))

        # Try candidates in order
        for c in sorted_cands:
            if len(chosen) + 1 > max_size:
                break

            # Apply this codeword
            new_uncovered = uncovered_mask.copy()
            new_uncovered[balls[c]] = False
            chosen.append(c)

            result = backtracking_search(new_uncovered, chosen, max_size, rng)
            if result is not None:
                return result

            chosen.pop()

        return None

    # Iterated backtracking search
    rng = random.Random(42)

    # Try to improve upon the initial solution
    search_timeout_per_attempt = 5.0  # seconds per backtracking attempt

    while time.time() < deadline - 10:
        # Try to find a solution of size best_size - 1
        target_size = best_size - 1

        # Set up the search: start with all words uncovered
        initial_uncovered = np.ones(total_words, dtype=bool)

        # Run backtracking search
        search_start = time.time()
        result = backtracking_search(initial_uncovered, [], target_size, rng)

        if result is not None:
            # Found a valid cover smaller than best_size
            result = local_search_remove(result)
            if len(result) < best_size:
                best_size = len(result)
                best_codewords = result[:]
                # Restart with new smaller best_size
                continue

        # If no improvement, try a different randomization
        # Vary the target slightly or use a different random seed pattern
        # We can also try to find a solution of the same size but different structure
        # to potentially enable further reductions

        # Occasionally, try a larger perturbation: start from a random subset
        if rng.random() < 0.3:
            # Start from a random valid cover and try to reduce
            random_cover = greedy_cover()
            random_cover = local_search_remove(random_cover)
            if len(random_cover) < best_size:
                best_size = len(random_cover)
                best_codewords = random_cover[:]

    # Final: ensure we have a valid cover
    if not best_codewords:
        best_codewords = greedy_cover()
        best_codewords = local_search_remove(best_codewords)

    # Convert to output format
    result = []
    for c in best_codewords:
        result.append(int_to_tuple(c, n, q))

    return result
