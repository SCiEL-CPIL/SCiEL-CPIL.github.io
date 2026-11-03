import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0
    random.seed(42)

    # This strategy is specialized for the ternary covering code K_3(8,1)
    # but we include a generic fallback for other parameters.
    if not (q == 3 and n == 8 and radius == 1):
        return _generic_construct(q, n, radius, deadline)

    total = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    def to_int(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val

    def to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    # Build balls: balls[i] is the set of indices within Hamming distance 1 of word i
    # For q=3, n=8: ball size = 17
    def build_balls():
        # Precompute digits
        digits = np.zeros((total, n), dtype=np.int32)
        for i in range(total):
            v = i
            for pos in range(n - 1, -1, -1):
                digits[i, pos] = v % q
                v //= q
        balls = np.zeros((total, ball_size), dtype=np.int32)
        balls[:, 0] = np.arange(total)
        for pos in range(n):
            for delta in range(1, q):
                new_digits = digits.copy()
                new_digits[:, pos] = (digits[:, pos] + delta) % q
                # Convert back to int
                vals = np.zeros(total, dtype=np.int32)
                for p in range(n):
                    vals = vals * q + new_digits[:, p]
                balls[:, 1 + pos * (q - 1) + delta - 1] = vals
        return balls

    balls = build_balls()

    # Algebraic construction: union of two structured codes
    # Part 1: Ternary Hamming code H_3(4) - 27 words of length 4
    # Part 2: Ternary Simplex code S_3(4) - 13 words of length 4
    # We'll create a product-like structure for length 8

    # For length 8, we can use a "folding" or "product" approach
    # Actually, for K_3(8,1), a good construction is based on the fact that
    # we can partition the 6561 words into structured subsets.

    # Let's use a different algebraic approach:
    # The ternary Hamming code of length 4 has 27 codewords and covers all 81 words of length 4
    # with radius 1. We can use this as a building block.

    # For length 8, we can think of it as two length-4 halves.
    # A code of size 27*27 = 729 would be too large.
    # Instead, we use a "switching" construction or a direct algebraic code.

    # Let's try a simpler structured approach:
    # Start with a random-ish but structured initial code, then optimize.
    # The key insight: greedy set cover is slow and suboptimal.
    # Instead, we'll use a "covering array" style construction followed by local search.

    # Initial construction: Use a greedy approach but with a better heuristic
    # Actually, let's just do a fast greedy with numpy, then aggressive local search.

    # Fast greedy initialization
    covered = np.zeros(total, dtype=np.bool_)
    covered_count = 0
    codewords = []

    # Precompute uncovered count per word for fast gain calculation
    # gain[w] = number of uncovered points in ball[w]
    # We'll maintain this incrementally

    while covered_count < total:
        if time.time() > deadline - 30:  # Leave time for local search
            break

        # Compute gains: for each word, how many uncovered points in its ball
        # Vectorized approach
        uncovered = ~covered
        # For each word w, gain[w] = sum(uncovered[balls[w]])
        # This is a matrix-vector product: balls is (total, ball_size), uncovered is (total,)
        # We need: for each row w, sum of uncovered at positions balls[w]
        # This is equivalent to: (balls.T @ uncovered) but balls is not sparse in a useful way

        # Alternative: use the fact that balls[w] contains w and its neighbors
        # gain[w] = uncovered[w] + sum over neighbors
        # Let's precompute the neighbor structure

        # For efficiency, let's just compute gains directly using numpy fancy indexing
        # covered_in_balls = covered[balls]  # (total, ball_size)
        # gains = ball_size - covered_in_balls.sum(axis=1)

        # This is O(total * ball_size) = O(6561 * 17) ≈ 111k operations per step
        # With ~500 steps, that's ~55M operations, which should be fast in numpy

        covered_in_balls = covered[balls]
        gains = ball_size - covered_in_balls.sum(axis=1)

        best_word = int(np.argmax(gains))
        best_new = int(gains[best_word])

        if best_new == 0:
            break

        codewords.append(best_word)
        ball_points = balls[best_word]
        newly_covered = ~covered[ball_points]
        if newly_covered.any():
            covered[ball_points] = True
            covered_count += int(newly_covered.sum())

    # Now do local search to reduce code size
    # Strategy: try to remove each codeword if possible, then try 1-in/1-out swaps

    # Build coverage count
    cov_count = np.zeros(total, dtype=np.int32)
    for cw in codewords:
        cov_count[balls[cw]] += 1

    # Remove redundant codewords
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for i in range(len(codewords)):
            if time.time() > deadline - 5:
                break
            cw = codewords[i]
            if (cov_count[balls[cw]] > 1).all():
                cov_count[balls[cw]] -= 1
                codewords.pop(i)
                changed = True
                break

    # 1-in/1-out local search
    # Try to replace one codeword with another to reduce size or improve coverage
    # Actually, we want to REDUCE size, so we try to find a codeword that can be removed
    # after adding a new one that covers the same points.

    # Better: try to find two codewords that can be replaced by one
    # Or: try to remove a codeword and see if we can cover the exposed points with existing codewords

    # Let's do a more systematic local search:
    # For each codeword, try to remove it. If this leaves some points uncovered,
    # try to add a new codeword that covers all those points (if possible with one word).

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(codewords)):
            if time.time() > deadline:
                break
            cw = codewords[i]
            # Tentatively remove cw
            exposed = balls[cw]
            still_covered = cov_count[exposed] > 1  # Points still covered by other codewords
            newly_exposed = exposed[~still_covered]  # Points that become uncovered

            if len(newly_exposed) == 0:
                # Can remove directly
                cov_count[balls[cw]] -= 1
                codewords.pop(i)
                improved = True
                break

            # Try to find a single word that covers all newly_exposed points
            # A word w covers a point x if x is in ball[w], i.e., d(w,x) <= 1
            # So w must be within distance 1 of all newly_exposed points
            # This means the intersection of balls[newly_exposed] must be non-empty

            if len(newly_exposed) > 0:
                # Find words that cover all newly_exposed points
                # A word w covers point x if x in balls[w]
                # Equivalently, w in balls[x] (since distance is symmetric)
                # So w must be in the intersection of balls[x] for all x in newly_exposed

                # For small sets, we can compute the intersection
                if len(newly_exposed) <= 4:
                    # Find common neighbors
                    candidate_set = set(balls[newly_exposed[0]].tolist())
                    for x in newly_exposed[1:]:
                        candidate_set &= set(balls[x].tolist())
                        if not candidate_set:
                            break

                    if candidate_set:
                        # Pick the best candidate (one that maximizes new coverage)
                        best_new_word = -1
                        best_gain = 0
                        for w in candidate_set:
                            w = int(w)
                            # Gain = number of currently uncovered points in ball[w]
                            # (excluding the newly_exposed which we're trying to cover)
                            gain = int((~covered[balls[w]]).sum())
                            if gain > best_gain:
                                best_gain = gain
                                best_new_word = w

                        if best_new_word >= 0:
                            # Perform the swap
                            cov_count[balls[cw]] -= 1
                            cov_count[balls[best_new_word]] += 1
                            codewords[i] = best_new_word
                            # Update covered
                            covered[balls[best_new_word]] = True
                            improved = True
                            break

    # Final redundancy removal
    cov_count = np.zeros(total, dtype=np.int32)
    for cw in codewords:
        cov_count[balls[cw]] += 1

    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(codewords)):
            if time.time() > deadline:
                break
            cw = codewords[i]
            if (cov_count[balls[cw]] > 1).all():
                cov_count[balls[cw]] -= 1
                codewords.pop(i)
                changed = True
                break

    result = [to_word(w) for w in codewords]

    # Ensure distinct
    seen = set()
    unique_result = []
    for w in result:
        key = tuple(w)
        if key not in seen:
            seen.add(key)
            unique_result.append(w)

    return unique_result

def _generic_construct(q, n, radius, deadline):
    # Fallback for non-ternary or non-length-8 cases
    start_time = time.time()
    total = q ** n

    def to_int(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val

    def to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    # Build balls
    ball_size = 1
    for pos in range(n):
        ball_size += (q - 1)

    balls = np.zeros((total, ball_size), dtype=np.int32)
    balls[:, 0] = np.arange(total)
    idx = 1
    digits = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        v = i
        for pos in range(n - 1, -1, -1):
            digits[i, pos] = v % q
            v //= q

    for pos in range(n):
        for delta in range(1, q):
            new_digits = digits.copy()
            new_digits[:, pos] = (digits[:, pos] + delta) % q
            vals = np.zeros(total, dtype=np.int32)
            for p in range(n):
                vals = vals * q + new_digits[:, p]
            balls[:, idx] = vals
            idx += 1

    covered = np.zeros(total, dtype=np.bool_)
    covered_count = 0
    codewords = []

    while covered_count < total:
        if time.time() > deadline - 5:
            break
        covered_in_balls = covered[balls]
        gains = ball_size - covered_in_balls.sum(axis=1)
        best_word = int(np.argmax(gains))
        best_new = int(gains[best_word])
        if best_new == 0:
            break
        codewords.append(best_word)
        ball_points = balls[best_word]
        newly_covered = ~covered[ball_points]
        if newly_covered.any():
            covered[ball_points] = True
            covered_count += int(newly_covered.sum())

    # Remove redundant
    cov_count = np.zeros(total, dtype=np.int32)
    for cw in codewords:
        cov_count[balls[cw]] += 1

    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(codewords)):
            if time.time() > deadline:
                break
            cw = codewords[i]
            if (cov_count[balls[cw]] > 1).all():
                cov_count[balls[cw]] -= 1
                codewords.pop(i)
                changed = True
                break

    result = [to_word(w) for w in codewords]
    seen = set()
    unique_result = []
    for w in result:
        key = tuple(w)
        if key not in seen:
            seen.add(key)
            unique_result.append(w)
    return unique_result
