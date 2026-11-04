"""Greedy set-covering + simulated annealing for K_3(6,1) ternary covering code."""
import time
import random
import itertools
import numpy as np


def _hamming_ball_indices(word, q, n, radius):
    """Return all indices of words within Hamming distance <= radius from word.
    word: tuple of length n, values 0..q-1
    Returns list of integer indices (base-q encoding).
    """
    indices = []
    # For radius 1: the word itself + n*(q-1) neighbors
    base = 0
    for i in range(n):
        base = base * q + word[i]
    indices.append(base)
    if radius >= 1:
        for pos in range(n):
            for val in range(q):
                if val == word[pos]:
                    continue
                w = list(word)
                w[pos] = val
                idx = 0
                for i in range(n):
                    idx = idx * q + w[i]
                indices.append(idx)
    return indices


def _greedy_cover(q, n, radius, max_iters=500, deadline=None):
    """Greedy set-covering: repeatedly pick the word covering the most uncovered words."""
    total = q ** n
    # Precompute all Hamming balls
    all_balls = {}
    for word_tuple in itertools.product(range(q), repeat=n):
        all_balls[word_tuple] = _hamming_ball_indices(word_tuple, q, n, radius)

    # Convert to numpy arrays for speed
    ball_arrays = [np.array(v, dtype=np.int32) for v in all_balls.values()]
    word_tuples = list(all_balls.keys())

    covered = np.zeros(total, dtype=bool)
    selected = []
    remaining = total

    # Precompute: for each candidate, the set of indices it covers
    # Greedy: pick candidate with max new coverage
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if remaining == 0:
            break

        best_gain = -1
        best_idx = -1
        # Sample candidates if too many
        candidates = word_tuples
        for ci, wt in enumerate(candidates):
            gain = int(np.sum(ball_arrays[ci][~covered]))
            if gain > best_gain:
                best_gain = gain
                best_idx = ci
            if best_gain == 0:
                break

        if best_gain <= 0:
            break

        covered[ball_arrays[best_idx]] = True
        selected.append(word_tuples[best_idx])
        remaining -= best_gain

    return selected


def _try_remove(code, q, n, radius, balls_cache=None):
    """Try to remove each codeword and check if coverage is still complete."""
    total = q ** n
    if balls_cache is None:
        balls_cache = {}

    def get_ball(wt):
        if wt not in balls_cache:
            balls_cache[wt] = _hamming_ball_indices(wt, q, n, radius)
        return balls_cache[wt]

    # Build coverage count: for each word in space, how many codewords cover it
    cover_count = np.zeros(total, dtype=np.int32)
    for wt in code:
        for idx in get_ball(wt):
            cover_count[idx] += 1

    new_code = []
    for wt in code:
        ball = get_ball(wt)
        # If removing this word still leaves all covered
        if np.all(cover_count[ball] > 1):
            for idx in ball:
                cover_count[idx] -= 1
            # Don't add to new_code (removed it)
        else:
            new_code.append(wt)

    return new_code


def _simulated_annealing(code, q, n, radius, deadline):
    """Simulated annealing: try to swap out one codeword for another to reduce size."""
    total = q ** n
    balls_cache = {}

    def get_ball(wt):
        if wt not in balls_cache:
            balls_cache[wt] = _hamming_ball_indices(wt, q, n, radius)
        return balls_cache[wt]

    current_code = list(code)
    best_code = list(current_code)
    best_len = len(current_code)

    # Build coverage count
    cover_count = np.zeros(total, dtype=np.int32)
    for wt in current_code:
        for idx in get_ball(wt):
            cover_count[idx] += 1

    temperature = 1.0
    cooling = 0.98
    min_temp = 0.01
    step_count = 0

    while temperature > min_temp and time.time() < deadline:
        step_count += 1
        # Pick a random codeword to try removing
        if len(current_code) == 0:
            break
        i = random.randint(0, len(current_code) - 1)
        wt_remove = current_code[i]
        ball_remove = get_ball(wt_remove)

        # Check if we can remove it
        if np.all(cover_count[ball_remove] > 1):
            # Remove it
            for idx in ball_remove:
                cover_count[idx] -= 1
            current_code.pop(i)
            # Try to add a random word back to explore
            # Actually, just keep it removed for now (it's an improvement)
            if len(current_code) < best_len:
                best_len = len(current_code)
                best_code = list(current_code)
        else:
            # Can't remove; try swapping: remove this, add a random other word
            # Pick a random candidate word
            cand = tuple(random.randint(0, q - 1) for _ in range(n))
            ball_cand = get_ball(cand)

            # Check: after removing wt_remove and adding cand, is everything covered?
            # We need cover_count[idx] >= 1 for all idx after the swap
            # Temporarily remove wt_remove
            for idx in ball_remove:
                cover_count[idx] -= 1
            # Add cand
            for idx in ball_cand:
                cover_count[idx] += 1

            if np.all(cover_count >= 1):
                # Valid swap, accept
                current_code[i] = cand
                if len(current_code) < best_len:
                    best_len = len(current_code)
                    best_code = list(current_code)
            else:
                # Revert
                for idx in ball_cand:
                    cover_count[idx] -= 1
                for idx in ball_remove:
                    cover_count[idx] += 1

        temperature *= cooling

    return best_code


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # For the specific case q=3, n=6, radius=1
    # Greedy set covering
    code = _greedy_cover(q, n, radius, max_iters=300, deadline=deadline)

    if len(code) == 0:
        # Fallback: use the tetracode construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Try to remove redundant codewords
    for _ in range(5):
        if time.time() > deadline - 30:
            break
        new_code = _try_remove(code, q, n, radius)
        if len(new_code) < len(code):
            code = new_code
        else:
            break

    # Simulated annealing to try to improve further
    if time.time() < deadline - 10:
        code = _simulated_annealing(code, q, n, radius, deadline)

    # Final cleanup: remove any redundant words
    code = _try_remove(code, q, n, radius)

    # Ensure distinctness
    seen = set()
    unique_code = []
    for c in code:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            unique_code.append(list(c))

    return unique_code
