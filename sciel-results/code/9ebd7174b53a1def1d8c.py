"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    if n < 4:
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Ternary tetracode words
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

    # Construct initial code
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_code.append(h + list(tail))

    size_space = 3 ** n

    # Precompute ball matrix: ball_matrix[i, j] = 1 if word j is in ball of word i
    # Use vectorized construction
    base = np.array([3 ** i for i in range(n)], dtype=np.int64)
    words_int = np.arange(size_space, dtype=np.int64)
    words_arr = np.zeros((size_space, n), dtype=np.int64)
    for pos in range(n):
        words_arr[:, pos] = (words_int // base[pos]) % 3

    # Ball matrix: for each word, which words are within Hamming distance <= radius
    # For radius 1: self + all single-position changes
    # ball_matrix[i] = set of words within radius of words_arr[i]
    # Construct via broadcasting: hamming distance between all pairs
    # words_arr: (size_space, n), diff: (size_space, size_space, n)
    # This is 729*729*6 = ~3.2M entries, fine
    diff = words_arr[:, None, :] != words_arr[None, :, :]
    ham = diff.sum(axis=2)
    ball_matrix = (ham <= radius).astype(np.int64)  # (size_space, size_space), 0/1
    # ball_matrix[i] is the ball of word i

    def greedy_remove_vectorized(active_mask, coverage, time_limit):
        """
        active_mask: bool array (size_space,), which codewords are active
        coverage: int array (size_space,), current coverage counts
        Returns updated active_mask, coverage.
        """
        # For each active codeword, its ball's min coverage = min over j where ball_matrix[i,j]=1 of coverage[j]
        # = min over j of (coverage[j] if ball_matrix[i,j] else +inf)
        # Compute for all i at once:
        # masked_cov = where(ball_matrix>0, coverage, +inf), shape (size_space, size_space)
        INF = 10**9
        masked_cov = np.where(ball_matrix > 0, coverage[None, :], INF)
        min_cov_all = masked_cov.min(axis=1)  # (size_space,)
        
        while time.time() < time_limit:
            # Find active codewords with min_cov > 1
            removable = active_mask & (min_cov_all > 1)
            if not removable.any():
                break
            # Pick the one with max min_cov (most redundant)
            candidates = np.where(removable)[0]
            scores = min_cov_all[candidates]
            best_local = candidates[np.argmax(scores)]
            # Remove it
            active_mask[best_local] = False
            coverage -= ball_matrix[best_local]
            # Update min_cov_all: only entries where ball_matrix[:, best_local] = 1 change
            # For those rows i, min_cov may decrease
            affected = ball_matrix[:, best_local] > 0
            masked_cov[affected] = np.where(ball_matrix[affected] > 0, coverage[None, :], INF)
            min_cov_all[affected] = masked_cov[affected].min(axis=1)
        return active_mask, coverage

    def try_random_removal_vectorized(active_mask, coverage, time_limit, max_attempts=20000):
        """Random removals to escape local optima."""
        attempts = 0
        while time.time() < time_limit and attempts < max_attempts:
            attempts += 1
            active_idx = np.where(active_mask)[0]
            if len(active_idx) == 0:
                break
            i = active_idx[random.randint(0, len(active_idx) - 1)]
            # Check removable: all words in ball have coverage > 1
            ball_words = np.where(ball_matrix[i] > 0)[0]
            if coverage[ball_words].min() > 1:
                active_mask[i] = False
                coverage -= ball_matrix[i]
        return active_mask, coverage

    best_code = [list(c) for c in initial_code]
    best_size = len(best_code)

    def code_to_mask(code_list):
        mask = np.zeros(size_space, dtype=bool)
        for cw in code_list:
            # convert word to int
            val = 0
            for i in range(n):
                val += cw[i] * (3 ** i)
            mask[val] = True
        return mask

    def mask_to_code(mask):
        code = []
        idxs = np.where(mask)[0]
        for val in idxs:
            w = [0] * n
            v = val
            for i in range(n):
                w[i] = v % 3
                v //= 3
            code.append(w)
        return code

    # Strategy 1: Pure greedy removal from initial code
    mask = code_to_mask(initial_code)
    coverage = ball_matrix[mask].sum(axis=0)
    mask, coverage = greedy_remove_vectorized(mask.copy(), coverage.copy(), deadline - 25.0)
    code = mask_to_code(mask)
    if len(code) < best_size:
        best_code = code
        best_size = len(code)

    # Strategy 2: Multiple restarts with random perturbation + greedy
    remaining_time = deadline - 5.0 - time.time()
    if remaining_time > 10:
        # Try a few different starting points and restart counts
        num_restarts = min(8, int(remaining_time // 4))
        for restart in range(num_restarts):
            if time.time() > deadline - 8.0:
                break

            # Start from best_code found so far
            mask = code_to_mask(best_code)
            coverage = ball_matrix[mask].sum(axis=0)

            # Random perturbation: remove some random removable codewords
            rand_time = min(1.0, remaining_time * 0.15)
            mask, coverage = try_random_removal_vectorized(mask.copy(), coverage.copy(), time.time() + rand_time, max_attempts=3000)

            # Then greedy removal
            greedy_time = min(deadline - 8.0, time.time() + (remaining_time * 0.3))
            mask, coverage = greedy_remove_vectorized(mask.copy(), coverage.copy(), greedy_time)
            code = mask_to_code(mask)
            if len(code) < best_size:
                best_code = code
                best_size = len(code)

            remaining_time = deadline - 5.0 - time.time()
            if remaining_time < 5:
                break

    # Final: try one more greedy pass on best_code
    if time.time() < deadline - 2.0:
        mask = code_to_mask(best_code)
        coverage = ball_matrix[mask].sum(axis=0)
        mask, coverage = greedy_remove_vectorized(mask.copy(), coverage.copy(), deadline - 1.0)
        code = mask_to_code(mask)
        if len(code) < best_size:
            best_code = code
            best_size = len(code)

    return best_code

# EVOLVE-BLOCK-END
