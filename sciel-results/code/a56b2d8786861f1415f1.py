"""Ternary covering code via multi-start simulated annealing with penalty-based coverage objective."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius `radius`.
    Strategy:
      1) Precompute the full distance/ball structure (N x N) since N = q^n is small (2187 here).
      2) Initialize with a random set of codewords sized near a heuristic lower bound.
      3) Run multi-start simulated annealing on the set of codewords:
         - Objective: |C| + penalty * (number of uncovered words)
         - Moves: add a random word, remove a random word, or swap a random word.
         - Accept moves that reduce objective; accept worsening moves with probability exp(-delta/T).
         - This allows temporary coverage violations to escape local minima.
      4) Return the best valid (zero-uncovered) code found.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin

    N = q ** n
    if N <= 0:
        return []

    # Precompute all words as integer indices 0..N-1
    # all_words[i] is the i-th word in base-q representation
    all_words = np.zeros((N, n), dtype=np.int32)
    for idx in range(N):
        rem = idx
        for j in range(n - 1, -1, -1):
            all_words[idx, j] = rem % q
            rem //= q

    # Compute Hamming distance matrix
    hamming = np.zeros((N, N), dtype=np.int32)
    for k in range(n):
        col = all_words[:, k].astype(np.int32)
        diff = (col[:, None] != col[None, :]).astype(np.int32)
        hamming += diff

    ball_mask = hamming <= radius  # ball_mask[c, w] = True if w is within radius of c
    del hamming

    # Precompute for each codeword, the set of words it covers (as a boolean array)
    # ball_mask is already this: ball_mask[c] is the coverage of codeword c

    # Heuristic initial size: use a rough lower bound
    # Sphere covering bound: N / (1 + n*(q-1)*radius) is a very rough lower bound
    # For q=3, n=7, radius=1: N=2187, ball_size = 1 + 7*2 = 15, so 2187/15 ≈ 146
    # Best known is 186, so start around 170-190
    ball_size = 1 + n * (q - 1) * radius
    initial_size = max(1, int(N / ball_size) + 10)
    initial_size = min(initial_size, N)

    best_code = None
    best_obj = float('inf')
    best_valid = None  # best valid (uncovered=0) code

    def compute_coverage(code_set):
        """Return (uncovered_count, uncovered_mask) for a set of codewords."""
        if len(code_set) == 0:
            return N, np.ones(N, dtype=bool)
        # covered = OR of ball_mask[c] for c in code_set
        covered = np.zeros(N, dtype=bool)
        for c in code_set:
            covered |= ball_mask[c]
        uncovered = ~covered
        return np.sum(uncovered), uncovered

    def objective(code_set, uncovered_count):
        """Objective function: |C| + penalty * uncovered_count"""
        penalty = 100.0  # high penalty to strongly discourage uncovered words
        return len(code_set) + penalty * uncovered_count

    # Multi-start simulated annealing
    num_starts = 5
    for start_idx in range(num_starts):
        if time.time() > deadline - 30:
            break

        # Initialize with a random set of codewords
        if start_idx == 0:
            # First start: use a greedy-like initialization for a good starting point
            uncovered_mask = np.ones(N, dtype=bool)
            code_set = []
            while np.any(uncovered_mask) and time.time() < deadline - 30:
                # Find codeword covering most uncovered
                new_cov = np.sum(ball_mask & uncovered_mask[None, :], axis=1)
                best = np.argmax(new_cov)
                if new_cov[best] == 0:
                    break
                code_set.append(int(best))
                uncovered_mask[ball_mask[best]] = False
            # Convert to a list
            code_list = list(code_set)
        else:
            # Subsequent starts: random initialization
            code_list = random.sample(range(N), min(initial_size, N))

        # Compute initial state
        uncovered_count, uncovered_mask = compute_coverage(code_list)
        current_obj = objective(code_list, uncovered_count)

        if uncovered_count == 0:
            if len(code_list) < best_obj:
                best_obj = len(code_list)
                best_code = code_list.copy()
                best_valid = code_list.copy()

        # Simulated annealing
        T_start = 10.0
        T_end = 0.1
        cooling = 0.995
        T = T_start
        max_iters = 5000
        iter_count = 0

        while T > T_end and iter_count < max_iters:
            if time.time() > deadline - 15:
                break

            # Choose a random move
            move_type = random.random()
            new_code_list = code_list.copy()

            if move_type < 0.3:
                # Add a random word
                if len(new_code_list) < N:
                    candidate = random.randint(0, N - 1)
                    if candidate not in new_code_list:
                        new_code_list.append(candidate)
                    else:
                        continue
            elif move_type < 0.6:
                # Remove a random word
                if len(new_code_list) > 1:
                    idx = random.randint(0, len(new_code_list) - 1)
                    new_code_list.pop(idx)
                else:
                    continue
            else:
                # Swap: remove one, add another
                if len(new_code_list) > 0:
                    idx = random.randint(0, len(new_code_list) - 1)
                    removed = new_code_list.pop(idx)
                    candidate = random.randint(0, N - 1)
                    if candidate not in new_code_list:
                        new_code_list.append(candidate)
                    else:
                        # Put it back
                        new_code_list.insert(idx, removed)
                        continue

            # Compute new uncovered count
            new_uncovered_count, _ = compute_coverage(new_code_list)
            new_obj = objective(new_code_list, new_uncovered_count)

            delta = new_obj - current_obj

            # Accept or reject
            if delta <= 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                code_list = new_code_list
                uncovered_count = new_uncovered_count
                current_obj = new_obj

                # Track best valid solution
                if uncovered_count == 0:
                    if len(code_list) < best_obj:
                        best_obj = len(code_list)
                        best_code = code_list.copy()
                        best_valid = code_list.copy()

            T *= cooling
            iter_count += 1

    # If no valid solution was found, fall back to greedy
    if best_valid is None:
        uncovered_mask = np.ones(N, dtype=bool)
        code_set = []
        while np.any(uncovered_mask) and time.time() < deadline - 5:
            new_cov = np.sum(ball_mask & uncovered_mask[None, :], axis=1)
            best = np.argmax(new_cov)
            if new_cov[best] == 0:
                break
            code_set.append(int(best))
            uncovered_mask[ball_mask[best]] = False
        best_valid = code_set

    # Final local improvement: try to delete redundant codewords
    if best_valid is not None and time.time() < deadline - 5:
        code_arr = list(best_valid)
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(len(code_arr) - 1, -1, -1):
                if time.time() > deadline - 3:
                    break
                remaining = code_arr[:i] + code_arr[i+1:]
                if len(remaining) == 0:
                    covered = np.zeros(N, dtype=bool)
                else:
                    covered = np.zeros(N, dtype=bool)
                    for c in remaining:
                        covered |= ball_mask[c]
                if np.all(covered):
                    code_arr = remaining
                    improved = True

        if len(code_arr) < len(best_valid):
            best_valid = code_arr

    # Final validity check
    if len(best_valid) == 0:
        covered = np.zeros(N, dtype=bool)
    else:
        covered = np.zeros(N, dtype=bool)
        for c in best_valid:
            covered |= ball_mask[c]

    if not np.all(covered):
        # Fallback: greedy
        uncovered_mask = np.ones(N, dtype=bool)
        code_set = []
        while np.any(uncovered_mask):
            new_cov = np.sum(ball_mask & uncovered_mask[None, :], axis=1)
            best = np.argmax(new_cov)
            if new_cov[best] == 0:
                break
            code_set.append(int(best))
            uncovered_mask[ball_mask[best]] = False
        best_valid = code_set

    return [list(all_words[i]) for i in best_valid]
# EVOLVE-BLOCK-END
