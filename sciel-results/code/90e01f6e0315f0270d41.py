"""Combined program: vectorized greedy set cover with local improvement for K_3(7,1)."""
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        # Return trivial valid code (all words) for safety
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Precompute: ball adjacency as index arrays
    # N = 3^7 = 2187
    N = q ** n
    powers = [3 ** i for i in range(n)]  # powers[i] = 3^i

    # Precompute ball membership: for each word index, list of indices in its ball (distance <= 1)
    # Ball of word w: w itself + all words differing in exactly one position
    ball_indices = np.zeros((N, 1 + n * (q - 1)), dtype=np.int32)
    ball_sizes = np.zeros(N, dtype=np.int32)

    for idx in range(N):
        # Get digits of idx in base 3
        digits = []
        tmp = idx
        for _ in range(n):
            digits.append(tmp % 3)
            tmp //= 3
        # digits[0] is least significant (position 0 in our power ordering)
        # But we want positions 0..n-1 where position i has weight 3^i
        # So digits[i] corresponds to position i

        ball = [idx]
        for pos in range(n):
            orig = digits[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                new_idx = idx + (new_val - orig) * powers[pos]
                ball.append(new_idx)
        ball_indices[idx, :len(ball)] = ball
        ball_sizes[idx] = len(ball)

    # Precompute reverse coverage: for each word, which codewords cover it
    # cover_mask[word_idx] = set of codeword indices whose ball contains word_idx
    # We'll represent this as a list of lists or use numpy for speed
    # Actually, for greedy we need: given a set of chosen codewords, which words are covered.
    # And for each candidate codeword, how many NEW words it covers.

    # Strategy: use numpy boolean arrays
    # covered: boolean array of size N, True if word is covered
    # For each greedy step, compute gain for each candidate = number of uncovered words in its ball

    # Precompute ball as a 2D array for vectorized operations
    # ball_indices[i, j] is the j-th word in ball of codeword i
    # We already have this.

    def greedy_cover(seed_code_indices, rng):
        """
        Greedy set cover starting from initial set of codeword indices.
        Returns list of codeword indices.
        """
        covered = np.zeros(N, dtype=bool)
        code_set = set(seed_code_indices)
        code_list = list(seed_code_indices)

        # Mark initially covered words
        for ci in seed_code_indices:
            bs = ball_sizes[ci]
            covered[ball_indices[ci, :bs]] = True

        uncovered_count = N - covered.sum()

        while uncovered_count > 0:
            # Compute gain for each candidate not in code_set
            # gain[i] = number of uncovered words in ball of i
            # We can compute this vectorially
            # For each candidate i, gain = sum(covered[ball_indices[i, :ball_sizes[i]]] == False)

            # Vectorized approach:
            # Create a matrix where row i has ball_indices[i, :ball_sizes[i]]
            # Then gain[i] = (~covered[ball_matrix[i]]).sum()
            # But ball_sizes vary, so we need to handle carefully.

            # Since all balls have the same size for q=3, n=7: 1 + 7*2 = 15
            ball_size = 1 + n * (q - 1)  # 15
            # ball_indices is (N, 15)
            # covered is (N,)
            # We want for each row i: sum of ~covered at positions ball_indices[i, :]

            # Use advanced indexing
            # indices_matrix = ball_indices  # (N, 15)
            # covered_at_ball = covered[ball_indices]  # (N, 15) boolean
            # gains = (~covered_at_ball).sum(axis=1)  # (N,)

            covered_at_ball = covered[ball_indices]  # (N, ball_size)
            gains = (~covered_at_ball).sum(axis=1)  # (N,)

            # Exclude already chosen codewords
            for ci in code_set:
                gains[ci] = 0

            # Find best gain
            best_gain = gains.max()
            if best_gain == 0:
                break  # Should not happen if uncovered_count > 0

            # Find all candidates with best gain
            candidates = np.where(gains == best_gain)[0]
            # Randomly pick one
            best_idx = int(rng.choice(candidates))

            code_list.append(best_idx)
            code_set.add(best_idx)

            # Update covered
            covered[ball_indices[best_idx, :ball_size]] = True
            uncovered_count -= best_gain

        return code_list

    def local_improve(code_indices, rng, max_iters=50):
        """
        Try to remove codewords if coverage is maintained.
        """
        code_set = set(code_indices)
        if not code_set:
            return code_indices

        # Compute coverage count for each word: how many codewords cover it
        cov_count = np.zeros(N, dtype=np.int32)
        for ci in code_set:
            cov_count[ball_indices[ci, :ball_sizes[ci]]] += 1

        improved = True
        iterations = 0
        while improved and iterations < max_iters:
            improved = False
            iterations += 1
            # Shuffle codewords for randomization
            ci_list = list(code_set)
            rng.shuffle(ci_list)
            for ci in ci_list:
                if ci not in code_set:
                    continue
                # Check if removing ci still covers all words
                # Words that become uncovered: those where cov_count == 1 and covered by ci
                # i.e., words in ball of ci where cov_count == 1
                ball_words = ball_indices[ci, :ball_sizes[ci]]
                would_uncover = (cov_count[ball_words] == 1).sum()
                if would_uncover == 0:
                    # Safe to remove
                    cov_count[ball_words] -= 1
                    code_set.remove(ci)
                    improved = True

        return list(code_set)

    # Initial code from tetracode construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Convert seed_code words to indices
    def word_to_idx(word):
        idx = 0
        for i, d in enumerate(word):
            idx += d * powers[i]
        return idx

    seed_indices = [word_to_idx(w) for w in seed_code]

    best_code = seed_code
    best_len = len(seed_code)

    start_time = time.time()
    deadline = start_time + 130  # Leave 20s buffer before 150s kill

    rng = random.Random(42)

    # First, do local improvement on seed
    improved_indices = local_improve(seed_indices, rng, max_iters=200)
    if len(improved_indices) < best_len:
        best_len = len(improved_indices)
        best_code = [[(idx // powers[i]) % 3 for i in range(n)] for idx in improved_indices]

    # Now run greedy with different random seeds
    run_count = 0
    while time.time() < deadline:
        run_count += 1
        # Use a subset of seed as starting point for greedy, or empty
        # Start from empty for pure greedy, or from seed for guided greedy
        if run_count % 2 == 0:
            # Guided: start from seed
            start_idx = seed_indices
        else:
            # Pure greedy: start from empty
            start_idx = []

        code_indices = greedy_cover(start_idx, rng)
        # Local improvement
        code_indices = local_improve(code_indices, rng, max_iters=100)

        if len(code_indices) < best_len:
            best_len = len(code_indices)
            best_code = [[(idx // powers[i]) % 3 for i in range(n)] for idx in code_indices]

    return best_code
