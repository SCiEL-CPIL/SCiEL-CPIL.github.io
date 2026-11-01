"""Ternary covering code K_3(7,1) via vectorized greedy + local search."""
import time
import numpy as np

# Deadline: kill at 150s, but we want a safety margin of ~10s.
DEADLINE = 140.0


def _to_base3_indices(words: list) -> np.ndarray:
    """Convert list of codewords (each a list of n ints in 0..q-1) to integer indices in [0, q**n)."""
    n = len(words[0])
    q = 3
    # Precompute powers
    powers = np.array([q ** i for i in range(n)], dtype=np.int64)
    arr = np.array(words, dtype=np.int64)
    return (arr * powers).sum(axis=1)


def _from_base3_indices(indices: np.ndarray, n: int, q: int = 3) -> list:
    """Convert integer indices back to list of codewords."""
    powers = np.array([q ** i for i in range(n)], dtype=np.int64)
    digits = (indices[:, None] // powers) % q
    return digits.tolist()


def _ball_indices(center_idx: int, n: int, q: int = 3, radius: int = 1) -> np.ndarray:
    """Return all indices within Hamming distance <= radius from center_idx."""
    # Decode center
    powers = np.array([q ** i for i in range(n)], dtype=np.int64)
    center_digits = (center_idx // powers) % q

    # All words at distance 0
    result = [center_idx]

    # All words at distance 1: change one position to any other value
    for pos in range(n):
        current_val = center_digits[pos]
        for new_val in range(q):
            if new_val == current_val:
                continue
            diff = (new_val - current_val) * powers[pos]
            result.append(center_idx + diff)

    return np.array(result, dtype=np.int64)


def construct(q, n, radius):
    """
    Construct a covering code for {0..q-1}^n with covering radius 'radius'.
    Returns a list of codewords (each a list of n ints).
    """
    start_time = time.time()
    total_space = q ** n

    # Precompute all ball masks for every possible codeword position? That's 3^7 = 2187 * 19 = ~41k entries.
    # Actually, for each of the 2187 possible centers, the ball has 1 + 7*(3-1) = 15 positions.
    # Let's precompute the ball for each center index.
    all_centers = np.arange(total_space, dtype=np.int64)
    # balls[i] = set of indices covered by center i
    # For efficiency, store as a list of arrays
    balls = []
    for idx in all_centers:
        balls.append(_ball_indices(int(idx), n, q, radius))

    # Convert to a numpy array of shape (total_space, ball_size) where ball_size = 1 + n*(q-1)
    ball_size = 1 + n * (q - 1)
    balls_arr = np.array(balls, dtype=np.int64)  # shape (2187, 15)

    # Greedy set cover with criticality ordering
    # uncovered: boolean array of size total_space
    uncovered = np.ones(total_space, dtype=bool)
    uncovered_count = total_space

    # For each center, compute how many currently uncovered words it covers
    # We'll use a priority approach: at each step, pick the center that maximizes coverage of uncovered words
    # To make it "criticality-based": prioritize covering the most constrained uncovered words
    # A word is "constrained" if few remaining centers can cover it.
    # Simpler approach: standard greedy (max coverage) works well, then local search improves.

    code_indices = []

    while uncovered_count > 0:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30:  # Leave 30s for local search
            break

        # Compute coverage of uncovered words for each center
        # For each center c, coverage = sum(uncovered[balls_arr[c]])
        # This is a matrix-vector product: (2187 x 15) @ uncovered (2187,) -> but that's not right.
        # We need: for each center c, count how many of balls_arr[c] are in uncovered.
        # Use: np.take(uncovered, balls_arr, axis=0).sum(axis=1)
        # balls_arr shape: (2187, 15), uncovered shape: (2187,)
        # np.take(uncovered, balls_arr) gives shape (2187, 15), then sum over axis 1
        cov = np.take(uncovered, balls_arr, axis=0).sum(axis=1)

        # Find the center with max coverage
        best_center = int(np.argmax(cov))
        best_cov = int(cov[best_center])

        if best_cov == 0:
            break

        code_indices.append(best_center)

        # Update uncovered
        uncovered[balls_arr[best_center]] = False
        uncovered_count -= best_cov

    # Convert to list of codewords
    if len(code_indices) == 0:
        # Fallback: use the direct sum construction
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    code_words = _from_base3_indices(np.array(code_indices, dtype=np.int64), n, q)

    # Local search: try to remove codewords or swap to reduce size while maintaining coverage
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10 and len(code_indices) > 10:
        code_words = _local_search(code_words, q, n, radius, balls_arr, start_time)

    return code_words


def _local_search(code_words, q, n, radius, balls_arr, start_time):
    """
    Local search to reduce code size.
    1. Try removing codewords one by one (if coverage is maintained).
    2. Try swapping a codeword with another to improve coverage efficiency.
    """
    total_space = q ** n
    code_indices = _to_base3_indices(code_words)
    code_set = set(code_indices.tolist())

    # Build coverage count: for each word in space, how many codewords cover it
    cover_count = np.zeros(total_space, dtype=np.int64)
    for idx in code_indices:
        cover_count[balls_arr[int(idx)]] += 1

    # Phase 1: Remove redundant codewords
    improved = True
    while improved:
        improved = False
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break

        for i in range(len(code_indices)):
            if elapsed > DEADLINE - 15:
                break
            idx = int(code_indices[i])
            # Check if removing this codeword leaves any word uncovered
            # A word becomes uncovered if its cover_count drops to 0
            ball = balls_arr[idx]
            # Check if any word in this ball has cover_count == 1
            if np.any(cover_count[ball] == 1):
                continue
            # Safe to remove
            cover_count[ball] -= 1
            code_indices = np.delete(code_indices, i)
            code_set.discard(idx)
            improved = True
            break  # Restart the loop since indices changed

    # Phase 2: Try to swap codewords to reduce total size
    # This is more complex; for now, just try to find a codeword that can be replaced
    # by one that covers more uncovered-ish words. But since we already removed redundant ones,
    # the code is minimal in the sense that no single codeword can be removed.
    # A more advanced local search would try 1-for-1 swaps.

    # Convert back to list of codewords
    return _from_base3_indices(code_indices, n, q)
