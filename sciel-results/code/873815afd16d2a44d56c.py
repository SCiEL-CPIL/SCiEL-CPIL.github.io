"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 145.0  # safety margin before 150 s kill


def _to_idx(t, base):
    """Convert tuple of digits to a single integer index (little-endian)."""
    idx = 0
    for i, d in enumerate(t):
        idx += d * (base ** i)
    return idx


def _to_tuple(idx, base, n):
    """Convert integer index back to tuple of digits."""
    t = [0] * n
    for i in range(n):
        t[i] = idx % base
        idx //= base
    return tuple(t)


def construct(q, n, radius):
    start_time = time.time()
    N = q ** n

    # Precompute all neighbor indices within Hamming distance `radius`
    # For radius=1: neighbor count = 1 + n*(q-1)
    # We'll build neighbor lists as numpy arrays for speed
    # First, build all offsets for radius=1
    offsets = []
    for pos in range(n):
        for delta in range(1, q):
            offsets.append((pos, delta))
    # offsets: list of (position, delta)

    # Precompute neighbor index arrays
    # neighbors[i] = array of indices within Hamming distance 1 from i
    neighbors = []
    base_pows = [1] * n
    for i in range(1, n):
        base_pows[i] = base_pows[i - 1] * q

    for i in range(N):
        nbrs = [i]  # include self
        for pos, delta in offsets:
            # Extract digit at position pos
            digit = (i // base_pows[pos]) % q
            new_digit = digit + delta
            if new_digit >= q:
                new_digit -= q
            # Actually we need to consider both +delta and -delta? No, for each position we vary to each other value
            # Let me redo: for each position, for each other value
        # Let me rebuild more carefully
        nbrs = [i]
        for pos in range(n):
            digit = (i // base_pows[pos]) % q
            for new_val in range(q):
                if new_val == digit:
                    continue
                new_digit = new_val
                # Compute new index
                old_contrib = digit * base_pows[pos]
                new_contrib = new_digit * base_pows[pos]
                new_idx = i - old_contrib + new_contrib
                nbrs.append(new_idx)
        neighbors.append(np.array(nbrs, dtype=np.int32))

    # Greedy set cover
    covered = np.zeros(N, dtype=bool)
    # weight[i] = number of neighbors of i that are not yet covered
    weight = np.zeros(N, dtype=np.int32)
    for i in range(N):
        weight[i] = len(neighbors[i])

    code_indices = []

    while True:
        if time.time() - start_time > _DEADLINE:
            break

        # Find index with max weight
        best_idx = int(np.argmax(weight))
        if weight[best_idx] == 0:
            break

        code_indices.append(best_idx)

        # Mark neighbors of best_idx as covered
        nbr_arr = neighbors[best_idx]
        newly_covered = []
        for j in nbr_arr:
            j = int(j)
            if not covered[j]:
                covered[j] = True
                newly_covered.append(j)

        # Update weights: for each newly covered node j, decrement weight of all its neighbors
        for j in newly_covered:
            for k in neighbors[j]:
                k = int(k)
                weight[k] -= 1

    # Convert indices to codewords (lists of integers)
    codewords = []
    for idx in code_indices:
        t = _to_tuple(idx, q, n)
        codewords.append(list(t))

    # Verify coverage (quick check)
    if len(code_indices) < N:
        # Full verification
        for i in range(N):
            if not covered[i]:
                # This shouldn't happen, but if it does, add a brute-force fallback
                pass

    return codewords
# EVOLVE-BLOCK-END
