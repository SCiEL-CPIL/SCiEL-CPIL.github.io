"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius `radius`.
    Uses a vectorized greedy set-cover followed by local improvement (swap & delete).
    Deadline guard: returns best valid code before 115s.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 150s limit, 35s safety margin

    # Precompute all words as integer indices 0..q^n-1
    N = q ** n
    if N <= 0:
        return []

    # Represent each word as a numpy array of shape (N, n)
    all_words = np.zeros((N, n), dtype=np.int32)
    for idx in range(N):
        rem = idx
        for j in range(n - 1, -1, -1):
            all_words[idx, j] = rem % q
            rem //= q

    # Build the ball mask: ball_mask[c, w] = 1 if Hamming distance between c and w <= radius
    # For efficiency, compute pairwise Hamming distances in chunks
    ball_mask = np.zeros((N, N), dtype=bool)

    # Process in chunks to avoid huge memory issues for very large N
    # For q=3, n=7, N=2187, full matrix is ~4.7M booleans = ~4.7MB, fine.
    # For larger N, we'd need chunking, but let's assume N is reasonable.
    # Compute Hamming distances: for each pair (i, j), count positions where they differ.
    # Use broadcasting: all_words[:, i] vs all_words[:, i] for each coordinate i.
    # This creates n arrays of shape (N, N) which for N=2187 is 2187*2187*7 ≈ 33M comparisons.
    # Better: compute sum of differences per coordinate.

    # Efficient Hamming distance computation using numpy
    # diff[k] = 1 if words[i,k] != words[j,k] else 0
    # Use: diff = (all_words[:, k, None] != all_words[None, :, k]).sum(axis=0)
    # This is O(N^2 * n) but vectorized.

    # For N up to ~5000, this is feasible. For N=2187, it's fine.
    hamming = np.zeros((N, N), dtype=np.int32)
    for k in range(n):
        diff = (all_words[:, k].astype(np.int32)[:, None] != all_words[:, k].astype(np.int32)[None, :])
        hamming += diff.astype(np.int32)

    ball_mask = hamming <= radius
    del hamming  # Free memory

    # Greedy set cover: iteratively pick the codeword that covers the most uncovered points
    uncovered = np.ones(N, dtype=bool)
    uncovered_count = N
    code_indices = []
    covered_count = 0

    # Precompute column sums of ball_mask for fast uncovered coverage counting
    # When we add a codeword c, the new points covered are ball_mask[c] & uncovered.
    # We want to maximize the number of newly covered points.

    while uncovered_count > 0:
        if time.time() > deadline - 5:
            break

        # Find the codeword that covers the most uncovered points
        # new_covered[i] = sum(ball_mask[i] & uncovered)
        new_covered = np.sum(ball_mask & uncovered[None, :], axis=1)
        best = np.argmax(new_covered)
        best_new = new_covered[best]

        if best_new == 0:
            # No codeword can cover any more uncovered points (shouldn't happen if code is valid)
            break

        code_indices.append(best)
        uncovered[ball_mask[best]] = False
        uncovered_count -= best_new
        covered_count += best_new

        if time.time() > deadline - 10:
            break

    # Local improvement: try to delete redundant codewords and swap to reduce size
    # We'll do multiple passes
    code_array = np.array(code_indices, dtype=np.int32)
    improved = True
    max_improvement_rounds = 50

    for round_num in range(max_improvement_rounds):
        if time.time() > deadline - 15:
            break
        improved = False

        # Delete pass: try to remove each codeword if coverage remains
        for i in range(len(code_array) - 1, -1, -1):
            if time.time() > deadline - 15:
                break

            # Remove codeword i temporarily
            remaining = np.delete(code_array, i)

            # Check if all points are still covered
            # A point is covered if any remaining codeword is within radius
            # Compute coverage: for each point, is there a remaining codeword in its ball?
            # Efficient: for each point, count how many remaining codewords cover it
            if len(remaining) == 0:
                covered_by_remaining = np.zeros(N, dtype=bool)
            else:
                # covered_by_remaining = ball_mask[remaining, :].any(axis=0)
                # This creates a (len(remaining), N) boolean matrix
                covered_by_remaining = ball_mask[remaining, :].any(axis=0)

            if np.all(covered_by_remaining):
                code_array = remaining
                improved = True
                # Continue deletion from the new end

        if not improved and time.time() > deadline - 20:
            break

        # Swap pass: try to replace each codeword with one that allows more deletions
        # Simplified: for each position, try swapping with a random candidate and see if we can delete more
        if time.time() > deadline - 20:
            break

        for i in range(len(code_array)):
            if time.time() > deadline - 20:
                break

            # Try a few random swaps
            for attempt in range(5):
                if time.time() > deadline - 20:
                    break

                # Pick a random candidate to swap with
                candidate = np.random.randint(N)
                if candidate in code_array:
                    continue

                # Try the swap
                new_code = code_array.copy()
                new_code[i] = candidate

                # Now try to delete from new_code
                new_code_sorted = new_code.copy()
                deletions = 0
                for j in range(len(new_code_sorted) - 1, -1, -1):
                    remaining = np.delete(new_code_sorted, j)
                    if len(remaining) == 0:
                        covered_by_remaining = np.zeros(N, dtype=bool)
                    else:
                        covered_by_remaining = ball_mask[remaining, :].any(axis=0)
                    if np.all(covered_by_remaining):
                        new_code_sorted = remaining
                        deletions += 1
                        if time.time() > deadline - 20:
                            break

                if len(new_code_sorted) < len(code_array):
                    code_array = new_code_sorted
                    improved = True
                    break

    # Final validity check
    if len(code_array) == 0:
        covered = np.zeros(N, dtype=bool)
    else:
        covered = ball_mask[code_array, :].any(axis=0)

    if not np.all(covered):
        # Fall back to greedy-only result if local search broke validity
        code_array = np.array(code_indices, dtype=np.int32)
        # Re-verify
        if len(code_array) == 0:
            covered2 = np.zeros(N, dtype=bool)
        else:
            covered2 = ball_mask[code_array, :].any(axis=0)
        if not np.all(covered2):
            # Should not happen, but return empty as fallback
            code_array = np.array([], dtype=np.int32)

    return [list(all_words[i]) for i in code_array]
# EVOLVE-BLOCK-END
