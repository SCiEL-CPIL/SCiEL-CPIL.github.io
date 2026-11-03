import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for safety

    # 1. Precompute t-subsets and their indices
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    t_sub_to_idx = {subset: i for i, subset in enumerate(t_subsets)}

    # 2. Precompute k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    # 3. Vectorized precomputation of block_t_indices
    t_in_k = list(itertools.combinations(range(k), t))
    n_t_in_k = len(t_in_k)

    # Build a lookup: for each position p in 0..v-1, map a global t-subset to its index
    # We'll use a different approach: build a 2D array where for each block and each local t-subset,
    # we get the global index.
    # 
    # Efficient approach: for each global t-subset, precompute a bitmask or use the fact that
    # t is small. Actually, let's use a clever trick:
    # For each block (which is a k-subset of v points), and each local t-subset (positions within the block),
    # the global t-subset is determined by the actual values.
    #
    # Let's precompute: for each of the C(v,t) t-subsets, store it as a frozenset or tuple.
    # Then for each block, for each local t-subset, look up the global index.
    #
    # To speed this up, we can precompute a mapping from sorted tuple to index.
    # But we already have t_sub_to_idx. The issue is the nested loop.
    #
    # Alternative: Use numpy to vectorize the index computation.
    # For each block, the k values are known. For each local t-subset (positions p1..pt),
    # the global t-subset is (block[p1], ..., block[pt]).
    #
    # Let's create a 3D array: blocks[k_subsets] x local_t_subsets x t, then sort each row and look up.
    # This is still complex. Let's just do the loop but optimize it.

    # Build block_t_indices efficiently
    # Precompute: for each local t-subset (positions), create an array of positions
    local_t_positions = np.array(t_in_k, dtype=np.int32)  # shape (n_t_in_k, t)

    # For each block, we need to index into it with local_t_positions
    # k_subsets is a list of tuples. Let's make it a 2D numpy array.
    k_subsets_arr = np.array(k_subsets, dtype=np.int32)  # shape (n_k, k)

    # For each block i and each local t-subset j, the global t-subset is k_subsets_arr[i, local_t_positions[j]]
    # We need to sort each of these and look up in t_sub_to_idx.
    #
    # To do this vectorized:
    # 1. Create a 3D array: shape (n_k, n_t_in_k, t)
    # 2. Sort along the last axis
    # 3. Look up each row in t_sub_to_idx

    # Step 1: Extract
    # k_subsets_arr[:, local_t_positions[j]] for each j
    # This gives (n_k, t) for each j. Stack to get (n_k, n_t_in_k, t)
    # Using advanced indexing:
    # For each j, local_t_positions[j] is a 1D array of length t
    # k_subsets_arr[:, local_t_positions[j]] has shape (n_k, t)

    # Build the 3D array
    # Shape: (n_k, n_t_in_k, t)
    # We can do this by:
    #   expanded = k_subsets_arr[:, local_t_positions]  # This won't work directly
    #
    # Let's use:
    #   idx = local_t_positions  # (n_t_in_k, t)
    #   We want: result[i, j, p] = k_subsets_arr[i, idx[j, p]]
    #   = k_subsets_arr[:, idx[j, p]] for each j, p, stacked
    #
    # This is: k_subsets_arr[:, idx] where idx is (n_t_in_k, t)
    # Advanced indexing: k_subsets_arr[:, idx] gives shape (n_k, n_t_in_k, t)
    # Let's verify: k_subsets_arr has shape (n_k, k). idx has shape (n_t_in_k, t).
    # k_subsets_arr[:, idx] -> for each i in range(n_k), for each j in range(n_t_in_k),
    #   take k_subsets_arr[i, idx[j, :]] which is a vector of length t.
    # Yes, this should work.

    extracted = k_subsets_arr[:, local_t_positions]  # shape (n_k, n_t_in_k, t)

    # Step 2: Sort along the last axis
    sorted_extracted = np.sort(extracted, axis=2)  # shape (n_k, n_t_in_k, t)

    # Step 3: Look up each row in t_sub_to_idx
    # We need to convert each (n_k, n_t_in_k, t) row to a tuple and look up.
    # This is hard to vectorize directly. Let's use a different approach.
    #
    # Alternative: Precompute a mapping from sorted t-tuple to index using a hash.
    # But we already have t_sub_to_idx as a dict.
    #
    # Let's flatten and use np.apply_along_axis or a loop over the first two dimensions.
    # Actually, let's just do a loop over n_k (75582 iterations) but vectorize over n_t_in_k (56).
    # For each block, we have 56 t-subsets. We can look them up in a vectorized way.

    # Create a lookup table: for each possible sorted t-tuple, store its index.
    # But there are C(19,5) = 11628 possible t-subsets. We can create an array indexed by
    # some encoding of the t-subset.
    #
    # Let's use a different approach: encode each t-subset as an integer.
    # For a t-subset (a1, a2, ..., at) with 0 <= a1 < a2 < ... < at < v,
    # we can encode it as a single integer using combinatorial number system.
    #
    # Actually, let's just use the dict lookup but in a loop over blocks.
    # 75582 blocks * 56 lookups = 4.2M dict lookups. This should be fast enough.

    block_t_indices = np.empty((n_k, n_t_in_k), dtype=np.int32)
    for i in range(n_k):
        # For each local t-subset, get the sorted tuple and look up
        # sorted_extracted[i] has shape (n_t_in_k, t)
        # Convert to tuples and look up
        tuples = [tuple(row) for row in sorted_extracted[i]]
        block_t_indices[i] = [t_sub_to_idx[tup] for tup in tuples]

    # 4. Calculate Weights for Weighted Greedy
    t_counts = np.bincount(block_t_indices.ravel(), minlength=n_t)
    t_counts[t_counts == 0] = 1
    t_weights = 1.0 / t_counts.astype(np.float64)

    # 5. Greedy Set Cover with Weighted Gain and Local Search
    best_solution = None
    best_size = float('inf')
    rng = random.Random(42)

    def remove_redundant(remaining, block_t_indices, n_t):
        if not remaining:
            return remaining
        coverage_count = np.zeros(n_t, dtype=np.int32)
        for bi in remaining:
            coverage_count[block_t_indices[bi]] += 1

        changed = True
        rng.shuffle(remaining)
        while changed:
            if time.time() > deadline - 2.0:
                break
            changed = False
            for i in range(len(remaining)):
                bi = remaining[i]
                ts = block_t_indices[bi]
                if np.all(coverage_count[ts] > 1):
                    coverage_count[ts] -= 1
                    remaining.pop(i)
                    changed = True
                    break
        return remaining

    def greedy_cover(uncovered, selected_blocks, block_t_indices, t_weights, n_t, deadline):
        """Greedy set cover starting from current state."""
        while True:
            if time.time() > deadline - 5.0:
                break
            if not uncovered.any():
                break

            # Calculate weighted gains
            gains = (uncovered[block_t_indices] * t_weights[block_t_indices]).sum(axis=1)

            max_gain = np.max(gains)
            if max_gain == 0:
                break

            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]

            # Pick one randomly to encourage diversity
            best_idx = rng.choice(candidates)

            selected_blocks.append(best_idx)

            # Update uncovered
            uncovered[block_t_indices[best_idx]] = False

        return selected_blocks

    restart_count = 0
    while time.time() < deadline:
        restart_count += 1

        # Reset state
        uncovered = np.ones(n_t, dtype=bool)
        selected_blocks = []

        # Greedy loop
        selected_blocks = greedy_cover(uncovered, selected_blocks, block_t_indices, t_weights, n_t, deadline)

        # Check if we finished within time limit
        if time.time() > deadline - 5.0:
            break

        # Post-optimization: Remove redundant blocks
        if selected_blocks:
            remaining = remove_redundant(list(selected_blocks), block_t_indices, n_t)

            if len(remaining) < best_size:
                best_size = len(remaining)
                best_solution = [k_subsets[i] for i in remaining]

            # Local Search: Shake and Repair
            if time.time() < deadline - 10.0 and len(remaining) > 0:
                shake_size = max(1, len(remaining) // 10)
                shake_indices = rng.sample(range(len(remaining)), min(shake_size, len(remaining)))
                to_remove = set(remaining[i] for i in shake_indices)
                new_remaining = [b for b in remaining if b not in to_remove]

                # Re-calculate uncovered based on new_remaining
                new_uncovered = np.ones(n_t, dtype=bool)
                for bi in new_remaining:
                    new_uncovered[block_t_indices[bi]] = False

                # Re-greedy to fill gaps
                new_remaining = greedy_cover(new_uncovered, new_remaining, block_t_indices, t_weights, n_t, deadline)

                # Re-remove redundants
                new_remaining = remove_redundant(new_remaining, block_t_indices, n_t)

                if len(new_remaining) < best_size:
                    best_size = len(new_remaining)
                    best_solution = [k_subsets[i] for i in new_remaining]

    if best_solution is None:
        return []

    return [list(b) for b in best_solution]
