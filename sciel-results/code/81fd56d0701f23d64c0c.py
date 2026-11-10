"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill

    space_size = q ** n
    # Map word to index
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    def idx_to_word(idx, q, n):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute all neighbors (words within radius 1) for every word in the space
    # neighbors[i] is a list of indices of words within radius 1 of word i
    neighbors = []
    for idx in range(space_size):
        w = idx_to_word(idx, q, n)
        nbrs = [idx]
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for v in range(q):
                    if v == orig:
                        continue
                    new_w = w.copy()
                    new_w[pos] = v
                    nbrs.append(word_to_idx(new_w))
        neighbors.append(nbrs)

    # Precompute neighbor sets as numpy arrays for fast coverage counting
    # coverage_matrix[i] = list of indices covered by codeword i
    # We'll use a 2D approach: for each possible codeword position (0..space_size-1),
    # we know which words it covers.

    # Greedy set cover: start with empty set, iteratively add the word that covers
    # the most currently-uncovered words.
    # But first, let's start from the tetracode direct sum as an initial upper bound,
    # then use greedy to potentially do better, or use greedy from scratch.

    # Actually, let's try a hybrid: start with greedy set cover from scratch,
    # then apply removal and local search.

    # Greedy set cover
    uncovered = np.ones(space_size, dtype=bool)
    selected = []
    selected_indices = []

    # For speed, precompute coverage arrays: cover[i] is a boolean array of length space_size
    # This is space_size x space_size = 729 x 729 = 531441 booleans ~ 0.5 MB, very manageable.
    # But building it for all 729 words upfront takes 729 * 13 ops, fine.

    # Actually, let's just compute on the fly using the neighbors list.
    # Greedy: at each step, find the word that covers the most uncovered words.
    # Naive: for each candidate, count uncovered neighbors. O(space_size * 13) per step.
    # With ~73 steps, that's 73 * 729 * 13 ~ 693k ops. Very fast.

    while uncovered.any():
        if time.time() > deadline - 10:
            break

        best_idx = -1
        best_count = -1
        # Sample a subset of candidates for speed if needed, but 729 is small
        for cand_idx in range(space_size):
            if not uncovered[cand_idx]:
                # If the candidate itself is already covered, it might still be useful
                # to cover other uncovered words, so we still consider it.
                pass
            # Count how many uncovered words this candidate covers
            count = 0
            for nbr_idx in neighbors[cand_idx]:
                if uncovered[nbr_idx]:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = cand_idx

        if best_idx == -1 or best_count <= 0:
            break

        # Select this codeword
        selected.append(idx_to_word(best_idx, q, n))
        selected_indices.append(best_idx)

        # Update uncovered
        for nbr_idx in neighbors[best_idx]:
            uncovered[nbr_idx] = False

    # Now we have a greedy solution. Let's see how many codewords we have.
    # Greedy set cover for this problem typically gives a good but not optimal solution.
    # Now apply iterative removal to reduce further.

    # Build coverage counts
    coverage_count = np.zeros(space_size, dtype=np.int32)
    for ci in selected_indices:
        for nbr_idx in neighbors[ci]:
            coverage_count[nbr_idx] += 1

    # Iterative removal
    removed = [False] * len(selected)
    for _pass in range(50):
        if time.time() > deadline - 10:
            break
        removed_any = False
        for ci in range(len(selected)):
            if removed[ci]:
                continue
            # Check if removing selected[ci] leaves any word uncovered
            can_remove = True
            for nbr_idx in neighbors[selected_indices[ci]]:
                if coverage_count[nbr_idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove it
                removed[ci] = True
                removed_any = True
                for nbr_idx in neighbors[selected_indices[ci]]:
                    coverage_count[nbr_idx] -= 1
        if not removed_any:
            break

    # Build result after removal
    result_indices = [selected_indices[i] for i in range(len(selected)) if not removed[i]]
    result = [idx_to_word(idx, q, n) for idx in result_indices]
    coverage_count = np.zeros(space_size, dtype=np.int32)
    for ci in result_indices:
        for nbr_idx in neighbors[ci]:
            coverage_count[nbr_idx] += 1

    # Local search: try swapping codewords
    if time.time() < deadline - 5:
        for iteration in range(200):
            if time.time() > deadline - 3:
                break

            improved = False
            # Try adding a random candidate and removing 2+ existing
            for trial in range(30):
                if time.time() > deadline - 3:
                    break

                cand_idx = random.randint(0, space_size - 1)
                if cand_idx in result_indices:
                    continue

                # Simulate adding cand_idx
                new_cov = coverage_count.copy()
                for nbr_idx in neighbors[cand_idx]:
                    new_cov[nbr_idx] += 1

                # Try to greedily remove codewords
                temp_removed = [False] * len(result_indices)
                removable = 0
                for ri, rcw_idx in enumerate(result_indices):
                    if temp_removed[ri]:
                        continue
                    can_remove = True
                    for nbr_idx in neighbors[rcw_idx]:
                        if new_cov[nbr_idx] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        temp_removed[ri] = True
                        removable += 1
                        for nbr_idx in neighbors[rcw_idx]:
                            new_cov[nbr_idx] -= 1

                if removable >= 2:
                    # Apply
                    new_result_indices = [result_indices[ri] for ri in range(len(result_indices)) if not temp_removed[ri]]
                    new_result_indices.append(cand_idx)
                    result_indices = new_result_indices
                    result = [idx_to_word(idx, q, n) for idx in result_indices]
                    coverage_count = new_cov
                    improved = True
                    break

            if not improved:
                break

    # Final validity check
    final_cov = np.zeros(space_size, dtype=np.int32)
    for ci in result_indices:
        for nbr_idx in neighbors[ci]:
            final_cov[nbr_idx] += 1

    if not (final_cov > 0).all():
        # Fallback to tetracode direct sum
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

    return result
# EVOLVE-BLOCK-END
