"""
Covering design C(19,7,4) via vectorized greedy + local search.
Combines:
  - Precomputation of all blocks and coverage matrix (inspiration 1)
  - Vectorized weighted greedy with multiple weightings (inspiration 1)
  - Local search with redundancy pruning and 1-for-1 swaps (inspiration 1)
  - Time-bounded deadline guard (current program)
"""
import itertools
import random
import time
import math
import numpy as np

# Global precomputation for v=19, k=7, t=4
V, K, T = 19, 7, 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets
all_blocks = list(itertools.combinations(range(V), K))
N_BLOCKS = len(all_blocks)

# Precompute per-block list of covered subset indices (35 each)
block_bits = []
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]
subset_frequencies = [0] * N_SUBSETS

for bi, block in enumerate(all_blocks):
    bits = []
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        bits.append(idx)
        blocks_containing_subset[idx].append(bi)
        subset_frequencies[idx] += 1
    block_bits.append(bits)

# Coverage matrix: coverage_matrix[i, j] = 1 if block i covers subset j
coverage_matrix = np.zeros((N_BLOCKS, N_SUBSETS), dtype=np.int8)
for i, bits in enumerate(block_bits):
    for b in bits:
        coverage_matrix[i, b] = 1

# Weight vectors
weights_inv = np.array([1.0 / f for f in subset_frequencies], dtype=np.float64)
weights_uniform = np.ones(N_SUBSETS, dtype=np.float64)
weights_sqrt_inv = np.array([1.0 / math.sqrt(f) for f in subset_frequencies], dtype=np.float64)


def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters: simple greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k = list(itertools.combinations(range(v), k))
        random.shuffle(all_k)
        for b in all_k:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 110  # safety margin

    def fast_greedy(weight_vector):
        """Vectorized weighted greedy using numpy."""
        uncovered = np.ones(N_SUBSETS, dtype=np.int8)
        chosen = []
        remaining = N_SUBSETS
        while remaining > 0:
            if time.time() > deadline - 2:
                break
            gains = coverage_matrix @ (weight_vector * uncovered)
            best_idx = int(np.argmax(gains))
            best_gain = gains[best_idx]
            if best_gain <= 0:
                break
            chosen.append(best_idx)
            newly_covered = coverage_matrix[best_idx]
            still_uncovered = uncovered & (1 - newly_covered)
            remaining = int(np.sum(still_uncovered))
            uncovered = still_uncovered
        return chosen

    # Try multiple greedy initializations
    greedy_results = []
    for w_vec in [weights_inv, weights_uniform, weights_sqrt_inv]:
        if time.time() > deadline - 10:
            break
        result = fast_greedy(w_vec)
        if result and len(result) > 0:
            greedy_results.append(result)

    if not greedy_results:
        greedy_results = [fast_greedy(weights_inv)]

    best_greedy = min(greedy_results, key=len)
    current = list(best_greedy)
    best_len = len(current)
    best_blocks = current.copy()

    # Build coverage counts
    counts = np.zeros(N_SUBSETS, dtype=np.int32)
    for idx in current:
        counts[block_bits[idx]] += 1

    def add_block(idx):
        current.append(idx)
        counts[block_bits[idx]] += 1

    def remove_block(pos):
        idx = current[pos]
        counts[block_bits[idx]] -= 1
        current.pop(pos)

    def is_redundant(pos):
        idx = current[pos]
        for bit in block_bits[idx]:
            if counts[bit] == 1:
                return False
        return True

    def prune_redundant():
        """Remove all redundant blocks. Returns number removed."""
        removed = 0
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for pos in range(len(current) - 1, -1, -1):
                if time.time() > deadline - 5:
                    return removed
                idx = current[pos]
                redundant = True
                for bit in block_bits[idx]:
                    if counts[bit] == 1:
                        redundant = False
                        break
                if redundant:
                    counts[block_bits[idx]] -= 1
                    current.pop(pos)
                    removed += 1
                    changed = True
                    if len(current) < best_len:
                        nonlocal best_len, best_blocks
                        best_len = len(current)
                        best_blocks = current.copy()
        return removed

    # Local search: prune + swaps
    rng = random.Random(123)
    no_improve = 0
    max_no_improve = 60

    while time.time() < deadline - 10 and no_improve < max_no_improve:
        # Phase A: prune redundant
        n_removed = prune_redundant()
        if n_removed > 0:
            no_improve = 0
            continue

        # Phase B: try 1-for-1 swaps
        # Find block with fewest unique (count==1) subsets
        n_unique_list = []
        for pos in range(len(current)):
            idx = current[pos]
            nu = 0
            for bit in block_bits[idx]:
                if counts[bit] == 1:
                    nu += 1
            n_unique_list.append(nu)

        # Sort by number of unique subsets (fewest first - most swappable)
        order = sorted(range(len(current)), key=lambda i: n_unique_list[i])

        swap_done = False
        for pos in order:
            if time.time() > deadline - 10:
                break
            if n_unique_list[pos] >= 10:
                break  # too many unique to swap easily

            idx_i = current[pos]
            unique_bits = [bit for bit in block_bits[idx_i] if counts[bit] == 1]

            if not unique_bits:
                # Fully redundant (shouldn't happen after prune, but just in case)
                counts[block_bits[idx_i]] -= 1
                current.pop(pos)
                if len(current) < best_len:
                    best_len = len(current)
                    best_blocks = current.copy()
                no_improve = 0
                swap_done = True
                break

            # Try to find a replacement block that covers all unique_bits
            # and doesn't add too many new unique subsets
            best_candidate = -1
            best_score = float('inf')

            # Candidates: blocks that cover all unique_bits
            # Intersect the lists of blocks containing each unique bit
            candidate_set = None
            for bit in unique_bits:
                blist = blocks_containing_subset[bit]
                if candidate_set is None:
                    candidate_set = set(blist)
                else:
                    candidate_set.intersection_update(blist)

            if candidate_set is None or len(candidate_set) == 0:
                continue

            # Sample candidates
            candidates = list(candidate_set)
            if len(candidates) > 50:
                candidates = rng.sample(candidates, 50)

            for cand in candidates:
                if cand == idx_i:
                    continue
                # Score: number of new unique subsets introduced
                score = 0
                for bit in block_bits[cand]:
                    if counts[bit] == 1:
                        score += 1
                if score < best_score:
                    best_score = score
                    best_candidate = cand
                if score == 0:
                    break

            if best_candidate >= 0 and best_score < n_unique_list[pos]:
                # Do the swap
                counts[block_bits[idx_i]] -= 1
                current[pos] = best_candidate
                counts[block_bits[best_candidate]] += 1
                if len(current) < best_len:
                    best_len = len(current)
                    best_blocks = current.copy()
                no_improve = 0
                swap_done = True
                break

        if not swap_done:
            no_improve += 1

    # Final prune
    prune_redundant()
    if len(current) < best_len:
        best_len = len(current)
        best_blocks = current.copy()

    # Verify coverage
    final_cover = np.zeros(N_SUBSETS, dtype=np.int32)
    for idx in best_blocks:
        final_cover[block_bits[idx]] += 1

    if np.any(final_cover == 0):
        # Fix uncovered subsets
        for i in range(N_SUBSETS):
            if final_cover[i] == 0:
                if time.time() > deadline - 2:
                    break
                target = subsets[i]
                remaining_pts = [x for x in range(V) if x not in target]
                suffix = tuple(rng.sample(remaining_pts, K - T))
                new_block = tuple(sorted(target + suffix))
                # Find its index
                new_idx = all_blocks.index(new_block)
                if new_idx not in best_blocks:
                    best_blocks.append(new_idx)
                    final_cover[block_bits[new_idx]] += 1

    return [list(all_blocks[idx]) for idx in best_blocks]
