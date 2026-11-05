import itertools
import random
import time
import math
import numpy as np

# Global Precomputation
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets and their masks
all_blocks = []
all_masks = []
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]
subset_frequencies = [0] * N_SUBSETS

for block in itertools.combinations(range(V), K):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(len(all_blocks))
        subset_frequencies[idx] += 1
    all_blocks.append(block)
    all_masks.append(mask)

N_BLOCKS = len(all_blocks)

# Precompute per-block list of covered subset indices (35 each) for fast iteration
block_bit_indices = []
for mask in all_masks:
    bits = []
    mm = mask
    while mm:
        lsb = mm & -mm
        bits.append(lsb.bit_length() - 1)
        mm ^= lsb
    block_bit_indices.append(bits)

# Weighted greedy weights: rare subsets get higher priority
weights = [1.0 / f for f in subset_frequencies]

# Precompute coverage matrix as numpy array for fast vectorized greedy
coverage_matrix = np.zeros((N_BLOCKS, N_SUBSETS), dtype=np.int8)
for i, bits in enumerate(block_bit_indices):
    for b in bits:
        coverage_matrix[i, b] = 1

# weights as numpy array
weights_np = np.array(weights, dtype=np.float64)
weights_uniform = np.ones(N_SUBSETS, dtype=np.float64)
weights_sqrt_inv = np.array([1.0 / math.sqrt(f) for f in subset_frequencies], dtype=np.float64)


def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def fast_greedy(weight_vector=None):
        """Vectorized weighted greedy using numpy for speed."""
        if weight_vector is None:
            weight_vector = weights_np
        uncovered = np.ones(N_SUBSETS, dtype=np.int8)
        chosen = []
        remaining = N_SUBSETS

        while remaining > 0:
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

    # 1. Try multiple greedy initializations with different weightings
    greedy_results = []
    for w_vec in [weights_np, weights_uniform, weights_sqrt_inv]:
        if time.time() > deadline - 10:
            break
        result = fast_greedy(w_vec)
        if result:
            greedy_results.append(result)

    # Pick the best greedy result
    if greedy_results:
        best_greedy = min(greedy_results, key=len)
    else:
        best_greedy = fast_greedy(weights_np)

    current_blocks = best_greedy.copy()
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()

    # Build counts
    counts = [0] * N_SUBSETS
    for idx in current_blocks:
        for bit in block_bit_indices[idx]:
            counts[bit] += 1

    def add_block(idx):
        current_blocks.append(idx)
        for bit in block_bit_indices[idx]:
            counts[bit] += 1

    def remove_block(pos):
        idx = current_blocks[pos]
        for bit in block_bit_indices[idx]:
            counts[bit] -= 1
        current_blocks.pop(pos)

    def is_redundant(pos):
        idx = current_blocks[pos]
        for bit in block_bit_indices[idx]:
            if counts[bit] == 1:
                return False
        return True

    def find_redundant():
        for pos in range(len(current_blocks)):
            if is_redundant(pos):
                return pos
        return -1

    # 2. Local search: repeatedly remove redundant blocks, then do 2-for-1 exchanges and swaps
    no_improve = 0
    max_no_improve = 300

    while time.time() < deadline:
        # Phase A: prune all redundant blocks
        progress = True
        while progress:
            progress = False
            for pos in range(len(current_blocks)):
                if is_redundant(pos):
                    remove_block(pos)
                    progress = True
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = current_blocks.copy()
                    no_improve = 0
                    break

        if len(current_blocks) == 0:
            break

        # Phase B: Try 2-for-1 exchange
        found_2_1 = False
        if len(current_blocks) >= 2:
            attempts = 80
            for _ in range(attempts):
                if time.time() > deadline:
                    break
                i1 = random.randint(0, len(current_blocks) - 1)
                i2 = random.randint(0, len(current_blocks) - 1)
                if i1 == i2:
                    continue

                b1_idx = current_blocks[i1]
                b2_idx = current_blocks[i2]

                # Find bits that become uncovered if we remove both b1 and b2
                uncovered_bits = set()

                # Bits in b1
                for bit in block_bit_indices[b1_idx]:
                    if counts[bit] == 1:
                        uncovered_bits.add(bit)
                    elif counts[bit] == 2:
                        # Check if b2 also covers this bit
                        if b2_idx in blocks_containing_subset[bit]:
                            uncovered_bits.add(bit)

                # Bits in b2 (that weren't already handled)
                for bit in block_bit_indices[b2_idx]:
                    if bit in uncovered_bits:
                        continue
                    if counts[bit] == 1:
                        uncovered_bits.add(bit)
                    elif counts[bit] == 2:
                        # If in b1, already handled; if not in b1, count drops to 1, still covered
                        pass

                if not uncovered_bits:
                    # Both are redundant together - should have been caught in Phase A
                    continue

                # Find a block b3 that covers all uncovered_bits
                bits_list = list(uncovered_bits)
                bits_list.sort(key=lambda b: len(blocks_containing_subset[b]))

                candidates = set(blocks_containing_subset[bits_list[0]])
                for b in bits_list[1:]:
                    if not candidates:
                        break
                    candidates &= set(blocks_containing_subset[b])

                if candidates:
                    candidates.discard(b1_idx)
                    candidates.discard(b2_idx)
                    if candidates:
                        b3_idx = random.choice(list(candidates))
                        # Remove i1 and i2, add b3_idx
                        # Remove higher index first to keep indices valid
                        if i1 > i2:
                            remove_block(i1)
                            remove_block(i2)
                        else:
                            remove_block(i2)
                            remove_block(i1)
                        add_block(b3_idx)

                        if len(current_blocks) < best_len:
                            best_len = len(current_blocks)
                            best_blocks = current_blocks.copy()

                        found_2_1 = True
                        no_improve = 0
                        break

        if found_2_1:
            continue

        # Phase C: Try 1-for-1 swap (neutral move for diversification)
        i = random.randint(0, len(current_blocks) - 1)
        idx_i = current_blocks[i]
        uncovered_bits = [bit for bit in block_bit_indices[idx_i] if counts[bit] == 1]

        if not uncovered_bits:
            remove_block(i)
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
            no_improve = 0
            continue

        if len(uncovered_bits) <= 5:
            uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
            candidates = set(blocks_containing_subset[uncovered_bits[0]])
            for b in uncovered_bits[1:]:
                if not candidates:
                    break
                candidates &= set(blocks_containing_subset[b])

            if candidates:
                candidates.discard(idx_i)
                if candidates:
                    j = random.choice(list(candidates))
                    remove_block(i)
                    add_block(j)
                    no_improve = 0
                    continue

        no_improve += 1
        if no_improve > max_no_improve:
            # Perturbation: Add a random block, then try to prune
            c = random.randint(0, N_BLOCKS - 1)
            if c not in current_blocks:
                add_block(c)
            no_improve = 0
            max_no_improve = min(500, max_no_improve * 2)

    # Final cleanup: one more pass of redundant removal
    for pos in range(len(current_blocks) - 1, -1, -1):
        if is_redundant(pos):
            remove_block(pos)
    if len(current_blocks) < best_len:
        best_len = len(current_blocks)
        best_blocks = current_blocks.copy()

    return [list(all_blocks[i]) for i in best_blocks]
