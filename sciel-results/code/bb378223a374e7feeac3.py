import itertools
import time
import random
import numpy as np

# Global Precomputation
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets and their data
all_blocks = list(itertools.combinations(range(V), K))
N_BLOCKS = len(all_blocks)

all_masks = [0] * N_BLOCKS
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]
subset_frequencies = [0] * N_SUBSETS
block_bit_indices = [[] for _ in range(N_BLOCKS)]

for i, block in enumerate(all_blocks):
    mask = 0
    bits = []
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(i)
        subset_frequencies[idx] += 1
        bits.append(idx)
    all_masks[i] = mask
    block_bit_indices[i] = bits

# Precompute weights for weighted greedy: prioritize rare subsets
weights = np.array([1.0 / f for f in subset_frequencies], dtype=np.float64)

# Dense incidence matrix for fast vectorized gain calculation
# M[i, j] = 1 if block i covers subset j, else 0
incidence_matrix = np.zeros((N_BLOCKS, N_SUBSETS), dtype=np.uint8)
for i in range(N_BLOCKS):
    for bit in block_bit_indices[i]:
        incidence_matrix[i, bit] = 1

# Weighted incidence matrix: W[i, j] = incidence[i, j] * weights[j]
weighted_incidence = incidence_matrix * weights

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters
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
    deadline = start_time + 145  # Safety margin

    # State
    counts = np.zeros(N_SUBSETS, dtype=np.int32)
    current_blocks = []
    current_block_set = set()
    is_uncovered = np.ones(N_SUBSETS, dtype=bool)
    local_in_sol = np.zeros(N_BLOCKS, dtype=bool)

    def add_block(idx):
        current_blocks.append(idx)
        current_block_set.add(idx)
        local_in_sol[idx] = True
        for bit in block_bit_indices[idx]:
            counts[bit] += 1
            if counts[bit] == 1:
                is_uncovered[bit] = False

    def remove_block(idx):
        current_block_set.remove(idx)
        pos = current_blocks.index(idx)
        current_blocks[pos] = current_blocks[-1]
        current_blocks.pop()
        local_in_sol[idx] = False
        for bit in block_bit_indices[idx]:
            counts[bit] -= 1
            if counts[bit] == 0:
                is_uncovered[bit] = True

    def is_redundant(idx):
        for bit in block_bit_indices[idx]:
            if counts[bit] == 1:
                return False
        return True

    # 1. Greedy Initialization
    while is_uncovered.any():
        if time.time() > deadline - 10:
            break
        gains = weighted_incidence @ is_uncovered
        gains[local_in_sol] = -1.0
        best_idx = np.argmax(gains)
        if gains[best_idx] <= 0:
            break
        add_block(best_idx)

    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()

    # 2. Local Search
    no_improve_count = 0
    max_no_improve = 20

    while time.time() < deadline - 5:
        # Phase 1: Prune
        removed_any = False
        indices = list(range(len(current_blocks)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 5:
                break
            if i >= len(current_blocks):
                continue
            idx = current_blocks[i]
            if is_redundant(idx):
                remove_block(idx)
                removed_any = True
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                break

        if removed_any:
            no_improve_count = 0
            continue

        # Phase 2: Swap
        found_swap = False
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            if not current_blocks:
                break
            
            i = random.randint(0, len(current_blocks) - 1)
            idx = current_blocks[i]
            
            unique_bits = [b for b in block_bit_indices[idx] if counts[b] == 1]
            
            if not unique_bits:
                remove_block(idx)
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                no_improve_count = 0
                found_swap = True
                break
            else:
                if len(unique_bits) <= 4:
                    candidates = set(blocks_containing_subset[unique_bits[0]])
                    for b in unique_bits[1:]:
                        if not candidates:
                            break
                        candidates &= set(blocks_containing_subset[b])
                    
                    valid_candidates = [c for c in candidates if c not in current_block_set]
                    
                    if valid_candidates:
                        best_cand = None
                        best_score = -1
                        for c in valid_candidates:
                            score = len(unique_bits)
                            for b in block_bit_indices[c]:
                                if counts[b] == 0:
                                    score += 0.5
                            if score > best_score:
                                best_score = score
                                best_cand = c
                        
                        if best_cand is not None:
                            remove_block(idx)
                            add_block(best_cand)
                            found_swap = True
                            no_improve_count = 0
                            break

        if not found_swap:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation
                n_remove = random.randint(2, 5)
                block_scores = []
                for idx in current_blocks:
                    unique_count = sum(1 for b in block_bit_indices[idx] if counts[b] == 1)
                    block_scores.append((unique_count, idx))
                block_scores.sort(key=lambda x: x[0])
                
                to_remove = [idx for _, idx in block_scores[:min(n_remove, len(current_blocks))]]
                for idx in to_remove:
                    remove_block(idx)
                
                # Re-greedy
                if time.time() < deadline - 10:
                    while is_uncovered.any():
                        if time.time() > deadline - 5:
                            break
                        gains = weighted_incidence @ is_uncovered
                        gains[local_in_sol] = -1.0
                        best_idx = np.argmax(gains)
                        if gains[best_idx] <= 0:
                            break
                        add_block(best_idx)
                
                no_improve_count = 0
                max_no_improve = min(100, max_no_improve * 2)

    # 3. Final Completion
    while is_uncovered.any():
        if time.time() > deadline:
            break
        gains = weighted_incidence @ is_uncovered
        gains[local_in_sol] = -1.0
        best_idx = np.argmax(gains)
        if gains[best_idx] <= 0:
            break
        add_block(best_idx)

    return [list(all_blocks[i]) for i in current_blocks]
