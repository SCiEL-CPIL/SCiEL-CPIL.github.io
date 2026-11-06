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

# Precompute bitmasks for each block as Python integers for fast bitwise operations
block_bitmasks = [0] * N_BLOCKS
for i, mask in enumerate(all_masks):
    block_bitmasks[i] = mask

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters (not expected in this specific task)
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
    deadline = start_time + 115  # Safety margin below 120s kill

    def greedy_init():
        """
        Weighted greedy initialization.
        """
        uncovered = np.ones(N_SUBSETS, dtype=np.int8)
        chosen = []
        remaining = N_SUBSETS
        
        while remaining > 0:
            weighted_uncovered = weights_np * uncovered
            gains = coverage_matrix @ weighted_uncovered
            best_idx = int(np.argmax(gains))
            if gains[best_idx] <= 0:
                break
            chosen.append(best_idx)
            newly_covered = coverage_matrix[best_idx]
            still_uncovered = uncovered & (1 - newly_covered)
            remaining = int(np.sum(still_uncovered))
            uncovered = still_uncovered
        
        return chosen

    def repair_greedy(initial_blocks):
        """
        Repair a set of blocks by adding more blocks until all subsets are covered.
        """
        # Calculate current coverage
        covered_mask = 0
        for idx in initial_blocks:
            covered_mask |= block_bitmasks[idx]
        
        # Identify uncovered subsets
        full_mask = (1 << N_SUBSETS) - 1
        uncovered_mask = full_mask & ~covered_mask
        
        if uncovered_mask == 0:
            return initial_blocks.copy()
        
        # Convert uncovered_mask to numpy array for vectorized greedy
        uncovered_vec = np.zeros(N_SUBSETS, dtype=np.int8)
        m = uncovered_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            uncovered_vec[idx] = 1
            m ^= lsb
            
        chosen = initial_blocks.copy()
        remaining = int(np.sum(uncovered_vec))
        
        while remaining > 0 and time.time() < deadline - 2:
            weighted_uncovered = weights_np * uncovered_vec
            gains = coverage_matrix @ weighted_uncovered
            best_idx = int(np.argmax(gains))
            if gains[best_idx] <= 0:
                break
            chosen.append(best_idx)
            newly_covered = coverage_matrix[best_idx]
            still_uncovered = uncovered_vec & (1 - newly_covered)
            remaining = int(np.sum(still_uncovered))
            uncovered_vec = still_uncovered
        
        return chosen

    def prune_solution(block_indices):
        """
        Remove redundant blocks using bitmasks.
        A block is redundant if all its subsets are covered by other blocks.
        """
        # Calculate covered mask for all blocks
        covered_mask = 0
        for idx in block_indices:
            covered_mask |= block_bitmasks[idx]
        
        # Find redundant blocks
        to_remove = []
        for idx in block_indices:
            b_mask = block_bitmasks[idx]
            other_mask = covered_mask & ~b_mask
            # If b_mask is subset of other_mask, then b_mask & ~other_mask == 0
            if (b_mask & ~other_mask) == 0:
                to_remove.append(idx)
        
        if to_remove:
            remove_set = set(to_remove)
            candidate_indices = [i for i in block_indices if i not in remove_set]
            
            # Verify candidate_indices
            new_covered = 0
            for idx in candidate_indices:
                new_covered |= block_bitmasks[idx]
            
            if new_covered == covered_mask:
                return candidate_indices
            else:
                return block_indices
        else:
            return block_indices

    # 1. Initial Greedy
    current_blocks = greedy_init()
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()
    
    # Initial Prune
    current_blocks = prune_solution(current_blocks)
    if len(current_blocks) < best_len:
        best_len = len(current_blocks)
        best_blocks = current_blocks.copy()

    # 2. Iterated Local Search (Perturb + Repair)
    while time.time() < deadline - 5:
        # Perturbation: Remove 3-5 random blocks
        if len(current_blocks) > 10:
            n_remove = random.randint(3, 5)
            to_remove = random.sample(current_blocks, min(n_remove, len(current_blocks)))
            perturbed = [b for b in current_blocks if b not in to_remove]
            
            # Repair
            repaired = repair_greedy(perturbed)
            
            # Prune
            repaired = prune_solution(repaired)
            
            if len(repaired) < best_len:
                best_len = len(repaired)
                best_blocks = repaired.copy()
                current_blocks = repaired.copy()

        # Quick 1-swap local search on the best solution if time permits
        if time.time() < deadline - 10 and len(best_blocks) > 0:
            covered_mask = 0
            for idx in best_blocks:
                covered_mask |= block_bitmasks[idx]
            
            sample_indices = random.sample(best_blocks, min(10, len(best_blocks)))
            
            for i in sample_indices:
                b_mask = block_bitmasks[i]
                other_mask = covered_mask & ~b_mask
                unique_mask = b_mask & ~other_mask
                
                if unique_mask == 0:
                    continue
                
                unique_subsets = []
                m = unique_mask
                while m:
                    lsb = m & -m
                    idx_s = lsb.bit_length() - 1
                    unique_subsets.append(idx_s)
                    m ^= lsb
                
                if not unique_subsets:
                    continue
                
                candidates = set(blocks_containing_subset[unique_subsets[0]])
                for s in unique_subsets[1:]:
                    candidates.intersection_update(blocks_containing_subset[s])
                    if not candidates:
                        break
                
                candidates.discard(i)
                sol_set = set(best_blocks)
                candidates -= sol_set
                
                if candidates:
                    j = next(iter(candidates))
                    new_sol = [b if b != i else j for b in best_blocks]
                    
                    new_covered = 0
                    for idx in new_sol:
                        new_covered |= block_bitmasks[idx]
                    
                    if new_covered == covered_mask:
                        pruned_new_sol = prune_solution(new_sol)
                        
                        if len(pruned_new_sol) < best_len:
                            best_len = len(pruned_new_sol)
                            best_blocks = pruned_new_sol.copy()
                            current_blocks = pruned_new_sol.copy()

    # Final validation and formatting
    seen = set()
    final_blocks = []
    for idx in best_blocks:
        if idx not in seen:
            seen.add(idx)
            final_blocks.append(list(all_blocks[idx]))
    
    return final_blocks
