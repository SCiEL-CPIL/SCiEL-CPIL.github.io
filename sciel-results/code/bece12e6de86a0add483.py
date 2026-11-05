import itertools
import random
import time
import math
import numpy as np
import scipy.sparse as sp
import scipy.optimize as opt

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
# coverage_matrix[i, j] = 1 if block i covers subset j
coverage_matrix = np.zeros((N_BLOCKS, N_SUBSETS), dtype=np.int8)
for i, bits in enumerate(block_bit_indices):
    for b in bits:
        coverage_matrix[i, b] = 1

# weights as numpy array
weights_np = np.array(weights, dtype=np.float64)

# Precompute alternative weightings for diverse greedy runs
weights_uniform = np.ones(N_SUBSETS, dtype=np.float64)
weights_sqrt_inv = np.array([1.0 / math.sqrt(f) for f in subset_frequencies], dtype=np.float64)
weights_inv_sqrt = np.array([1.0 / math.sqrt(f) for f in subset_frequencies], dtype=np.float64)

# Precompute block coverage counts for faster redundancy checks
block_covered_count = np.zeros(N_BLOCKS, dtype=np.int32)
for i, bits in enumerate(block_bit_indices):
    block_covered_count[i] = len(bits)

# Precompute sparse coverage matrix for LASSO-style initialization
cov_sparse = sp.csr_matrix(coverage_matrix.T)  # N_SUBSETS x N_BLOCKS


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
    deadline = start_time + 112  # Safety margin below 120s kill

    def fast_greedy(weight_vector=None):
        """Vectorized weighted greedy using numpy for speed."""
        if weight_vector is None:
            weight_vector = weights_np
        uncovered = np.ones(N_SUBSETS, dtype=np.int8)
        chosen = []
        remaining = N_SUBSETS
        
        while remaining > 0:
            # Compute gain for each block: sum of weights of uncovered subsets it covers
            gains = coverage_matrix @ (weight_vector * uncovered)
            best_idx = int(np.argmax(gains))
            best_gain = gains[best_idx]
            
            if best_gain <= 0:
                break
            
            chosen.append(best_idx)
            
            # Update uncovered
            newly_covered = coverage_matrix[best_idx]
            still_uncovered = uncovered & (1 - newly_covered)
            remaining = int(np.sum(still_uncovered))
            uncovered = still_uncovered
        
        return chosen

    # 1. Try multiple greedy initializations with different weightings
    greedy_results = []
    for w_vec in [weights_np, weights_uniform, weights_sqrt_inv]:
        if time.time() > deadline - 5:
            break
        result = fast_greedy(w_vec)
        if result:
            greedy_results.append(result)
    
    # Pick the best greedy result
    if greedy_results:
        best_greedy = min(greedy_results, key=len)
    else:
        best_greedy = fast_greedy(weights_np)
    
    # 1b. LASSO-inspired sparse initialization
    if time.time() < deadline - 20:
        try:
            lassO_result = lassO_init(deadline)
            if lassO_result and len(lassO_result) < best_greedy:
                best_greedy = min(best_greedy, len(lassO_result))
                if len(lassO_result) < len(best_greedy):
                    best_greedy = lassO_result
            if lassO_result:
                greedy_results.append(lassO_result)
        except Exception:
            pass
    
    if greedy_results:
        best_greedy = min(greedy_results, key=len)
    
    current_blocks = best_greedy
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()

    # Build counts (list of ints, fast)
    counts = [0] * N_SUBSETS
    for idx in current_blocks:
        for bit in block_bit_indices[idx]:
            counts[bit] += 1

    def lassO_init(deadline):
        """Sparse initialization via projected gradient with L1 penalty."""
        # Start with a random subset of blocks as initial guess
        n_init = 120
        init_indices = random.sample(range(N_BLOCKS), n_init)
        x = np.zeros(N_BLOCKS, dtype=np.float64)
        for i in init_indices:
            x[i] = 1.0
        
        # Projected gradient descent with L1 penalty
        lr = 0.05
        l1_lambda = 0.01
        l2_lambda = 10.0  # Large L2 to push toward binary
        
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            
            # Residual: uncovered subsets (where sum of x * cov < 1)
            cov_sum = cov_sparse @ x
            residual = np.maximum(1.0 - cov_sum, 0)
            
            # Gradient of loss: 2 * cov^T @ residual + l2 * 2 * x
            grad = (2.0 * (cov_sparse.T @ residual)) + (2.0 * l2_lambda * x)
            
            # Soft-thresholding for L1
            x_new = x - lr * grad
            # Soft-threshold
            x_new = np.sign(x_new) * np.maximum(np.abs(x_new) - lr * l1_lambda, 0)
            # Project to [0, 1]
            x_new = np.clip(x_new, 0, 1)
            
            # Check convergence
            if np.max(np.abs(x_new - x)) < 1e-4:
                break
            x = x_new
        
        # Extract blocks with high coefficient
        threshold = 0.3
        selected = [i for i in range(N_BLOCKS) if x[i] > threshold]
        
        if not selected:
            return None
        
        # Verify coverage and add missing blocks via greedy
        cov_count = np.zeros(N_SUBSETS, dtype=np.int32)
        for idx in selected:
            for bit in block_bit_indices[idx]:
                cov_count[bit] += 1
        
        uncovered = np.where(cov_count == 0)[0]
        if len(uncovered) > 0:
            # Add blocks to cover remaining subsets
            uncovered_set = set(uncovered)
            for idx in range(N_BLOCKS):
                if idx in set(selected):
                    continue
                for bit in block_bit_indices[idx]:
                    if bit in uncovered_set:
                        selected.append(idx)
                        for b2 in block_bit_indices[idx]:
                            cov_count[b2] += 1
                        uncovered_set.discard(bit)
                        break
                if not uncovered_set:
                    break
        
        return selected

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

    # 2. Local search: repeatedly remove redundant blocks, then do swaps
    no_improve = 0
    max_no_improve = 40
    current_set = set(current_blocks)

    def rebuild_set():
        nonlocal current_set
        current_set = set(current_blocks)

    def do_local_search(blocks, counts, deadline):
        """Perform local search on the given blocks and counts."""
        nonlocal best_len, best_blocks
        current = blocks.copy()
        cnt = counts.copy()
        cset = set(current)
        n_imp = 0
        
        while time.time() < deadline:
            # Phase A: prune all redundant blocks
            progress = True
            while progress:
                progress = False
                for pos in range(len(current)):
                    idx = current[pos]
                    redundant = True
                    for bit in block_bit_indices[idx]:
                        if cnt[bit] == 1:
                            redundant = False
                            break
                    if redundant:
                        for bit in block_bit_indices[idx]:
                            cnt[bit] -= 1
                        current.pop(pos)
                        cset.discard(idx)
                        progress = True
                        if len(current) < best_len:
                            best_len = len(current)
                            best_blocks = current.copy()
                        n_imp = 0
                        break

            if len(current) == 0:
                break

            # Phase B: try 1-for-1 swap or 2-for-1 swap
            # First try 2-for-1: remove two blocks, add one that covers their unique subsets
            two_for_one_done = False
            if len(current) > 2 and random.random() < 0.3:
                # Sample pairs
                n_pairs = min(len(current) * 2, 50)
                for _ in range(n_pairs):
                    i1 = random.randint(0, len(current) - 1)
                    i2 = random.randint(0, len(current) - 1)
                    if i1 == i2:
                        continue
                    idx1 = current[i1]
                    idx2 = current[i2]
                    # Find unique subsets for each
                    unique1 = [bit for bit in block_bit_indices[idx1] if cnt[bit] == 1]
                    unique2 = [bit for bit in block_bit_indices[idx2] if cnt[bit] == 1]
                    if not unique1 and not unique2:
                        # Both redundant, remove both
                        for bit in block_bit_indices[idx1]:
                            cnt[bit] -= 1
                        for bit in block_bit_indices[idx2]:
                            cnt[bit] -= 1
                        # Remove in reverse order
                        pos1 = current.index(idx1)
                        pos2 = current.index(idx2)
                        if pos1 > pos2:
                            pos1, pos2 = pos2, pos1
                        current.pop(pos2)
                        current.pop(pos1)
                        cset.discard(idx1)
                        cset.discard(idx2)
                        if len(current) < best_len:
                            best_len = len(current)
                            best_blocks = current.copy()
                        n_imp = 0
                        two_for_one_done = True
                        break
                    elif unique1 and unique2:
                        # Find a block that covers all unique subsets
                        all_unique = set(unique1) | set(unique2)
                        if len(all_unique) <= 6:
                            cands = set(blocks_containing_subset[all_unique.pop()])
                            for b in all_unique:
                                cands.intersection_update(blocks_containing_subset[b])
                                if not cands:
                                    break
                            if cands:
                                cands.discard(idx1)
                                cands.discard(idx2)
                                if cands:
                                    j = random.choice(list(cands))
                                    # Check if j adds no new uncovered issues
                                    for bit in block_bit_indices[j]:
                                        if cnt[bit] == 0:
                                            break
                                    else:
                                        # Safe swap
                                        for bit in block_bit_indices[idx1]:
                                            cnt[bit] -= 1
                                        for bit in block_bit_indices[idx2]:
                                            cnt[bit] -= 1
                                        pos1 = current.index(idx1)
                                        pos2 = current.index(idx2)
                                        if pos1 > pos2:
                                            pos1, pos2 = pos2, pos1
                                        current.pop(pos2)
                                        current.pop(pos1)
                                        cset.discard(idx1)
                                        cset.discard(idx2)
                                        
                                        for bit in block_bit_indices[j]:
                                            cnt[bit] += 1
                                        current.append(j)
                                        cset.add(j)
                                        
                                        if len(current) < best_len:
                                            best_len = len(current)
                                            best_blocks = current.copy()
                                        n_imp = 0
                                        two_for_one_done = True
                                        break
            if two_for_one_done:
                continue

            # Fall back to 1-for-1 swap
            best_swap_candidate = -1
            min_unique = float('inf')
            
            n_sample = min(len(current), 30)
            sampled_indices = random.sample(range(len(current)), n_sample)
            
            for i in sampled_indices:
                idx_i = current[i]
                n_unique = 0
                for bit in block_bit_indices[idx_i]:
                    if cnt[bit] == 1:
                        n_unique += 1
                if 0 < n_unique < min_unique:
                    min_unique = n_unique
                    best_swap_candidate = i
            
            if best_swap_candidate < 0:
                i = random.randint(0, len(current) - 1)
                idx_i = current[i]
                uncovered_bits = [bit for bit in block_bit_indices[idx_i] if cnt[bit] == 1]
                if not uncovered_bits:
                    for bit in block_bit_indices[idx_i]:
                        cnt[bit] -= 1
                    current.pop(i)
                    cset.discard(idx_i)
                    if len(current) < best_len:
                        best_len = len(current)
                        best_blocks = current.copy()
                    n_imp = 0
                    continue
                best_swap_candidate = i
                idx_i = current[i]
                uncovered_bits = [bit for bit in block_bit_indices[idx_i] if cnt[bit] == 1]
            else:
                i = best_swap_candidate
                idx_i = current[i]
                uncovered_bits = [bit for bit in block_bit_indices[idx_i] if cnt[bit] == 1]

            if not uncovered_bits:
                for bit in block_bit_indices[idx_i]:
                    cnt[bit] -= 1
                current.pop(i)
                cset.discard(idx_i)
                if len(current) < best_len:
                    best_len = len(current)
                    best_blocks = current.copy()
                n_imp = 0
                continue

            if len(uncovered_bits) > 5:
                n_imp += 1
                if n_imp > max_no_improve:
                    under_covered = [b for b in range(N_SUBSETS) if cnt[b] == 1]
                    if under_covered:
                        target = random.choice(under_covered)
                        cands_for_target = blocks_containing_subset[target]
                        c = random.choice(cands_for_target)
                        if c not in cset:
                            for bit in block_bit_indices[c]:
                                cnt[bit] += 1
                            current.append(c)
                            cset.add(c)
                    n_imp = 0
                continue

            uncovered_bits_sorted = sorted(uncovered_bits, key=lambda b: len(blocks_containing_subset[b]))
            cands = set(blocks_containing_subset[uncovered_bits_sorted[0]])
            for b in uncovered_bits_sorted[1:]:
                cands.intersection_update(blocks_containing_subset[b])
                if not cands:
                    break

            if cands:
                cands.discard(idx_i)
                if cands:
                    cands_list = list(cands)
                    if len(cands_list) <= 10:
                        best_j = cands_list[0]
                        best_extra = -1
                        for j in cands_list:
                            extra = 0
                            for bit in block_bit_indices[j]:
                                if cnt[bit] == 0:
                                    extra += 1
                            if extra > best_extra:
                                best_extra = extra
                                best_j = j
                        j = best_j
                    else:
                        j = random.choice(cands_list)
                    
                    # Remove i, add j
                    for bit in block_bit_indices[idx_i]:
                        cnt[bit] -= 1
                    current.pop(i)
                    cset.discard(idx_i)
                    
                    for bit in block_bit_indices[j]:
                        cnt[bit] += 1
                    current.append(j)
                    cset.add(j)
                    
                    n_imp = 0
                    continue

            n_imp += 1
            if n_imp > max_no_improve:
                under_covered = [b for b in range(N_SUBSETS) if cnt[b] == 1]
                if under_covered:
                    for _ in range(random.randint(2, 3)):
                        target = random.choice(under_covered)
                        cands_for_target = blocks_containing_subset[target]
                        c = random.choice(cands_for_target)
                        if c not in cset:
                            for bit in block_bit_indices[c]:
                                cnt[bit] += 1
                            current.append(c)
                            cset.add(c)
                else:
                    for _ in range(random.randint(1, 2)):
                        c = random.randint(0, N_BLOCKS - 1)
                        if c not in cset:
                            for bit in block_bit_indices[c]:
                                cnt[bit] += 1
                            current.append(c)
                            cset.add(c)
                n_imp = 0

        return current, cnt

    # Run local search
    current_blocks, counts = do_local_search(current_blocks, counts, deadline)

    # 3. If time remains, try more greedy restarts with different weightings
    if time.time() < deadline - 10:
        for w_vec in [weights_uniform, weights_sqrt_inv]:
            if time.time() > deadline - 5:
                break
            alt = fast_greedy(w_vec)
            if len(alt) < best_len:
                best_len = len(alt)
                best_blocks = alt.copy()
                # Quick prune pass
                counts2 = [0] * N_SUBSETS
                for idx in best_blocks:
                    for bit in block_bit_indices[idx]:
                        counts2[bit] += 1
                changed = True
                while changed and time.time() < deadline - 3:
                    changed = False
                    for pos in range(len(best_blocks)):
                        idx = best_blocks[pos]
                        redundant = True
                        for bit in block_bit_indices[idx]:
                            if counts2[bit] == 1:
                                redundant = False
                                break
                        if redundant:
                            for bit in block_bit_indices[idx]:
                                counts2[bit] -= 1
                            best_blocks.pop(pos)
                            changed = True
                            if len(best_blocks) < best_len:
                                best_len = len(best_blocks)
                            break

    # Final validation: ensure all blocks are distinct and valid
    seen = set()
    final_blocks = []
    for idx in best_blocks:
        if idx not in seen:
            seen.add(idx)
            final_blocks.append(list(all_blocks[idx]))
    
    return final_blocks
