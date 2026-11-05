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

# Generate all 7-subsets and their covered 4-subset indices
all_blocks = []
block_subsets = [] # List of lists of 4-subset indices for each block
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]

for block in itertools.combinations(range(V), K):
    idxs = []
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        idxs.append(idx)
        blocks_containing_subset[idx].append(len(all_blocks))
    all_blocks.append(block)
    block_subsets.append(idxs)

N_BLOCKS = len(all_blocks)

# Precompute frequencies for weighting
subset_frequencies = [0] * N_SUBSETS
for block_id in range(N_BLOCKS):
    for sub_idx in block_subsets[block_id]:
        subset_frequencies[sub_idx] += 1

# Weight vectors
weights_inv_freq = np.array([1.0 / f for f in subset_frequencies])
weights_uniform = np.ones(N_SUBSETS)
weights_sqrt_inv_freq = np.array([1.0 / math.sqrt(f) for f in subset_frequencies])

# Coverage matrix for greedy (sparse or dense? Dense is 50388 x 3876 ~ 200M bytes, too big for int8? 50388*3876 = 195M bytes. Too big. Use sparse or list of lists)
# We'll stick to list of lists for coverage checks in local search, and use a fast greedy that doesn't require a full matrix multiplication if we can, or use a sparse representation if needed.
# Actually, for greedy, we can just iterate through blocks and calculate gain. 50k blocks * 35 subsets/block = 1.75M ops per step. 150 steps = 262M ops. This might be too slow in pure Python.
# We need a faster greedy.
# Let's use the "weighted greedy" where we pick the block that maximizes sum of weights of uncovered subsets.
# To speed this up, we can precompute the list of subsets for each block.
# And in the loop, we iterate through candidates.
# Optimization: Sort blocks by some heuristic? Or just random shuffle and take best?

def fast_greedy(weight_vector):
    uncovered = np.ones(N_SUBSETS, dtype=bool)
    chosen = []
    remaining = N_SUBSETS
    
    # Precompute gain for each block? No, it changes as uncovered changes.
    # We can maintain an array 'gains' and update it incrementally.
    # gains[b] = sum(weight[s] for s in block_subsets[b] if uncovered[s])
    
    gains = np.zeros(N_BLOCKS, dtype=np.float64)
    # Initial gain
    for b in range(N_BLOCKS):
        # Sum weights of subsets in this block
        # We can do this faster if we have a matrix, but let's try loop first.
        # To speed up, we can precompute the mask? No.
        pass
        
    # Alternative: Just do the standard greedy step by step.
    # Step 1: Calculate gains for all blocks.
    # This is O(N_BLOCKS * 35). 50k * 35 = 1.75M.
    # If we do this 100 times, 175M. Python is slow (~10-50M ops/sec). So 3-17 seconds. Acceptable if we have 112s budget.
    
    while remaining > 0:
        # Update gains
        # We only need to update gains for blocks that cover newly covered subsets?
        # Actually, when we pick a block, we mark its subsets as covered.
        # The gain for other blocks decreases by the weight of the subsets they share with the picked block.
        
        # Let's just recalculate gains for all blocks every step. It's simple and robust.
        # Optimization: Use NumPy for the sum?
        # weights * uncovered is a vector of length 3876.
        # block_subsets is a list of lists.
        
        # Better: Precompute a sparse matrix?
        # Let's stick to the loop but optimize the inner part.
        
        best_gain = -1
        best_idx = -1
        w_uncovered = weight_vector * uncovered # Vector
        
        for b in range(N_BLOCKS):
            # Sum of w_uncovered[s] for s in block_subsets[b]
            # Using sum() over a generator is slow.
            # If we had a dense matrix, we could do matrix-vector product.
            # Let's create a sparse matrix for coverage.
            pass
        
        # I will use a sparse matrix approach for the greedy to make it fast.
        # But I didn't import scipy in the previous thought. I will import it here.
        pass

    return chosen

# To ensure speed, I will use scipy.sparse for the coverage matrix in the greedy phase.
# And pure Python for local search.

import scipy.sparse as sp

# Build sparse coverage matrix: Rows = blocks, Cols = subsets
# 50388 x 3876
# COO format
rows = []
cols = []
data = []
for b, subs in enumerate(block_subsets):
    for s in subs:
        rows.append(b)
        cols.append(s)
        data.append(1)

cov_matrix = sp.coo_matrix((data, (rows, cols)), shape=(N_BLOCKS, N_SUBSETS)).tocsr()

def run_greedy(weight_vector):
    uncovered = np.ones(N_SUBSETS, dtype=bool)
    chosen = []
    remaining = N_SUBSETS
    w_uncovered = weight_vector * uncovered # Only non-zero where uncovered
    
    while remaining > 0:
        gains = cov_matrix @ w_uncovered
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain <= 0:
            break
            
        chosen.append(best_idx)
        
        # Update uncovered
        newly_covered = cov_matrix[best_idx].toarray().flatten()
        # newly_covered is 0/1 array. 1 means covered by this block.
        # We want to mark these as uncovered = False.
        # w_uncovered needs to be updated: set weight to 0 for these subsets.
        
        # Get indices of newly covered subsets
        # cov_matrix[best_idx] is sparse.
        nnz_indices = cov_matrix[best_idx].indices
        uncovered[nnz_indices] = False
        remaining -= len(nnz_indices)
        
        # Update w_uncovered
        w_uncovered[nnz_indices] = 0.0
        
    return chosen

def local_search(blocks, counts, deadline):
    # blocks: list of block indices
    # counts: array of length N_SUBSETS, count of coverage for each subset
    
    current = blocks.copy()
    cnt = counts.copy()
    cset = set(current)
    
    # Precompute unique subsets for each block? No, depends on cnt.
    
    def is_redundant(b_idx):
        for s in block_subsets[b_idx]:
            if cnt[s] == 1:
                return False
        return True
        
    def try_remove(b_idx):
        nonlocal current, cnt, cset
        if is_redundant(b_idx):
            for s in block_subsets[b_idx]:
                cnt[s] -= 1
            current.remove(b_idx)
            cset.discard(b_idx)
            return True
        return False

    def try_swap(b_idx, new_b_idx):
        # Remove b_idx, add new_b_idx
        # Check if all subsets with cnt==1 in b_idx are covered by new_b_idx
        nonlocal current, cnt, cset
        if b_idx == new_b_idx:
            return False
            
        unique_in_b = [s for s in block_subsets[b_idx] if cnt[s] == 1]
        
        # Check if new_b_idx covers all unique_in_b
        new_subs_set = set(block_subsets[new_b_idx])
        for s in unique_in_b:
            if s not in new_subs_set:
                return False
        
        # Also, ensure we don't introduce "uncovered" sets? 
        # No, we are adding a block, so coverage only increases or stays same.
        # The only risk is if we remove a block that was the ONLY cover for some set.
        # We checked that.
        
        # Perform swap
        for s in block_subsets[b_idx]:
            cnt[s] -= 1
        current.remove(b_idx)
        cset.discard(b_idx)
        
        for s in block_subsets[new_b_idx]:
            cnt[s] += 1
        current.append(new_b_idx)
        cset.add(new_b_idx)
        
        return True

    no_improve = 0
    
    while time.time() < deadline:
        # Phase 1: Remove redundant blocks
        progress = True
        while progress:
            progress = False
            for b_idx in current[:]: # Copy to allow deletion
                if is_redundant(b_idx):
                    for s in block_subsets[b_idx]:
                        cnt[s] -= 1
                    current.remove(b_idx)
                    cset.discard(b_idx)
                    progress = True
                    no_improve = 0
        
        # Phase 2: Swaps
        # Sample some blocks to try swapping
        if len(current) == 0:
            break
            
        n_samples = min(len(current), 50)
        samples = random.sample(current, n_samples)
        
        swap_done = False
        for b_idx in samples:
            # Find candidates for new block
            # Ideally, we want a block that covers the unique subsets of b_idx
            unique_in_b = [s for s in block_subsets[b_idx] if cnt[s] == 1]
            if not unique_in_b:
                # It's redundant, should have been caught in Phase 1? 
                # Maybe counts changed?
                # Just remove it
                for s in block_subsets[b_idx]:
                    cnt[s] -= 1
                current.remove(b_idx)
                cset.discard(b_idx)
                no_improve = 0
                swap_done = True
                break
                
            # Intersect blocks containing all unique subsets
            # Start with blocks containing the first unique subset
            cands = set(blocks_containing_subset[unique_in_b[0]])
            for s in unique_in_b[1:]:
                cands.intersection_update(blocks_containing_subset[s])
                if not cands:
                    break
            
            # Remove self from candidates
            cands.discard(b_idx)
            
            if cands:
                # Pick a random candidate or the best one?
                # Let's try random to explore
                new_b_idx = random.choice(list(cands))
                if try_swap(b_idx, new_b_idx):
                    no_improve = 0
                    swap_done = True
                    break
        
        if not swap_done:
            no_improve += 1
            if no_improve > 50:
                # Stuck, try 2-for-1 or break
                # Simple 2-for-1: Pick two redundant-ish blocks, see if one block covers their uniques
                # This is expensive. Let's just break or add a random block?
                # Adding a random block increases size, not good.
                # Let's try to find a 2-for-1
                if len(current) > 2:
                    i1, i2 = random.sample(current, 2)
                    u1 = [s for s in block_subsets[i1] if cnt[s] == 1]
                    u2 = [s for s in block_subsets[i2] if cnt[s] == 1]
                    all_uniques = set(u1) | set(u2)
                    
                    if all_uniques:
                        cands = set(blocks_containing_subset[list(all_uniques)[0]])
                        for s in list(all_uniques)[1:]:
                            cands.intersection_update(blocks_containing_subset[s])
                            if not cands:
                                break
                        
                        cands.discard(i1)
                        cands.discard(i2)
                        
                        if cands:
                            new_b_idx = random.choice(list(cands))
                            # Check if valid 2-for-1
                            # Remove i1, i2, add new_b_idx
                            # Must cover all uniques of i1 and i2.
                            # We constructed cands to cover all_uniques.
                            # So it is valid.
                            
                            for s in block_subsets[i1]: cnt[s] -= 1
                            for s in block_subsets[i2]: cnt[s] -= 1
                            current.remove(i1)
                            current.remove(i2)
                            cset.discard(i1)
                            cset.discard(i2)
                            
                            for s in block_subsets[new_b_idx]: cnt[s] += 1
                            current.append(new_b_idx)
                            cset.add(new_b_idx)
                            
                            no_improve = 0
                            swap_done = True

    return current, cnt

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback
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
    deadline = start_time + 115 # Safety margin

    best_blocks = None
    best_len = float('inf')
    
    # Different weightings for greedy
    weightings = [
        weights_inv_freq,
        weights_uniform,
        weights_sqrt_inv_freq,
        np.array([1.0 / math.log(f + 1) for f in subset_frequencies])
    ]

    for w in weightings:
        if time.time() > deadline - 20:
            break
            
        greedy_result = run_greedy(w)
        
        if not greedy_result:
            continue
            
        # Calculate counts
        counts = np.zeros(N_SUBSETS, dtype=np.int32)
        for b_idx in greedy_result:
            for s in block_subsets[b_idx]:
                counts[s] += 1
        
        # Local Search
        ls_blocks, ls_counts = local_search(greedy_result, counts, deadline)
        
        # Verify
        # Check if all subsets are covered
        if np.any(ls_counts == 0):
            # Add missing blocks (should not happen if local search is correct, but safety)
            missing = np.where(ls_counts == 0)[0]
            for m in missing:
                # Find any block containing m
                # Use first one in blocks_containing_subset[m]
                if blocks_containing_subset[m]:
                    b_idx = blocks_containing_subset[m][0]
                    ls_blocks.append(b_idx)
                    for s in block_subsets[b_idx]:
                        ls_counts[s] += 1
        
        # Remove duplicates just in case
        ls_blocks = list(set(ls_blocks))
        
        # Recalculate counts for final check and length
        final_counts = np.zeros(N_SUBSETS, dtype=np.int32)
        for b_idx in ls_blocks:
            for s in block_subsets[b_idx]:
                final_counts[s] += 1
                
        if np.all(final_counts > 0):
            if len(ls_blocks) < best_len:
                best_len = len(ls_blocks)
                best_blocks = ls_blocks.copy()

    if best_blocks is None:
        # Fallback to simple greedy if everything failed
        greedy_result = run_greedy(weights_inv_freq)
        best_blocks = greedy_result

    # Convert indices to point lists
    final_output = [list(all_blocks[idx]) for idx in best_blocks]
    
    return final_output
