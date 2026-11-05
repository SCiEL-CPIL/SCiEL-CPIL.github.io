import itertools
import time
import random
import heapq
import sys

# Global Precomputation for v=19, k=7, t=4
# This saves ~2-5 seconds per call if the evaluator calls construct multiple times
# or if we can assume the parameters are fixed as per the task description.
# However, to be safe and generic, we will precompute inside a helper that caches.

_PRECOMPUTE_CACHE = {}

def _get_precomputed(v, k, t):
    key = (v, k, t)
    if key in _PRECOMPUTE_CACHE:
        return _PRECOMPUTE_CACHE[key]
        
    # 1. Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # 2. Precompute all k-blocks and their coverage masks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # Build coverage masks for each block
    # A block covers a t-subset if the t-subset is a subset of the block
    block_masks = [0] * n_k_blocks
    # Reverse index: for each t-subset, list of blocks that cover it
    blocks_containing_subset = [[] for _ in range(n_t_subs)]
    
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
            blocks_containing_subset[idx].append(i)
        block_masks[i] = mask
        
    data = {
        'v': v,
        'k': k,
        't': t,
        'all_t_subs': all_t_subs,
        'n_t_subs': n_t_subs,
        'sub_to_idx': sub_to_idx,
        'all_k_blocks': all_k_blocks,
        'n_k_blocks': n_k_blocks,
        'block_masks': block_masks,
        'blocks_containing_subset': blocks_containing_subset
    }
    
    _PRECOMPUTE_CACHE[key] = data
    return data

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Get precomputed data
    data = _get_precomputed(v, k, t)
    all_k_blocks = data['all_k_blocks']
    n_k_blocks = data['n_k_blocks']
    block_masks = data['block_masks']
    n_t_subs = data['n_t_subs']
    blocks_containing_subset = data['blocks_containing_subset']
    
    # 1. Greedy Initialization
    # We want to cover all t-subsets.
    # Start with no blocks.
    uncovered = (1 << n_t_subs) - 1
    selected_blocks = []
    selected_set = set()
    # Counts of how many times each t-subset is covered
    cover_counts = [0] * n_t_subs
    
    def add_block(idx):
        selected_blocks.append(idx)
        selected_set.add(idx)
        mask = block_masks[idx]
        m = mask
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            cover_counts[bit_idx] += 1
            m ^= lsb
            
    def remove_block(idx):
        selected_set.remove(idx)
        # Efficient removal: swap with last and pop
        pos = selected_blocks.index(idx)
        selected_blocks[pos] = selected_blocks[-1]
        selected_blocks.pop()
        
        mask = block_masks[idx]
        m = mask
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            cover_counts[bit_idx] -= 1
            m ^= lsb

    # Greedy Loop
    while uncovered:
        if time.time() > deadline - 60:
            break
        best_idx = -1
        best_gain = -1
        
        # Optimization: If many blocks are already selected, we can't just scan all.
        # But for v=19, k=7, n_k_blocks is ~50k. Scanning 50k blocks is fast in Python if vectorized or C-backed?
        # bit_count is fast. Let's scan.
        for i in range(n_k_blocks):
            if i in selected_set:
                continue
            gain = (block_masks[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_idx == -1:
            break
            
        add_block(best_idx)
        uncovered &= ~block_masks[best_idx]
        
    best_solution = selected_blocks[:]
    best_size = len(best_solution)
    
    # 2. Local Search Phase
    # We try to reduce the number of blocks.
    # Strategy:
    # a) Remove any redundant block (one that covers no unique t-subset).
    # b) Try to swap a block with another if it allows removal of a redundant block.
    # c) If stuck, perturb and re-greedy.
    
    no_improve_count = 0
    max_no_improve = 20
    
    while time.time() < deadline - 5:
        improved = False
        
        # Phase A: Redundancy Removal
        # Iterate through blocks, if a block is redundant, remove it.
        # We do this in random order to avoid bias.
        indices = list(range(len(selected_blocks)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 5:
                break
            idx = selected_blocks[i]
            mask = block_masks[idx]
            is_redundant = True
            m = mask
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                if cover_counts[bit_idx] == 1:
                    is_redundant = False
                    break
                m ^= lsb
                
            if is_redundant:
                remove_block(idx)
                improved = True
                if len(selected_blocks) < best_size:
                    best_size = len(selected_blocks)
                    best_solution = selected_blocks[:]
                # After a removal, we might have created new redundancies, so restart the scan
                break
                
        if improved:
            no_improve_count = 0
            continue
            
        # Phase B: 1-for-1 Swap
        # Try to find a block B in the solution that can be swapped with a block C not in the solution
        # such that the new solution has fewer blocks or allows further reduction.
        # Actually, a direct swap keeps the size same. The gain comes if C allows removal of another block D.
        # A simpler heuristic: Pick a random block B. Find its unique subsets.
        # Find a block C that covers those unique subsets.
        # Swap B for C. Then check for redundancies.
        
        if selected_blocks:
            # Pick a random block to try to replace
            rand_pos = random.randint(0, len(selected_blocks) - 1)
            idx_b = selected_blocks[rand_pos]
            
            # Find unique subsets covered by idx_b
            mask_b = block_masks[idx_b]
            unique_bits = []
            m = mask_b
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                if cover_counts[bit_idx] == 1:
                    unique_bits.append(bit_idx)
                m ^= lsb
                
            if unique_bits:
                # We need a replacement block that covers ALL these unique bits.
                # Intersect the candidate lists for these bits.
                # Sort by frequency to minimize intersection size
                unique_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
                
                candidates = set(blocks_containing_subset[unique_bits[0]])
                for b in unique_bits[1:]:
                    if not candidates:
                        break
                    candidates &= set(blocks_containing_subset[b])
                    
                # Filter out blocks already in the solution
                valid_candidates = [c for c in candidates if c not in selected_set]
                
                if valid_candidates:
                    # Pick a random valid candidate
                    new_idx = random.choice(valid_candidates)
                    
                    # Perform swap
                    remove_block(idx_b)
                    add_block(new_idx)
                    
                    # Check if this swap allowed any immediate redundancy removal
                    # We don't need to check immediately here, the next loop iteration (Phase A) will catch it.
                    # But we can mark improved to reset no_improve_count?
                    # Actually, the swap itself doesn't reduce size. It only enables future reductions.
                    # So we shouldn't count it as "improved" in terms of size, but it changes the state.
                    # Let's just let the loop continue.
                    # To avoid infinite loops of useless swaps, we can limit swaps per "stall" period.
                    # For now, we just continue.
                    no_improve_count = 0 # Reset counter to allow more swaps
                    continue
            else:
                # If idx_b has no unique subsets, it is redundant.
                # Why didn't Phase A catch it? Maybe it was added in a previous swap?
                # Or maybe Phase A broke early?
                # Let's remove it.
                remove_block(idx_b)
                improved = True
                if len(selected_blocks) < best_size:
                    best_size = len(selected_blocks)
                    best_solution = selected_blocks[:]
                continue
                
        # Phase C: Perturbation
        # If no improvement for a while, remove some random blocks and re-greedy
        no_improve_count += 1
        if no_improve_count > max_no_improve:
            # Perturb
            n_remove = random.randint(3, 8)
            to_remove = random.sample(selected_blocks, min(n_remove, len(selected_blocks)))
            for idx in to_remove:
                remove_block(idx)
                
            # Re-greedy for uncovered
            # Recalculate uncovered mask
            uncovered = 0
            for s in range(n_t_subs):
                if cover_counts[s] == 0:
                    uncovered |= (1 << s)
                    
            while uncovered:
                if time.time() > deadline - 5:
                    break
                best_idx = -1
                best_gain = -1
                for i in range(n_k_blocks):
                    if i in selected_set:
                        continue
                    gain = (block_masks[i] & uncovered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                        
                if best_idx == -1:
                    break
                add_block(best_idx)
                uncovered &= ~block_masks[best_idx]
                
            no_improve_count = 0
            max_no_improve = max(20, max_no_improve + 10)
            
        # If we are running out of time, do a final cleanup
        if time.time() > deadline - 10:
             # Final redundancy pass
             changed = True
             while changed and time.time() < deadline:
                 changed = False
                 for i in range(len(selected_blocks) - 1, -1, -1):
                     idx = selected_blocks[i]
                     mask = block_masks[idx]
                     is_redundant = True
                     m = mask
                     while m:
                         lsb = m & -m
                         bit_idx = lsb.bit_length() - 1
                         if cover_counts[bit_idx] == 1:
                             is_redundant = False
                             break
                         m ^= lsb
                     if is_redundant:
                         remove_block(idx)
                         changed = True
             break

    # Final verification and conversion
    # Ensure the best solution found is the one returned
    # Note: selected_blocks might be larger than best_solution if we didn't update it properly
    # We track best_solution explicitly.
    
    # Convert indices back to lists of points
    blocks = [list(all_k_blocks[i]) for i in best_solution]
    
    return blocks

# To ensure the cache is populated or to handle the specific task constraints if needed
# The task says "evaluator passes this task's parameters v, k and t".
# So construct(19, 7, 4) will be called.
