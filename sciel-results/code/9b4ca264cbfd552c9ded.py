import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    # Precompute t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    # Precompute k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # Build bitmasks for each block as Python ints
    block_bitmasks = [0] * n_k_blocks
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
        block_bitmasks[i] = mask
    
    full_mask = (1 << n_t_subsets) - 1
    
    # Precompute list of set bits for each mask for faster iteration
    mask_bits = []
    for mask in block_bitmasks:
        bits = []
        m = mask
        while m:
            lsb = m & -m
            bits.append(lsb.bit_length() - 1)
            m ^= lsb
        mask_bits.append(bits)
    
    def compute_gain_vector(uncovered_mask, candidate_indices):
        """Compute gains for candidate blocks against uncovered mask."""
        gains = np.zeros(len(candidate_indices), dtype=np.int32)
        for i, idx in enumerate(candidate_indices):
            gains[i] = (uncovered_mask & block_bitmasks[idx]).bit_count()
        return gains
    
    def greedy_cover(uncovered_mask=None, top_k=5):
        """Greedy set cover with top-k candidate selection."""
        if uncovered_mask is None:
            uncovered_mask = full_mask
        
        selected_blocks = []
        coverage_counts = np.zeros(n_t_subsets, dtype=np.int32)
        all_indices = list(range(n_k_blocks))
        
        while uncovered_mask:
            if time.time() > deadline:
                break
            
            # Compute gains for all blocks
            gains = np.array([(uncovered_mask & block_bitmasks[i]).bit_count() for i in all_indices], dtype=np.int32)
            
            # Get top-k candidates
            top_indices = np.argsort(gains)[-top_k:][::-1]
            best_idx = -1
            best_gain = 0
            
            for idx in top_indices:
                gain = gains[idx]
                if gain > best_gain:
                    best_gain = gain
                    best_idx = int(idx)
            
            if best_idx == -1 or best_gain == 0:
                break
            
            selected_blocks.append(best_idx)
            
            # Update coverage counts and uncovered mask
            for bit in mask_bits[best_idx]:
                coverage_counts[bit] += 1
            uncovered_mask &= ~block_bitmasks[best_idx]
        
        return selected_blocks, coverage_counts
    
    def local_search_remove(blocks_indices, coverage_counts):
        """Remove redundant blocks iteratively."""
        if time.time() > deadline:
            return blocks_indices, coverage_counts
        
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            to_remove = []
            
            for b_idx in blocks_indices:
                essential = False
                for bit in mask_bits[b_idx]:
                    if coverage_counts[bit] == 1:
                        essential = True
                        break
                if not essential:
                    to_remove.append(b_idx)
            
            if to_remove:
                remove_set = set(to_remove)
                blocks_indices = [b for b in blocks_indices if b not in remove_set]
                for b_idx in to_remove:
                    for bit in mask_bits[b_idx]:
                        coverage_counts[bit] -= 1
                changed = True
        
        return blocks_indices, coverage_counts
    
    def local_search_swap(blocks_indices, coverage_counts):
        """Try to swap blocks to reduce count."""
        if time.time() > deadline:
            return blocks_indices, coverage_counts
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            for i, b_idx in enumerate(blocks_indices):
                if time.time() > deadline:
                    break
                
                # Temporarily remove b_idx
                for bit in mask_bits[b_idx]:
                    coverage_counts[bit] -= 1
                
                # Find uncovered t-subsets
                uncovered_mask = 0
                for j in range(n_t_subsets):
                    if coverage_counts[j] == 0:
                        uncovered_mask |= (1 << j)
                
                if uncovered_mask == 0:
                    # Can remove b_idx entirely
                    blocks_indices = [b for j, b in enumerate(blocks_indices) if j != i]
                    improved = True
                    break
                
                # Try to find a block that covers the most uncovered
                best_new_idx = -1
                best_new_gain = 0
                for new_idx in range(n_k_blocks):
                    if new_idx == b_idx or new_idx in blocks_indices:
                        continue
                    gain = (uncovered_mask & block_bitmasks[new_idx]).bit_count()
                    if gain > best_new_gain:
                        best_new_gain = gain
                        best_new_idx = new_idx
                
                # Restore b_idx
                for bit in mask_bits[b_idx]:
                    coverage_counts[bit] += 1
                
                # If we can cover all uncovered with one new block, do the swap
                if best_new_idx != -1 and best_new_gain >= uncovered_mask.bit_count():
                    # Remove old block
                    for bit in mask_bits[b_idx]:
                        coverage_counts[bit] -= 1
                    
                    blocks_indices[i] = best_new_idx
                    
                    # Add new block
                    for bit in mask_bits[best_new_idx]:
                        coverage_counts[bit] += 1
                    
                    improved = True
                    break
        
        return blocks_indices, coverage_counts
    
    def local_search_2exchange(blocks_indices, coverage_counts):
        """Try 2-exchange: remove 2 blocks, add 1 block that covers all their unique coverage."""
        if time.time() > deadline:
            return blocks_indices, coverage_counts
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            n = len(blocks_indices)
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i+1, n):
                    if time.time() > deadline:
                        break
                    
                    # Temporarily remove blocks i and j
                    mask_i = block_bitmasks[blocks_indices[i]]
                    mask_j = block_bitmasks[blocks_indices[j]]
                    
                    for bit in mask_bits[blocks_indices[i]]:
                        coverage_counts[bit] -= 1
                    for bit in mask_bits[blocks_indices[j]]:
                        coverage_counts[bit] -= 1
                    
                    # Find uncovered
                    uncovered_mask = 0
                    for bit in range(n_t_subsets):
                        if coverage_counts[bit] == 0:
                            uncovered_mask |= (1 << bit)
                    
                    # Restore
                    for bit in mask_bits[blocks_indices[i]]:
                        coverage_counts[bit] += 1
                    for bit in mask_bits[blocks_indices[j]]:
                        coverage_counts[bit] += 1
                    
                    if uncovered_mask == 0:
                        # Can remove both
                        blocks_indices = [b for idx, b in enumerate(blocks_indices) if idx != i and idx != j]
                        improved = True
                        break
                    
                    # Try to find one block that covers all uncovered
                    best_new_idx = -1
                    best_new_gain = 0
                    for new_idx in range(n_k_blocks):
                        if new_idx in blocks_indices:
                            continue
                        gain = (uncovered_mask & block_bitmasks[new_idx]).bit_count()
                        if gain > best_new_gain:
                            best_new_gain = gain
                            best_new_idx = new_idx
                    
                    if best_new_idx != -1 and best_new_gain >= uncovered_mask.bit_count():
                        # Remove both, add one
                        for bit in mask_bits[blocks_indices[i]]:
                            coverage_counts[bit] -= 1
                        for bit in mask_bits[blocks_indices[j]]:
                            coverage_counts[bit] -= 1
                        
                        blocks_indices[i] = best_new_idx
                        blocks_indices.pop(j)
                        
                        for bit in mask_bits[best_new_idx]:
                            coverage_counts[bit] += 1
                        
                        improved = True
                        break
                
                if improved:
                    break
        
        return blocks_indices, coverage_counts
    
    def compute_coverage_mask(blocks_list):
        """Compute the coverage mask for a list of block indices."""
        mask = 0
        for b_idx in blocks_list:
            mask |= block_bitmasks[b_idx]
        return mask
    
    def cover_holes_greedy(existing_blocks, holes_mask):
        """Cover holes using greedy, returning new blocks added."""
        new_blocks = []
        uncovered = holes_mask
        while uncovered:
            if time.time() > deadline:
                break
            best_idx = -1
            best_gain = 0
            for i in range(n_k_blocks):
                gain = (uncovered & block_bitmasks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            new_blocks.append(best_idx)
            uncovered &= ~block_bitmasks[best_idx]
        return new_blocks
    
    def rebuild_counts(blocks_list):
        """Rebuild coverage counts from scratch."""
        counts = np.zeros(n_t_subsets, dtype=np.int32)
        for b_idx in blocks_list:
            for bit in mask_bits[b_idx]:
                counts[bit] += 1
        return counts
    
    # Initial greedy
    initial_blocks, initial_counts = greedy_cover(top_k=5)
    
    # Local search: remove, swap, 2-exchange
    current_blocks, current_counts = local_search_remove(initial_blocks, initial_counts)
    current_blocks, current_counts = local_search_swap(current_blocks, current_counts)
    current_blocks, current_counts = local_search_2exchange(current_blocks, current_counts)
    
    best_blocks = list(current_blocks)
    best_counts = current_counts.copy()
    best_count = len(current_blocks)
    
    # Perturbation-based local search with more aggressive perturbation
    attempts = 0
    while time.time() < deadline - 15:
        attempts += 1
        
        # Create perturbation: remove random number of blocks
        num_remove = min(len(best_blocks), random.randint(2, max(2, len(best_blocks) // 5)))
        if num_remove == 0:
            break
        
        to_remove = random.sample(best_blocks, num_remove)
        remove_set = set(to_remove)
        perturbed_blocks = [b for b in best_blocks if b not in remove_set]
        
        # Compute holes
        covered_mask = compute_coverage_mask(perturbed_blocks)
        holes_mask = full_mask & ~covered_mask
        
        # Cover holes
        added_blocks = cover_holes_greedy(perturbed_blocks, holes_mask)
        
        new_total_blocks = perturbed_blocks + added_blocks
        new_count = len(new_total_blocks)
        
        if new_count <= best_count:
            # Rebuild counts and run local search
            new_counts = rebuild_counts(new_total_blocks)
            cleaned_blocks, cleaned_counts = local_search_remove(new_total_blocks, new_counts)
            cleaned_blocks, cleaned_counts = local_search_swap(cleaned_blocks, cleaned_counts)
            cleaned_blocks, cleaned_counts = local_search_2exchange(cleaned_blocks, cleaned_counts)
            
            if len(cleaned_blocks) < best_count:
                best_blocks = cleaned_blocks
                best_counts = cleaned_counts
                best_count = len(best_blocks)
            elif len(new_total_blocks) < best_count:
                best_blocks = new_total_blocks
                best_counts = new_counts
                best_count = len(best_blocks)
    
    # Final local search pass
    if time.time() < deadline - 5:
        best_blocks, best_counts = local_search_remove(best_blocks, best_counts)
        best_blocks, best_counts = local_search_swap(best_blocks, best_counts)
        best_blocks, best_counts = local_search_2exchange(best_blocks, best_counts)
    
    # Convert to output format
    final_blocks = [list(all_k_blocks[i]) for i in best_blocks]
    
    return final_blocks
