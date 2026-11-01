import itertools
import time

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a greedy set cover algorithm
    followed by local search with deletion, 1-for-1 swaps, and 2-for-1 swaps.
    """
    # 1. Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # 2. Precompute all k-blocks, their masks, and reverse index
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_blocks)
    block_masks = []
    blocks_containing = [[] for _ in range(num_t_subsets)]
    block_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
    
    for b_idx, block in enumerate(all_k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
            blocks_containing[idx].append(b_idx)
        block_masks.append(mask)
        
    # 3. Greedy Set Cover
    uncovered = (1 << num_t_subsets) - 1
    block_indices = []
    
    start_time = time.time()
    deadline = 140  # safety margin
    
    while uncovered:
        if time.time() - start_time > deadline:
            break
            
        first_bit = (uncovered & -uncovered).bit_length() - 1
        candidates = blocks_containing[first_bit]
        
        best_block_idx = -1
        best_gain = -1
        
        for b_idx in candidates:
            mask = block_masks[b_idx]
            overlap = mask & uncovered
            if overlap:
                gain = overlap.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                    if gain == uncovered.bit_count():
                        break
                        
        if best_block_idx == -1:
            break
            
        block_indices.append(best_block_idx)
        uncovered &= ~block_masks[best_block_idx]
    
    # 4. Local search
    def local_search(block_indices):
        """Perform local search to minimize block count."""
        if not block_indices:
            return block_indices
        
        # Deletion pass
        changed = True
        while changed and time.time() - start_time < deadline:
            changed = False
            n = len(block_indices)
            for i in range(n - 1, -1, -1):
                bi = block_indices[i]
                # Check if block bi is redundant
                other_mask = 0
                for j, bj in enumerate(block_indices):
                    if j != i:
                        other_mask |= block_masks[bj]
                if (block_masks[bi] & ~other_mask) == 0:
                    block_indices.pop(i)
                    changed = True
                    break
        
        # 1-for-1 swap pass
        changed = True
        while changed and time.time() - start_time < deadline:
            changed = False
            n = len(block_indices)
            for i in range(n):
                bi = block_indices[i]
                other_mask = 0
                for j, bj in enumerate(block_indices):
                    if j != i:
                        other_mask |= block_masks[bj]
                
                only_by_bi = block_masks[bi] & ~other_mask
                if only_by_bi == 0:
                    continue
                
                # Find best replacement
                best_replacement = -1
                best_gain = -1
                for bj in block_indices:
                    if bj == bi:
                        continue
                    gain = (block_masks[bj] & ~other_mask).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_replacement = bj
                
                if best_replacement != -1 and best_gain > 0:
                    new_mask = other_mask | block_masks[best_replacement]
                    if (block_masks[bi] & ~new_mask) == 0:
                        block_indices[i] = best_replacement
                        changed = True
                        break
        
        # 2-for-1 swap pass
        changed = True
        while changed and time.time() - start_time < deadline:
            changed = False
            n = len(block_indices)
            for i in range(n):
                if time.time() - start_time > deadline:
                    break
                for j in range(i + 1, n):
                    other_mask = 0
                    for m in range(n):
                        if m != i and m != j:
                            other_mask |= block_masks[block_indices[m]]
                    
                    only_by_ij = (block_masks[block_indices[i]] | block_masks[block_indices[j]]) & ~other_mask
                    if only_by_ij == 0:
                        continue
                    
                    # Find a single block that covers only_by_ij
                    for bk in range(num_blocks):
                        if (block_masks[bk] & only_by_ij) == only_by_ij:
                            if bk not in block_indices:
                                # Remove i and j, add bk
                                if i < j:
                                    block_indices.pop(j)
                                block_indices.pop(i)
                                block_indices.append(bk)
                                changed = True
                                break
                    if changed:
                        break
        
        return block_indices
    
    block_indices = local_search(block_indices)
    
    # Verify coverage
    final_union = 0
    for bi in block_indices:
        final_union |= block_masks[bi]
    
    if final_union != (1 << num_t_subsets) - 1:
        # Coverage broken, fall back to greedy result
        # Re-run greedy
        uncovered = (1 << num_t_subsets) - 1
        block_indices = []
        while uncovered:
            if time.time() - start_time > deadline:
                break
            first_bit = (uncovered & -uncovered).bit_length() - 1
            candidates = blocks_containing[first_bit]
            best_block_idx = -1
            best_gain = -1
            for b_idx in candidates:
                mask = block_masks[b_idx]
                overlap = mask & uncovered
                if overlap:
                    gain = overlap.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_block_idx = b_idx
                        if gain == uncovered.bit_count():
                            break
            if best_block_idx == -1:
                break
            block_indices.append(best_block_idx)
            uncovered &= ~block_masks[best_block_idx]
    
    return [list(all_k_blocks[bi]) for bi in block_indices]
