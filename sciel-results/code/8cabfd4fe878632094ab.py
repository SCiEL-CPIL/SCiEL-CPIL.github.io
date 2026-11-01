import itertools
import random
import time

def construct(v, k, t):
    # Precompute all t-subsets and map to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    
    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k_subs = len(all_k_subs)
    
    # Map t-subset to index
    t_sub_to_idx = {tsub: i for i, tsub in enumerate(all_t_subs)}
    
    # Precompute block_masks (Python ints)
    block_masks = [0] * n_k_subs
    # Also build t_to_blocks: list of lists, t_to_blocks[t_idx] = list of block indices covering t_idx
    t_to_blocks = [[] for _ in range(n_t_subs)]
    
    for i, ksub in enumerate(all_k_subs):
        mask = 0
        for tsub in itertools.combinations(ksub, t):
            idx = t_sub_to_idx[tsub]
            mask |= (1 << idx)
            t_to_blocks[idx].append(i)
        block_masks[i] = mask
        
    # Greedy Set Cover
    uncovered = (1 << n_t_subs) - 1
    selected = []
    selected_set = set()
    freq = [0] * n_t_subs # Frequency of coverage for each t-subset
    
    # Optimization: Keep a list of available blocks? 
    # Greedy usually picks the best one each time.
    # We can just iterate all blocks each time.
    
    while uncovered:
        best_idx = -1
        best_gain = 0
        for i in range(n_k_subs):
            if i in selected_set:
                continue
            gain = (block_masks[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1:
            break
        selected.append(best_idx)
        selected_set.add(best_idx)
        uncovered &= ~block_masks[best_idx]
        # Update freq
        m = block_masks[best_idx]
        # Iterate set bits in m to update freq
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            freq[idx] += 1
            m ^= lsb
            
    current_blocks = selected[:]
    current_size = len(current_blocks)
    
    # Local Search
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    def get_unique_t_idx(block_idx):
        """Return list of t-subset indices covered ONLY by this block (freq==1)"""
        m = block_masks[block_idx]
        uniques = []
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if freq[idx] == 1:
                uniques.append(idx)
            m ^= lsb
        return uniques

    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        
        # Phase 1: Remove redundant blocks
        # A block is redundant if it has no unique t-subsets (all its t-subsets are covered by others)
        # i.e., get_unique_t_idx(block) is empty.
        
        removed_any = False
        for i in range(len(current_blocks)):
            if time.time() - start_time > time_limit:
                break
            b_idx = current_blocks[i]
            uniques = get_unique_t_idx(b_idx)
            if not uniques:
                # Redundant, remove it
                # Update freq
                m = block_masks[b_idx]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    freq[idx] -= 1
                    m ^= lsb
                current_blocks.pop(i)
                removed_any = True
                improved = True
                break # Restart loop
        
        if removed_any:
            continue
            
        # Phase 2: Try swaps to enable removal
        # For each block i, find a replacement j such that unique_to_i is covered by j.
        swapped_any = False
        for i in range(len(current_blocks)):
            if time.time() - start_time > time_limit:
                break
            
            b_idx = current_blocks[i]
            uniques = get_unique_t_idx(b_idx)
            
            if not uniques:
                # Should have been caught in Phase 1, but just in case
                m = block_masks[b_idx]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    freq[idx] -= 1
                    m ^= lsb
                current_blocks.pop(i)
                swapped_any = True
                improved = True
                break
                
            # Find a block j not in current_blocks that covers all t-subsets in 'uniques'
            # Strategy: Pick one unique t-subset, say u0.
            # Candidates are in t_to_blocks[u0].
            # Filter candidates: must not be in current_blocks.
            
            u0 = uniques[0]
            candidates = t_to_blocks[u0]
            
            # Shuffle candidates to find different solutions?
            # For now, iterate deterministically or randomly.
            # Let's try to find ANY valid swap.
            
            found_swap = False
            for j in candidates:
                if j == b_idx:
                    continue
                if j in selected_set:
                    # If j is already in the set, swapping i for j doesn't change the set of blocks, just removes i and adds j? 
                    # If j is in selected_set, and we replace i with j, we just remove i.
                    # But we already checked if i is redundant (uniques empty). If uniques is not empty, i is NOT redundant.
                    # So removing i would lose coverage of 'uniques'.
                    # If j is in selected_set, does j cover 'uniques'?
                    # If j covers 'uniques', then 'uniques' are covered by j AND i. So freq[u] >= 2 for u in uniques.
                    # But 'uniques' are defined as freq==1. Contradiction.
                    # So if j is in selected_set, it CANNOT cover all 'uniques' because that would imply freq >= 2.
                    # Wait. freq[u] == 1 means ONLY i covers u.
                    # So NO other block in the current set covers u.
                    # So j (if in current set) does NOT cover u.
                    # So j cannot be a valid replacement if j is in current set.
                    continue
                
                # Check if j covers all other uniques
                valid = True
                mask_j = block_masks[j]
                for u in uniques:
                    if not (mask_j & (1 << u)):
                        valid = False
                        break
                if valid:
                    # Perform swap
                    # Remove i from freq
                    m_i = block_masks[b_idx]
                    while m_i:
                        lsb = m_i & -m_i
                        idx = lsb.bit_length() - 1
                        freq[idx] -= 1
                        m_i ^= lsb
                    
                    # Add j to freq
                    m_j = mask_j
                    while m_j:
                        lsb = m_j & -m_j
                        idx = lsb.bit_length() - 1
                        freq[idx] += 1
                        m_j ^= lsb
                        
                    # Update selected set
                    selected_set.remove(b_idx)
                    selected_set.add(j)
                    
                    # Update current_blocks list
                    current_blocks[i] = j
                    
                    found_swap = True
                    swapped_any = True
                    improved = True
                    break
                    
            if found_swap:
                break # Restart loop after a swap

    return [list(all_k_subs[i]) for i in current_blocks]
