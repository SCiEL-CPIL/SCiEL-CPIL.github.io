import itertools
import numpy as np
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_t_subsets)
    t_subset_to_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_subsets)
    
    # Precompute coverage
    block_to_tsubs = []
    for block in all_k_subsets:
        covered = []
        for tsub in itertools.combinations(block, t):
            idx = t_subset_to_index[tsub]
            covered.append(idx)
        block_to_tsubs.append(covered)
    
    # Inverse: which blocks cover each t-subset
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, tsubs in enumerate(block_to_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
    
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # Use numpy for efficient bit operations
    # Represent coverage as numpy boolean arrays
    block_masks = []
    for tsubs in block_to_tsubs:
        mask = np.zeros(num_tsubs, dtype=bool)
        for s in tsubs:
            mask[s] = True
        block_masks.append(mask)
    
    def greedy_construction(strategy='max_gain'):
        """
        strategy: 'max_gain' - pick block with max new coverage
                  'min_uncovered' - pick block covering most currently uncovered
                  'hybrid' - mix of both
        """
        uncovered = np.ones(num_tsubs, dtype=bool)
        selected = []
        
        while uncovered.any():
            if time.time() > deadline:
                return None
            
            if strategy == 'max_gain':
                # For each block, count how many currently uncovered tsubs it covers
                gains = np.array([np.sum(block_masks[b] & uncovered) for b in range(num_blocks)])
                best_block = int(np.argmax(gains))
                if gains[best_block] == 0:
                    return None
            else:
                # Pick a random uncovered tsub, then pick best block covering it
                uncovered_indices = np.where(uncovered)[0]
                s_idx = int(random.choice(uncovered_indices))
                candidates = tsub_to_blocks_np[s_idx]
                gains = np.array([np.sum(block_masks[b] & uncovered) for b in candidates])
                best_in_candidates = int(candidates[np.argmax(gains)])
                if gains[np.argmax(gains)] == 0:
                    return None
                best_block = best_in_candidates
            
            selected.append(best_block)
            uncovered[block_masks[best_block]] = False
        
        return selected
    
    def local_optimize(blocks, max_iterations=100):
        """
        Aggressive local search with 1-swap, 2-swap, and 3-swap moves.
        """
        if not blocks:
            return []
        
        selected_set = set(blocks)
        selected_list = list(blocks)
        
        # Build cover_count
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected_list:
            cover_count[block_to_tsubs[b]] += 1
        
        def is_redundant(b_idx):
            """Check if removing b_idx still leaves all tsubs covered"""
            for s in block_to_tsubs[b_idx]:
                if cover_count[s] == 1:
                    return False
            return True
        
        def find_replacement_for_unique(unique_tsubs, exclude_set, max_candidates=50):
            """Find a block that covers all unique_tsubs and is not in exclude_set"""
            if not unique_tsubs:
                return None
            
            # Use the first unique tsub to get candidates
            first_s = unique_tsubs[0]
            candidates = tsub_to_blocks_np[first_s]
            
            valid = []
            for c in candidates:
                if c in exclude_set:
                    continue
                # Check if c covers all unique tsubs
                c_mask = block_masks[c]
                all_covered = all(c_mask[s] for s in unique_tsubs)
                if all_covered:
                    valid.append(c)
                    if len(valid) >= max_candidates:
                        break
            
            if not valid:
                return None
            return random.choice(valid)
        
        improved = True
        iteration = 0
        while improved and iteration < max_iterations and time.time() < deadline:
            improved = False
            iteration += 1
            
            # Step 1: Remove redundant blocks
            to_remove = [b for b in selected_list if is_redundant(b)]
            if to_remove:
                for b in to_remove:
                    selected_list.remove(b)
                    selected_set.remove(b)
                    cover_count[block_to_tsubs[b]] -= 1
                improved = True
                continue
            
            # Step 2: 1-swap - try to replace one block with a better one
            # Find block with minimum unique coverage
            unique_counts = {}
            unique_tsubs_map = {}
            for b in selected_list:
                u_list = [s for s in block_to_tsubs[b] if cover_count[s] == 1]
                unique_tsubs_map[b] = u_list
                unique_counts[b] = len(u_list)
            
            if selected_list:
                min_unique = min(unique_counts.values())
                candidates_out = [b for b, c in unique_counts.items() if c == min_unique]
                random.shuffle(candidates_out)
                
                swap_made = False
                for b_out in candidates_out:
                    s_uniq = unique_tsubs_map[b_out]
                    b_in = find_replacement_for_unique(s_uniq, selected_set)
                    if b_in is not None:
                        # Perform swap
                        selected_list.remove(b_out)
                        selected_set.remove(b_out)
                        selected_list.append(b_in)
                        selected_set.add(b_in)
                        cover_count[block_to_tsubs[b_out]] -= 1
                        cover_count[block_to_tsubs[b_in]] += 1
                        swap_made = True
                        improved = True
                        break
                
                if swap_made:
                    continue
            
            # Step 3: 2-swap - try to replace 2 blocks with 1 or 2 blocks
            if len(selected_list) >= 2 and time.time() < deadline:
                # Try removing pairs and finding replacements
                # This is expensive, so only try a limited number of pairs
                n_pairs = min(20, len(selected_list) * (len(selected_list) - 1) // 2)
                pairs = list(itertools.combinations(range(len(selected_list)), 2))
                random.shuffle(pairs)
                
                two_swap_made = False
                for i, j in pairs[:n_pairs]:
                    if time.time() > deadline:
                        break
                    
                    b1 = selected_list[i]
                    b2 = selected_list[j]
                    
                    # Find unique tsubs that would be uncovered if we remove both
                    temp_cover = cover_count.copy()
                    temp_cover[block_to_tsubs[b1]] -= 1
                    temp_cover[block_to_tsubs[b2]] -= 1
                    
                    unique_after_removal = [s for s in range(num_tsubs) if temp_cover[s] == 0]
                    
                    if not unique_after_removal:
                        # Both blocks are redundant together - remove them
                        selected_list.remove(b1)
                        selected_list.remove(b2)
                        selected_set.remove(b1)
                        selected_set.remove(b2)
                        cover_count[block_to_tsubs[b1]] -= 1
                        cover_count[block_to_tsubs[b2]] -= 1
                        two_swap_made = True
                        improved = True
                        break
                    
                    # Try to find 1 or 2 blocks that cover all unique_after_removal
                    # First try 1 block
                    b_in1 = find_replacement_for_unique(unique_after_removal, selected_set, max_candidates=20)
                    if b_in1 is not None:
                        selected_list.remove(b1)
                        selected_list.remove(b2)
                        selected_set.remove(b1)
                        selected_set.remove(b2)
                        selected_list.append(b_in1)
                        selected_set.add(b_in1)
                        cover_count[block_to_tsubs[b1]] -= 1
                        cover_count[block_to_tsubs[b2]] -= 1
                        cover_count[block_to_tsubs[b_in1]] += 1
                        two_swap_made = True
                        improved = True
                        break
                
                if two_swap_made:
                    continue
            
            # No improvement found
            break
        
        return selected_list
    
    best_solution = None
    best_len = float('inf')
    
    # Multi-start with different strategies
    strategies = ['max_gain', 'min_uncovered', 'max_gain', 'min_uncovered']
    num_starts = 8
    
    for i in range(num_starts):
        if time.time() > deadline:
            break
        
        strategy = strategies[i % len(strategies)]
        random.seed(i * 12345 + 678)  # Different seed for diversity
        
        blocks = greedy_construction(strategy=strategy)
        
        if blocks is None:
            continue
        
        optimized_blocks = local_optimize(blocks)
        
        if len(optimized_blocks) < best_len:
            best_len = len(optimized_blocks)
            best_solution = optimized_blocks
        
        if best_len <= 151:
            break
    
    if best_solution is None:
        blocks = greedy_construction(strategy='max_gain')
        if blocks:
            best_solution = local_optimize(blocks)
        else:
            return []
    
    result = [list(all_k_subsets[i]) for i in best_solution]
    return result
