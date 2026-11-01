import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin for 120s limit (kill at 150s)
    
    # 1. Precomputation
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_t_subsets)
    t_subset_to_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Generate all k-subsets (blocks)
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_subsets)
    
    # Precompute coverage: which t-subsets does each block cover?
    block_to_tsubs = []
    block_to_tsubs_set = []
    for block in all_k_subsets:
        covered = []
        for tsub in itertools.combinations(block, t):
            idx = t_subset_to_index[tsub]
            covered.append(idx)
        block_to_tsubs.append(covered)
        block_to_tsubs_set.append(set(covered))
        
    # Precompute inverse: which blocks cover each t-subset?
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, tsubs in enumerate(block_to_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
            
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # Helper: Greedy Construction
    def greedy_construction(randomize_ties=False, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        uncovered = np.ones(num_tsubs, dtype=bool)
        k_choose_t = len(block_to_tsubs[0])
        gains = np.full(num_blocks, k_choose_t, dtype=np.int32)
        
        selected_block_indices = []
        
        while uncovered.any():
            if time.time() > deadline:
                return None # Timeout
            
            # Find block with max gain
            if randomize_ties:
                # Pick a random block from top 10 gains to break symmetry
                # Use argsort to get top indices efficiently
                # Note: argsort is O(N log N), but N=50k is small enough
                # Alternatively, we can just pick argmax and if there are ties, pick randomly?
                # Simpler: get top 10
                # Since gains are small ints, we can use bincount or just argsort
                # For speed, let's just use argmax usually, but occasionally randomize among high gainers
                # Let's stick to a simple randomization: with prob 0.2, pick from top 10
                if random.random() < 0.2:
                    # Get top 10
                    # np.argpartition is faster than argsort for top k
                    top_k = 10
                    if num_blocks > top_k:
                        # Get indices of top k gains
                        # argpartition returns indices that would partition the array
                        # The top k are in the last k positions of the returned array?
                        # Actually, np.argpartition(gains, -top_k)[-top_k:] gives the indices of the top k largest elements, unsorted
                        top_indices = np.argpartition(gains, -top_k)[-top_k:]
                        b = random.choice(top_indices)
                    else:
                        b = np.argmax(gains)
                else:
                    b = np.argmax(gains)
            else:
                b = np.argmax(gains)
            
            if gains[b] == 0:
                break
            
            selected_block_indices.append(b)
            
            # Update uncovered and gains
            for s_idx in block_to_tsubs[b]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    gains[tsub_to_blocks_np[s_idx]] -= 1
                    
        if uncovered.any():
            return None # Failed to cover
            
        return selected_block_indices

    # Helper: Local Optimization
    def local_optimize(blocks):
        if not blocks:
            return []
            
        # Use a set for O(1) lookup of selected blocks
        selected_set = set(blocks)
        selected_list = list(blocks)
        
        # Build cover_count: how many selected blocks cover each t-subset
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected_list:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1
        
        # Helper: Check if a block is redundant
        def is_redundant(b_idx):
            for s in block_to_tsubs[b_idx]:
                if cover_count[s] == 1:
                    return False
            return True
        
        # Helper: Find a replacement block for a set of unique t-subsets
        def find_replacement(unique_tsubs, exclude_set):
            if not unique_tsubs:
                return None
            
            # Get candidates from the first unique t-subset
            first_s = unique_tsubs[0]
            candidates = tsub_to_blocks_np[first_s]
            
            valid_candidates = []
            for c in candidates:
                if c in exclude_set:
                    continue
                # Check if c covers all other unique t-subsets
                c_set = block_to_tsubs_set[c]
                covers_all = True
                for s in unique_tsubs:
                    if s not in c_set:
                        covers_all = False
                        break
                
                if covers_all:
                    valid_candidates.append(c)
            
            if not valid_candidates:
                return None
            
            return random.choice(valid_candidates)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Step A: Redundancy Removal
            redundant_blocks = []
            for b in selected_list:
                if is_redundant(b):
                    redundant_blocks.append(b)
            
            if redundant_blocks:
                for b in redundant_blocks:
                    selected_list.remove(b)
                    selected_set.remove(b)
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
                improved = True
                continue # Restart loop to check for new redundancies
            
            # Step B: Swap
            # Calculate unique t-subsets for each block
            unique_tsubs_map = {}
            unique_counts = {}
            
            for b in selected_list:
                u_list = []
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        u_list.append(s)
                unique_tsubs_map[b] = u_list
                unique_counts[b] = len(u_list)
            
            if not unique_counts:
                break
                
            min_unique = min(unique_counts.values())
            
            # Candidates for removal: blocks with min_unique
            candidates_out = [b for b, c in unique_counts.items() if c == min_unique]
            
            # Shuffle to break symmetry
            random.shuffle(candidates_out)
            
            swap_made = False
            for b_out in candidates_out:
                s_uniq = unique_tsubs_map[b_out]
                
                # Find a replacement
                # Exclude all currently selected blocks
                exclude_set = selected_set
                
                b_in = find_replacement(s_uniq, exclude_set)
                
                if b_in is not None:
                    # Perform swap
                    selected_list.remove(b_out)
                    selected_set.remove(b_out)
                    selected_list.append(b_in)
                    selected_set.add(b_in)
                    
                    # Update cover counts
                    for s in block_to_tsubs[b_out]:
                        cover_count[s] -= 1
                    for s in block_to_tsubs[b_in]:
                        cover_count[s] += 1
                    
                    swap_made = True
                    improved = True
                    break # Restart loop after one swap
            
            if not swap_made:
                break
                
        return selected_list

    best_solution = None
    best_len = float('inf')
    
    # Multi-start strategy
    # We can afford several restarts.
    # Greedy is fast (~0.1s), Local Search is the bottleneck.
    # Let's try 10-20 restarts.
    
    # First run deterministic greedy
    blocks = greedy_construction(randomize_ties=False, random_seed=42)
    if blocks is not None:
        optimized_blocks = local_optimize(blocks)
        if len(optimized_blocks) < best_len:
            best_len = len(optimized_blocks)
            best_solution = optimized_blocks
    
    # Subsequent runs with randomization
    for i in range(1, 20):
        if time.time() > deadline:
            break
            
        # Randomize the greedy tie-breaking
        blocks = greedy_construction(randomize_ties=True, random_seed=42 + i)
        
        if blocks is None:
            continue
            
        # Local search
        optimized_blocks = local_optimize(blocks)
        
        if len(optimized_blocks) < best_len:
            best_len = len(optimized_blocks)
            best_solution = optimized_blocks
            
        # If we hit the theoretical lower bound or a very good number, we could stop early
        # But let's just use the time budget.
        
    if best_solution is None:
        # Fallback: return empty or a known valid small cover if possible?
        # But we must return a valid cover.
        # If greedy failed, something is wrong.
        return []
        
    result = [list(all_k_subsets[i]) for i in best_solution]
    return result
