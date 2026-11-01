import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (kill at 150s)
    
    # 1. Precomputation
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_t_subsets)
    t_subset_to_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Generate all k-subsets (blocks)
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_subsets)
    
    # Precompute coverage: which t-subsets does each block cover?
    # block_to_tsubs[block_idx] = list of t_subset indices
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
    # tsub_to_blocks[tsub_idx] = numpy array of block indices
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, tsubs in enumerate(block_to_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
            
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # Helper: Greedy Construction
    def greedy_construction(randomize_ties=False):
        uncovered = np.ones(num_tsubs, dtype=bool)
        k_choose_t = len(block_to_tsubs[0])
        gains = np.full(num_blocks, k_choose_t, dtype=np.int32)
        
        selected_block_indices = []
        
        while uncovered.any():
            if time.time() > deadline:
                return None # Timeout
            
            # Find block with max gain
            if randomize_ties and random.random() < 0.2:
                # Pick a random block from top 10 gains to break symmetry
                top_indices = np.argsort(gains)[-10:]
                b = random.choice(top_indices)
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
    
    for i in range(20):
        if time.time() > deadline:
            break
            
        # Randomize the greedy tie-breaking
        blocks = greedy_construction(randomize_ties=(i > 0))
        
        if blocks is None:
            continue
            
        # Local search
        optimized_blocks = local_optimize(blocks)
        
        if len(optimized_blocks) < best_len:
            best_len = len(optimized_blocks)
            best_solution = optimized_blocks
            
        # If we hit the theoretical lower bound or a very good number, we can stop early?
        # Lower bound is 131. Best known is 151.
        # If we get <= 151, it's a great result.
        if best_len <= 151:
            break

    if best_solution is None:
        # Fallback: run standard greedy without randomization if all failed?
        # Or return empty?
        # Greedy should always succeed.
        blocks = greedy_construction(randomize_ties=False)
        if blocks:
            best_solution = local_optimize(blocks)
        else:
            return []
            
    result = [list(all_k_subsets[i]) for i in best_solution]
    return result
