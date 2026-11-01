import itertools
import numpy as np
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 105  # Safety margin for 120s limit
    
    # Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subsets_set = {tuple(s): i for i, s in enumerate(t_subsets)}
    num_tsubs = len(t_subsets)
    
    # Generate all k-subsets (blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(k_subsets)
    
    # Precompute for each block, which t-subsets it covers
    block_to_tsubs = []
    for block in k_subsets:
        covered_tsubs = []
        for tsub in itertools.combinations(block, t):
            idx = t_subsets_set[tsub]
            covered_tsubs.append(idx)
        block_to_tsubs.append(covered_tsubs)
        
    # Precompute for each t-subset, which blocks cover it
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for i, tsubs in enumerate(block_to_tsubs):
        for s in tsubs:
            tsub_to_blocks[s].append(i)
            
    # Convert tsub_to_blocks lists to numpy arrays for faster indexing
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # Initial gains: each block covers C(k, t) t-subsets
    k_choose_t = len(block_to_tsubs[0])
    gains = np.full(num_blocks, k_choose_t, dtype=np.int32)
    
    # Track uncovered t-subsets
    uncovered = np.ones(num_tsubs, dtype=bool)
    
    selected_block_indices = []
    
    while uncovered.any():
        if time.time() > deadline:
            break
            
        # Find the block with the maximum gain
        b = int(np.argmax(gains))
        
        if gains[b] == 0:
            break
            
        selected_block_indices.append(b)
        
        # Update gains for all blocks that share any t-subset with b
        for s in block_to_tsubs[b]:
            if uncovered[s]:
                uncovered[s] = False
                gains[tsub_to_blocks_np[s]] -= 1

    # Local optimization phase
    if time.time() < deadline - 10:
        selected_set = set(selected_block_indices)
        
        # Track coverage count for each t-subset
        coverage_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected_set:
            for s in block_to_tsubs[b]:
                coverage_count[s] += 1
        
        # Phase 1: Remove redundant blocks
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            for b in list(selected_set):
                if time.time() > deadline - 5:
                    break
                can_remove = True
                for s in block_to_tsubs[b]:
                    if coverage_count[s] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for s in block_to_tsubs[b]:
                        coverage_count[s] -= 1
                    selected_set.remove(b)
                    improved = True
                    break
        
        # Phase 2: Try 1-for-1 swaps
        if time.time() < deadline - 3:
            # Precompute a fast lookup: for each block, a set of t-subsets it covers
            block_tsub_set = [set(tsubs) for tsubs in block_to_tsubs]
            
            swap_improved = True
            while swap_improved:
                if time.time() > deadline - 2:
                    break
                swap_improved = False
                
                for b in list(selected_set):
                    if time.time() > deadline - 2:
                        break
                    
                    # Find t-subsets uniquely covered by b
                    unique_subs = [s for s in block_to_tsubs[b] if coverage_count[s] == 1]
                    if not unique_subs:
                        continue
                    
                    # Find candidate replacement blocks
                    candidates = set()
                    for s in unique_subs:
                        candidates.update(tsub_to_blocks[s])
                    candidates.discard(b)
                    candidates -= selected_set
                    
                    best_swap = None
                    best_new_uncovered = 0
                    
                    for cb in candidates:
                        # Check if cb covers all unique_subs
                        if not all(s in block_tsub_set[cb] for s in unique_subs):
                            continue
                        
                        # Count how many currently-uncovered t-subsets cb would cover
                        new_cov = 0
                        for s in block_to_tsubs[cb]:
                            if coverage_count[s] == 0:
                                new_cov += 1
                        
                        if new_cov > best_new_uncovered:
                            best_new_uncovered = new_cov
                            best_swap = cb
                    
                    if best_swap is not None:
                        for s in block_to_tsubs[b]:
                            coverage_count[s] -= 1
                        for s in block_to_tsubs[best_swap]:
                            coverage_count[s] += 1
                        selected_set.remove(b)
                        selected_set.add(best_swap)
                        swap_improved = True
                        break
        
        selected_block_indices = list(selected_set)
    
    # Convert selected block indices back to the actual subsets
    result = [list(k_subsets[i]) for i in selected_block_indices]
    
    return result
