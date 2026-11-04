import itertools
import time
import random
import numpy as np
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit

    # Precompute all t-subsets and their indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}

    # Precompute all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)

    # For each t-subset, store the list of block indices that cover it
    # This is the "inverse" of the coverage matrix
    # t_sub_blocks[t_idx] = list of block indices that contain this t-subset
    t_sub_blocks = [[] for _ in range(n_t_subs)]
    for bi, block in enumerate(all_k_blocks):
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            t_sub_blocks[idx].append(bi)

    # Helper: compute coverage count for a set of blocks
    def compute_cover_count(block_indices):
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for bi in block_indices:
            block = all_k_blocks[bi]
            for t_sub in itertools.combinations(block, t):
                idx = sub_to_idx[t_sub]
                cover_count[idx] += 1
        return cover_count

    # Helper: check if a set of blocks covers everything
    def is_complete(block_indices):
        cover_count = compute_cover_count(block_indices)
        return np.all(cover_count >= 1)

    # Helper: find uncovered t-subsets
    def find_uncovered(block_indices):
        cover_count = compute_cover_count(block_indices)
        return np.where(cover_count == 0)[0]

    # Phase 1: Greedy set cover with randomized tie-breaking
    def greedy_cover_randomized():
        uncovered = set(range(n_t_subs))
        selected = []
        while uncovered:
            if time.time() > deadline - 60:
                break
            best_gain = -1
            candidates = []
            for bi in range(n_k_blocks):
                gain = 0
                block = all_k_blocks[bi]
                for t_sub in itertools.combinations(block, t):
                    idx = sub_to_idx[t_sub]
                    if idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    candidates = [bi]
                elif gain == best_gain and gain > 0:
                    candidates.append(bi)
            if not candidates or best_gain <= 0:
                break
            # Randomly pick among best candidates
            chosen = random.choice(candidates)
            selected.append(chosen)
            block = all_k_blocks[chosen]
            for t_sub in itertools.combinations(block, t):
                idx = sub_to_idx[t_sub]
                uncovered.discard(idx)
        return selected

    # Phase 2: Remove redundant blocks
    def remove_redundant(blocks):
        if not blocks:
            return blocks
        cover_count = compute_cover_count(blocks)
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for i, bi in enumerate(blocks):
                block = all_k_blocks[bi]
                is_critical = False
                for t_sub in itertools.combinations(block, t):
                    idx = sub_to_idx[t_sub]
                    if cover_count[idx] == 1:
                        is_critical = True
                        break
                if is_critical:
                    new_blocks.append(bi)
                else:
                    for t_sub in itertools.combinations(block, t):
                        idx = sub_to_idx[t_sub]
                        cover_count[idx] -= 1
                    changed = True
            blocks = new_blocks
        return blocks

    # Phase 3: Local search - try to replace blocks with better ones
    def local_search(blocks, max_iters=200):
        blocks = blocks[:]
        current_cover = compute_cover_count(blocks)
        
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            improved = False
            
            # Try removing each block and see if we can cover the gaps with fewer blocks
            for i in range(len(blocks)):
                if time.time() > deadline - 10:
                    break
                bi = blocks[i]
                block = all_k_blocks[bi]
                
                # Find which t-subsets would become uncovered
                newly_uncovered = []
                for t_sub in itertools.combinations(block, t):
                    idx = sub_to_idx[t_sub]
                    if current_cover[idx] == 1:
                        newly_uncovered.append(idx)
                
                if not newly_uncovered:
                    # Can remove this block
                    blocks.pop(i)
                    for t_sub in itertools.combinations(block, t):
                        idx = sub_to_idx[t_sub]
                        current_cover[idx] -= 1
                    improved = True
                    break
                
                # Try to find a single block that covers all newly_uncovered
                # and also provides good coverage
                target_set = set(newly_uncovered)
                best_replacement = -1
                best_extra = -1
                for j in range(n_k_blocks):
                    if j in blocks:
                        continue
                    block_j = all_k_blocks[j]
                    covers_target = True
                    extra = 0
                    for t_sub in itertools.combinations(block_j, t):
                        idx = sub_to_idx[t_sub]
                        if idx in target_set:
                            target_set.discard(idx)
                        elif current_cover[idx] == 1:
                            # This would create redundancy issues
                            pass
                        else:
                            extra += 1
                    if target_set:  # Not all covered
                        # Reset and skip
                        for idx in newly_uncovered:
                            if idx not in target_set:
                                target_set.add(idx)
                        continue
                    if extra > best_extra:
                        best_extra = extra
                        best_replacement = j
                
                if best_replacement != -1:
                    # Replace block i with best_replacement
                    blocks[i] = best_replacement
                    # Update coverage
                    for t_sub in itertools.combinations(block, t):
                        idx = sub_to_idx[t_sub]
                        current_cover[idx] -= 1
                    new_block = all_k_blocks[best_replacement]
                    for t_sub in itertools.combinations(new_block, t):
                        idx = sub_to_idx[t_sub]
                        current_cover[idx] += 1
                    improved = True
                    break
            
            if not improved:
                # Try adding a block and then removing two
                # This is more complex, skip for now
                break
        
        return blocks

    # Phase 4: Multi-start greedy with local search
    best_solution = None
    best_size = float('inf')
    
    n_starts = 10
    for s in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        # Randomized greedy
        solution = greedy_cover_randomized()
        if not solution:
            continue
        
        # Remove redundant
        solution = remove_redundant(solution)
        
        # Local search
        solution = local_search(solution)
        
        # Final redundancy removal
        solution = remove_redundant(solution)
        
        if len(solution) < best_size:
            best_size = len(solution)
            best_solution = solution[:]
    
    # If we didn't get a good solution, use a simple greedy
    if best_solution is None:
        best_solution = greedy_cover_randomized()
        best_solution = remove_redundant(best_solution)
    
    # Convert to list of lists
    blocks = [list(all_k_blocks[i]) for i in best_solution]
    
    return blocks
