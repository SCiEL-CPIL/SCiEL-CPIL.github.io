import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Precomputation
    # Map t-subsets to indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    
    # Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(k_blocks)
    
    # For each block, store the list of t-subset indices it covers
    block_t_indices = []
    for b in k_blocks:
        t_indices = []
        for sub in itertools.combinations(b, t):
            t_indices.append(t_sub_to_idx[sub])
        block_t_indices.append(t_indices)
        
    # For each t-subset, store the list of block indices that cover it
    t_to_blocks = [[] for _ in range(num_t_subs)]
    for b_idx, t_indices in enumerate(block_t_indices):
        for t_idx in t_indices:
            t_to_blocks[t_idx].append(b_idx)
            
    # Precompute block_t_sets for faster membership check in local search
    block_t_sets = [set(indices) for indices in block_t_indices]

    def greedy_cover():
        uncovered = [1] * num_t_subs  # 1 if uncovered, 0 if covered
        blocks = []
        
        while any(uncovered):
            # Find an uncovered t-subset with fewest candidates
            min_count = float('inf')
            chosen_t_idx = -1
            for i in range(num_t_subs):
                if uncovered[i]:
                    count = len(t_to_blocks[i])
                    if count < min_count:
                        min_count = count
                        chosen_t_idx = i
                        if count == 1:
                            break
            if chosen_t_idx == -1:
                break
                
            # Find the block among candidates for chosen_t_idx that covers the most uncovered t-subsets
            best_block = -1
            max_cover = -1
            candidates = t_to_blocks[chosen_t_idx]
            
            # To speed up, we can filter candidates
            # But 455 is small.
            for b_idx in candidates:
                cover_count = 0
                for t_idx in block_t_indices[b_idx]:
                    if uncovered[t_idx]:
                        cover_count += 1
                if cover_count > max_cover:
                    max_cover = cover_count
                    best_block = b_idx
                elif cover_count == max_cover and cover_count > 0:
                    # Random tie-breaking
                    if random.random() < 0.5:
                        best_block = b_idx
                    
            if best_block == -1:
                break
                
            blocks.append(best_block)
            for t_idx in block_t_indices[best_block]:
                uncovered[t_idx] = 0
                
        return blocks

    def local_search_optimized(blocks):
        current_blocks = set(blocks)
        cover_count = [0] * num_t_subs
        
        for b_idx in current_blocks:
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] += 1
                
        improved = True
        while improved:
            improved = False
            for b_idx in list(current_blocks):
                # Check if b_idx is essential
                essential = False
                for t_idx in block_t_indices[b_idx]:
                    if cover_count[t_idx] == 1:
                        essential = True
                        break
                
                if not essential:
                    for t_idx in block_t_indices[b_idx]:
                        cover_count[t_idx] -= 1
                    current_blocks.remove(b_idx)
                    improved = True
                else:
                    to_cover = [t_idx for t_idx in block_t_indices[b_idx] if cover_count[t_idx] == 1]
                    if not to_cover:
                        continue
                        
                    target_t = to_cover[0]
                    found_replacement = False
                    # Shuffle candidates to escape local minima?
                    # candidates = t_to_blocks[target_t]
                    # random.shuffle(candidates)
                    for b_new in t_to_blocks[target_t]:
                        if b_new in current_blocks:
                            continue
                            
                        # Check if b_new covers all t in to_cover
                        if all(t_idx in block_t_sets[b_new] for t_idx in to_cover):
                            # Replace b_idx with b_new
                            for t_idx in block_t_indices[b_idx]:
                                cover_count[t_idx] -= 1
                            for t_idx in block_t_indices[b_new]:
                                cover_count[t_idx] += 1
                            
                            current_blocks.remove(b_idx)
                            current_blocks.add(b_new)
                            improved = True
                            found_replacement = True
                            break
                            
        return list(current_blocks)

    best_blocks = None
    best_len = float('inf')
    
    while time.time() < deadline:
        blocks = greedy_cover()
        blocks = local_search_optimized(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        if best_len <= 145:
            break
            
    return [list(k_blocks[b_idx]) for b_idx in best_blocks]
