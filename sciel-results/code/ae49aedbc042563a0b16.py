import itertools
import random

def construct(v, k, t):
    all_points = list(range(v))
    
    # 1. Precompute t-subsets and assign IDs
    t_combs = list(itertools.combinations(all_points, t))
    num_t = len(t_combs)
    t_id = {comb: i for i, comb in enumerate(t_combs)}
    
    # 2. Precompute all k-blocks and their coverage masks
    k_combs = list(itertools.combinations(all_points, k))
    num_blocks = len(k_combs)
    
    block_masks = [0] * num_blocks
    # t_to_blocks[i] will store list of block indices covering t-subset i
    t_to_blocks = [[] for _ in range(num_t)]
    
    for b_idx, block in enumerate(k_combs):
        block_set = set(block)
        mask = 0
        # Find all t-subsets contained in this block
        for t_comb in itertools.combinations(block, t):
            tid = t_id[t_comb]
            mask |= (1 << tid)
            t_to_blocks[tid].append(b_idx)
        block_masks[b_idx] = mask
        
    # 3. Greedy Initial Solution
    def greedy_cover_seeded(rng):
        uncovered_mask = (1 << num_t) - 1
        selected = []
        while uncovered_mask:
            best_idx = -1
            best_count = 0
            # Optimization: Iterate all blocks
            # To speed up, we can break early if we find a block covering all remaining?
            # Unlikely to cover all remaining in one go.
            
            # We can also randomize the order to find different local optima
            indices = list(range(num_blocks))
            rng.shuffle(indices)
            
            for b_idx in indices:
                mask = block_masks[b_idx]
                # Count how many currently uncovered t-subsets this block covers
                cnt = (mask & uncovered_mask).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_idx = b_idx
                    if cnt == uncovered_mask.bit_count():
                        break # Can't do better
            
            if best_idx != -1:
                selected.append(best_idx)
                uncovered_mask &= ~block_masks[best_idx]
            else:
                break # Should not happen
                
        return selected

    # 4. Local Search Optimization
    def optimize(selected_indices, rng):
        # Initialize cover counts
        cover_counts = [0] * num_t
        for b_idx in selected_indices:
            mask = block_masks[b_idx]
            m = mask
            while m:
                lsb = m & -m
                tid = lsb.bit_length() - 1
                cover_counts[tid] += 1
                m ^= lsb
                
        selected_set = set(selected_indices)
        
        improved = True
        max_iterations = 200 # Safety valve to prevent long loops
        iter_count = 0
        
        while improved and iter_count < max_iterations:
            improved = False
            iter_count += 1
            
            # Iterate through a copy of selected indices
            current_selected = list(selected_set)
            rng.shuffle(current_selected)
            
            for b_idx in current_selected:
                if b_idx not in selected_set:
                    continue
                
                # Check if we can remove b_idx
                mask = block_masks[b_idx]
                broken_tids = []
                m = mask
                while m:
                    lsb = m & -m
                    tid = lsb.bit_length() - 1
                    if cover_counts[tid] == 1:
                        broken_tids.append(tid)
                    m ^= lsb
                
                if not broken_tids:
                    # Safe to remove
                    m = mask
                    while m:
                        lsb = m & -m
                        tid = lsb.bit_length() - 1
                        cover_counts[tid] -= 1
                        m ^= lsb
                    
                    selected_set.remove(b_idx)
                    improved = True
                else:
                    # Need replacement
                    candidate_blocks = set(t_to_blocks[broken_tids[0]])
                    
                    for tid in broken_tids[1:]:
                        candidate_blocks &= set(t_to_blocks[tid])
                        if not candidate_blocks:
                            break
                    
                    candidate_blocks -= selected_set
                    
                    if candidate_blocks:
                        new_b_idx = rng.choice(list(candidate_blocks))
                        
                        # Perform swap
                        m = mask
                        while m:
                            lsb = m & -m
                            tid = lsb.bit_length() - 1
                            cover_counts[tid] -= 1
                            m ^= lsb
                        
                        selected_set.remove(b_idx)
                        
                        new_mask = block_masks[new_b_idx]
                        m = new_mask
                        while m:
                            lsb = m & -m
                            tid = lsb.bit_length() - 1
                            cover_counts[tid] += 1
                            m ^= lsb
                        
                        selected_set.add(new_b_idx)
                        # Swap doesn't reduce size, but may enable future reductions.
                        # We don't set improved=True here to avoid infinite swapping loops
                        # unless we add a counter. The outer loop restarts the scan.
        
        return list(selected_set)

    # Main
    best_solution = None
    best_size = float('inf')
    
    num_restarts = 5
    base_seed = 42
    
    for r in range(num_restarts):
        rng = random.Random(base_seed + r)
        init_blocks = greedy_cover_seeded(rng)
        opt_blocks = optimize(init_blocks, rng)
        
        if len(opt_blocks) < best_size:
            best_size = len(opt_blocks)
            best_solution = opt_blocks[:]
            
            if best_size <= 264:
                break

    final_blocks = [list(k_combs[i]) for i in best_solution]
    
    # Ensure distinct
    seen = set()
    unique_blocks = []
    for b in final_blocks:
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)
            
    return unique_blocks
