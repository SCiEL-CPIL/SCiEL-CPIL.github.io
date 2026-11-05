import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Precompute all t-subsets and assign them indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # For each possible k-block, precompute which t-subsets it covers
    # C(19,7) = 50388 blocks, each covers C(7,4) = 35 t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute coverage bitsets for each block
    # Use Python integers as bitsets
    block_coverage = [0] * n_blocks
    for bi, block in enumerate(all_blocks):
        mask = 0
        for tsub in itertools.combinations(block, t):
            idx = t_sub_index[tsub]
            mask |= (1 << idx)
        block_coverage[bi] = mask
    
    # Precompute block index lookup
    block_to_idx = {block: i for i, block in enumerate(all_blocks)}
    
    # Greedy set cover using bitsets
    def greedy_set_cover(seed, max_blocks=None):
        random.seed(seed)
        uncovered = (1 << n_t_subs) - 1  # All bits set
        blocks = []
        
        while uncovered and (max_blocks is None or len(blocks) < max_blocks):
            if time.time() > deadline - 5:
                break
            
            # Find block with maximum coverage of uncovered subsets
            best_gain = -1
            best_block_idx = -1
            best_gain_list = []
            
            # Sample a subset of blocks to speed up (or use all if fast enough)
            # For v=19, k=7, we have ~50K blocks. Let's try all first.
            for bi in range(n_blocks):
                cov = block_coverage[bi]
                gain = bin(cov & uncovered).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = bi
                    best_gain_list = [bi]
                elif gain == best_gain and gain > 0:
                    best_gain_list.append(bi)
            
            if best_gain <= 0:
                break
            
            # Pick randomly from best candidates for diversity
            if best_gain_list:
                chosen = random.choice(best_gain_list)
            else:
                chosen = best_block_idx
            
            blocks.append(all_blocks[chosen])
            uncovered &= ~block_coverage[chosen]
        
        return blocks
    
    # Local search: try to replace a block with another that maintains coverage
    def local_search(blocks, seed, max_iterations=100):
        random.seed(seed)
        blocks = list(blocks)
        
        # Build coverage map: which blocks cover each t-subset
        # And count how many times each t-subset is covered
        coverage_count = [0] * n_t_subs
        for block in blocks:
            for tsub in itertools.combinations(block, t):
                idx = t_sub_index[tsub]
                coverage_count[idx] += 1
        
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try removing each block and see if we can replace it
            for i in range(len(blocks)):
                if time.time() > deadline - 10:
                    return blocks
                
                block_i = blocks[i]
                # Find t-subsets only covered by block_i
                essential = []
                for tsub in itertools.combinations(block_i, t):
                    idx = t_sub_index[tsub]
                    if coverage_count[idx] == 1:
                        essential.append(idx)
                
                if not essential:
                    # Block i is redundant, remove it
                    del blocks[i]
                    # Update coverage counts
                    for tsub in itertools.combinations(block_i, t):
                        idx = t_sub_index[tsub]
                        coverage_count[idx] -= 1
                    improved = True
                    break
                else:
                    # Try to find a replacement block that covers all essential subsets
                    # and has maximum additional coverage
                    best_replace = None
                    best_add_gain = -1
                    
                    for bi in range(n_blocks):
                        if all_blocks[bi] == block_i:
                            continue
                        cov = block_coverage[bi]
                        
                        # Check if it covers all essential subsets
                        essential_mask = 0
                        for idx in essential:
                            essential_mask |= (1 << idx)
                        
                        if (cov & essential_mask) != essential_mask:
                            continue
                        
                        # Count additional coverage (t-subsets not currently covered)
                        add_gain = bin(cov & ~((1 << n_t_subs) - 1)).count('1')
                        # Actually, we want t-subsets that are uncovered or have low coverage
                        # Simplified: count t-subsets in cov that have coverage_count == 0
                        # But we need to be careful about double counting
                        # For now, just use a heuristic: prefer blocks with more total coverage
                        total_gain = bin(cov).count('1')
                        
                        if total_gain > best_add_gain:
                            best_add_gain = total_gain
                            best_replace = bi
                    
                    if best_replace is not None:
                        # Replace block_i with all_blocks[best_replace]
                        old_block = block_i
                        new_block = all_blocks[best_replace]
                        
                        # Update coverage counts
                        for tsub in itertools.combinations(old_block, t):
                            idx = t_sub_index[tsub]
                            coverage_count[idx] -= 1
                        for tsub in itertools.combinations(new_block, t):
                            idx = t_sub_index[tsub]
                            coverage_count[idx] += 1
                        
                        blocks[i] = new_block
                        improved = True
                        break
        
        return blocks
    
    # Try multiple greedy runs with different seeds
    best_blocks = None
    best_count = float('inf')
    
    trial = 0
    while time.time() < deadline - 30:
        # Run greedy set cover
        blocks = greedy_set_cover(trial)
        
        if blocks and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks.copy()
        
        trial += 1
    
    # If we found a good solution, try local search refinement
    if best_blocks is not None and time.time() < deadline - 10:
        ls_blocks = local_search(best_blocks, 42, max_iterations=50)
        if len(ls_blocks) < best_count:
            best_count = len(ls_blocks)
            best_blocks = ls_blocks
    
    # Fallback if nothing was found
    if best_blocks is None:
        best_blocks = greedy_set_cover(0)
    
    return best_blocks
