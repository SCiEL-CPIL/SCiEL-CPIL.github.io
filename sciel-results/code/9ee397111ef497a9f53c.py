import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Precompute all t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Bit masks for each t-subset
    t_masks = [1 << i for i in range(num_t_subsets)]
    all_ones_mask = (1 << num_t_subsets) - 1
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy instances
    max_runs = 200 # Adjust based on time
    
    for run_idx in range(max_runs):
        if time.time() > deadline:
            break
            
        random.seed(run_idx) # Deterministic per run, different across runs
        
        covered = 0
        blocks = []
        uncovered = all_ones_mask
        
        # Greedy Construction
        while uncovered:
            if time.time() > deadline:
                break
                
            # Pick a seed t-subset from uncovered
            # Use random selection to diversify
            seed_idx = None
            # Try a few random picks
            for _ in range(5):
                cand_idx = random.randint(0, num_t_subsets - 1)
                if (uncovered >> cand_idx) & 1:
                    seed_idx = cand_idx
                    break
            
            if seed_idx is None:
                # Fallback to lowest set bit
                seed_idx = (uncovered & -uncovered).bit_length() - 1
                
            seed = all_t_subsets[seed_idx]
            
            # Build block starting with seed
            block = list(seed)
            block_mask = t_masks[seed_idx]
            
            while len(block) < k:
                best_x = -1
                best_gain = -1
                best_new_mask = 0
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    # Calculate mask of block U {x}
                    new_mask = block_mask
                    for sub in itertools.combinations(block, t-1):
                        new_t_sub = tuple(sorted(sub + (x,)))
                        idx = subset_to_idx[new_t_sub]
                        new_mask |= t_masks[idx]
                    
                    gain = (new_mask & uncovered).bit_count()
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_new_mask = new_mask
                
                if best_x == -1:
                    break
                    
                block.append(best_x)
                block_mask = best_new_mask
            
            block.sort()
            blocks.append(block)
            covered |= block_mask
            uncovered &= ~block_mask
            
        # Local Search: Elimination
        # Precompute masks for current blocks
        block_masks = []
        for b in blocks:
            m = 0
            for sub in itertools.combinations(b, t):
                m |= t_masks[subset_to_idx[sub]]
            block_masks.append(m)
        
        # Iterative elimination
        changed = True
        while changed and time.time() < deadline:
            changed = False
            i = 0
            while i < len(blocks):
                if time.time() > deadline:
                    break
                others_mask = 0
                for j in range(len(blocks)):
                    if j != i:
                        others_mask |= block_masks[j]
                    
                unique_to_i = block_masks[i] & ~others_mask
                
                if unique_to_i == 0:
                    blocks.pop(i)
                    block_masks.pop(i)
                    changed = True
                else:
                    i += 1
                    
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        # Fallback: return empty or a single block? 
        # The problem guarantees a solution exists.
        # If we timed out before finishing even one run, we might have partial blocks.
        # But the loop breaks on deadline. 
        # If no run finished, best_blocks is None.
        # We should ensure at least one run completes or handle this.
        # Given the speed, it's very unlikely.
        return []
        
    return best_blocks
