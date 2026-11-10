import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    points = list(range(v))
    
    # Precompute for each block (k-subset), the array of t-subset indices it covers
    # There are C(19,8) = 75582 blocks, each covers C(8,5) = 56 t-subsets
    # Precompute all block coverage as arrays
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map block (tuple) to index
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Precompute coverage: for each block, array of t-subset indices
    block_coverage = np.zeros((num_blocks, 56), dtype=np.int32)  # C(8,5) = 56
    for bi, block in enumerate(all_blocks):
        subs = list(itertools.combinations(block, t))
        for si, sub in enumerate(subs):
            block_coverage[bi, si] = t_subset_to_idx[sub]
    
    # For each t-subset, precompute which blocks contain it
    # t_subset_to_blocks: for each t-subset index, list of block indices
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]
    for bi in range(num_blocks):
        for si in range(56):
            tsub_idx = block_coverage[bi, si]
            t_subset_to_blocks[tsub_idx].append(bi)
    
    def greedy_once(random_seed):
        """Fast greedy: pick uncovered t-subset, find best block covering it."""
        rng = random.Random(random_seed)
        uncovered = np.ones(num_t_subsets, dtype=bool)
        blocks = []
        
        while np.any(uncovered):
            if time.time() > time_limit - 3.0:
                return None
            
            # Pick a random uncovered t-subset
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            s_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
            
            # Find all blocks containing this t-subset
            candidate_blocks = t_subset_to_blocks[s_idx]
            
            if not candidate_blocks:
                break
            
            # Find the block with maximum gain (most uncovered t-subsets covered)
            best_bi = -1
            best_gain = -1
            best_block = None
            
            for bi in candidate_blocks:
                cov = block_coverage[bi]
                gain = np.sum(uncovered[cov])
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
                    best_block = all_blocks[bi]
                elif gain == best_gain and gain > 0 and rng.random() < 0.3:
                    best_bi = bi
                    best_block = all_blocks[bi]
            
            if best_bi == -1 or best_gain <= 0:
                break
            
            blocks.append(list(best_block))
            
            # Update uncovered
            cov = block_coverage[best_bi]
            uncovered[cov] = False
        
        return blocks
    
    def local_search_remove(blocks, random_seed):
        """Try to remove redundant blocks."""
        rng = random.Random(random_seed)
        n = len(blocks)
        
        # Compute coverage count for each t-subset
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        block_masks = []
        for block in blocks:
            mask = np.zeros(num_t_subsets, dtype=bool)
            for sub in itertools.combinations(block, t):
                idx = t_subset_to_idx[sub]
                mask[idx] = True
            cover_count += mask.astype(np.int32)
            block_masks.append(mask)
        
        removed = set()
        improved = True
        
        while improved and time.time() < time_limit - 5.0:
            improved = False
            active = [i for i in range(n) if i not in removed]
            rng.shuffle(active)
            
            for bi in active:
                if bi in removed:
                    continue
                if time.time() > time_limit - 5.0:
                    break
                
                # Check if block bi is redundant: all its t-subsets are covered by others
                mask = block_masks[bi]
                # A t-subset is exclusively covered by this block if cover_count == 1
                # Check if any t-subset has cover_count == 1 in this block's coverage
                covered_by_this = mask
                if np.any(cover_count[covered_by_this] <= 1):
                    continue  # Not redundant
                
                # Remove this block
                removed.add(bi)
                cover_count -= mask.astype(np.int32)
                improved = True
        
        if not removed:
            return blocks
        return [b for i, b in enumerate(blocks) if i not in removed]
    
    best_blocks = None
    best_len = float('inf')
    
    num_restarts = 0
    while time.time() < time_limit - 10.0:
        seed = num_restarts
        res = greedy_once(seed)
        num_restarts += 1
        
        if res is not None:
            # Apply local search removal
            res = local_search_remove(res, seed)
            
            if len(res) < best_len:
                best_len = len(res)
                best_blocks = res
        
        if best_len <= 264:
            break
    
    if best_blocks is None:
        best_blocks = greedy_once(0)
        if best_blocks:
            best_blocks = local_search_remove(best_blocks, 0)
    
    return best_blocks
