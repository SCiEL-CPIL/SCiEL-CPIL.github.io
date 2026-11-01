import numpy as np
import itertools
import time
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    time_limit = 140.0  # Increased safety margin
    
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    # Generate all k-subsets
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res
            
    t_count = nCr(k, t)
    
    # Precompute block_t_indices
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]
            
    # Precompute sets of triple indices for each block for faster swap/replacement checks
    # This uses ~100MB memory but significantly speeds up local search
    block_sets = [set(row) for row in block_t_indices]
    
    best_solution = None
    best_score = float('inf')
    
    def run_greedy(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        
        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None, None
            
            covered_counts = covered[block_t_indices].sum(axis=1)
            gains = t_count - covered_counts
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            if n_ks > 20:
                top_indices = np.argpartition(gains, -20)[-20:]
            else:
                top_indices = np.arange(n_ks)
                
            best_idx = top_indices[random.randint(0, len(top_indices) - 1)]
            
            blocks_indices.append(best_idx)
            
            subs_to_cover = block_t_indices[best_idx]
            prev_state = covered[subs_to_cover]
            new_covers = np.sum(~prev_state)
            remaining -= new_covers
            covered[subs_to_cover] = True
            
        current_blocks = list(blocks_indices)
        
        # Local Search Phase
        if not current_blocks:
            return current_blocks, 0
            
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
            
        improved = True
        while improved and (time.time() - start_time < time_limit - 2.0):
            improved = False
            new_blocks = []
            
            # Pass 1: Deletion
            random.shuffle(current_blocks)
            for b_idx in current_blocks:
                subs_in_block = block_t_indices[b_idx]
                if np.min(coverage_count[subs_in_block]) > 1:
                    coverage_count[subs_in_block] -= 1
                    improved = True
                else:
                    new_blocks.append(b_idx)
            
            current_blocks = new_blocks
            
            # If no deletion happened, try swaps and replacements
            if not improved and len(current_blocks) > 0:
                swaps_tried = 0
                max_swaps = 100 # Increased
                swap_applied = False
                
                while swaps_tried < max_swaps and not swap_applied and (time.time() - start_time < time_limit - 2.0):
                    swaps_tried += 1
                    b_pos = random.randint(0, len(current_blocks) - 1)
                    b_idx = current_blocks[b_pos]
                    old_set = block_sets[b_idx]
                    
                    # Strategy A: 1-Point Swap
                    if random.random() < 0.5:
                        block = all_ks[b_idx]
                        block_list = list(block)
                        x = random.choice(block_list)
                        y = random.randint(0, v - 1)
                        if y in block:
                            continue
                            
                        new_block_list = [p for p in block_list if p != x]
                        new_block_list.append(y)
                        new_block_list.sort()
                        new_block_tuple = tuple(new_block_list)
                        
                        # Find index
                        # Using bisect is O(log N), but we can also just compute the set
                        # However, we need the index to update current_blocks
                        # We don't have a dict for block->index.
                        # Let's use bisect.
                        import bisect
                        pos = bisect.bisect_left(all_ks, new_block_tuple)
                        if pos >= n_ks or all_ks[pos] != new_block_tuple:
                            continue
                            
                        new_set = block_sets[pos]
                        lost = old_set - new_set
                        gained = new_set - old_set
                        
                        valid = True
                        for idx in lost:
                            if coverage_count[idx] <= 1:
                                valid = False
                                break
                                
                        if valid:
                            for idx in lost:
                                coverage_count[idx] -= 1
                            for idx in gained:
                                coverage_count[idx] += 1
                            current_blocks[b_pos] = pos
                            swap_applied = True
                            improved = True
                    else:
                        # Strategy B: Block Replacement
                        # Pick a random block from the universe
                        new_idx = random.randint(0, n_ks - 1)
                        if new_idx in current_blocks:
                            continue
                            
                        new_set = block_sets[new_idx]
                        lost = old_set - new_set
                        gained = new_set - old_set
                        
                        valid = True
                        for idx in lost:
                            if coverage_count[idx] <= 1:
                                valid = False
                                break
                                
                        if valid:
                            for idx in lost:
                                coverage_count[idx] -= 1
                            for idx in gained:
                                coverage_count[idx] += 1
                            current_blocks[b_pos] = new_idx
                            swap_applied = True
                            improved = True
                        
        return current_blocks, len(current_blocks)

    restart_count = 0
    # Run multiple restarts
    while time.time() - start_time < time_limit - 10.0:
        seed = restart_count
        blocks, score = run_greedy(seed)
        
        if blocks is None:
            break
            
        if score < best_score:
            best_score = score
            best_solution = blocks
            
        restart_count += 1
        
    if best_solution is None:
        blocks, score = run_greedy(0)
        if blocks is not None:
            best_solution = blocks
            best_score = score
        else:
            return []

    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
