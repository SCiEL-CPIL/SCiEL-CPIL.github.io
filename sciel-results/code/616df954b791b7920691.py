import itertools
import time
import random
import numpy as np
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115

    # Precomputation
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubsets)
    tsub_to_idx = {tsub: i for i, tsub in enumerate(all_tsubsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Precompute block masks as frozensets of t-subset indices
    block_tsubs = [None] * num_blocks
    for i, block in enumerate(all_blocks):
        subs = frozenset(tsub_to_idx[tsub] for tsub in itertools.combinations(block, t))
        block_tsubs[i] = subs

    def get_mask(block):
        return block_tsubs[block_to_idx[block]]

    def greedy_construct():
        """Greedy set cover with randomized restarts."""
        covered_mask = np.zeros(num_tsubs, dtype=bool)
        current_blocks = []
        
        while not covered_mask.all():
            uncovered_indices = np.where(~covered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            
            target_tsub = uncovered_indices[random.randint(0, len(uncovered_indices)-1)]
            
            best_block = None
            best_gain = -1
            
            sample_size = min(100, num_blocks)
            sampled_indices = random.sample(range(num_blocks), sample_size)
            
            for b_idx in sampled_indices:
                mask = block_tsubs[b_idx]
                if target_tsub not in mask:
                    continue
                gain = 0
                for s in mask:
                    if not covered_mask[s]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block is None:
                for b_idx in range(num_blocks):
                    if target_tsub in block_tsubs[b_idx]:
                        best_block = b_idx
                        break
            
            if best_block is not None:
                current_blocks.append(all_blocks[best_block])
                for s in block_tsubs[best_block]:
                    covered_mask[s] = True
            else:
                break
        
        return current_blocks

    def local_search(initial_blocks):
        """Local search to minimize block count while maintaining coverage."""
        current_blocks = list(set(initial_blocks))
        N = len(current_blocks)
        
        cover_count = np.zeros(num_tsubs, dtype=int)
        block_masks = []
        
        for b in current_blocks:
            mask = get_mask(b)
            block_masks.append(mask)
            cover_count[list(mask)] += 1
        
        current_uncovered = np.sum(cover_count == 0)
        
        best_blocks = current_blocks[:]
        best_uncovered = current_uncovered
        
        if current_uncovered == 0:
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                for i in range(len(current_blocks) - 1, -1, -1):
                    if time.time() > deadline - 5:
                        break
                    mask = block_masks[i]
                    if np.any(cover_count[list(mask)] == 1):
                        continue
                    cover_count[list(mask)] -= 1
                    del current_blocks[i]
                    del block_masks[i]
                    changed = True
                    best_blocks = current_blocks[:]
                    best_uncovered = 0
                    N -= 1
        
        T = 5.0
        cooling = 0.999
        steps = 0
        max_steps = 5000
        
        while T > 0.5 and steps < max_steps and time.time() < deadline - 5:
            i = random.randint(0, N - 1)
            old_block = current_blocks[i]
            old_mask = block_masks[i]
            
            new_block = random.choice(all_blocks)
            
            if new_block in current_blocks:
                steps += 1
                continue
            
            new_mask = get_mask(new_block)
            
            old_set = set(old_mask)
            new_set = set(new_mask)
            
            only_old = old_set - new_set
            only_new = new_set - old_set
            
            delta_uncovered = 0
            
            for idx in only_old:
                if cover_count[idx] == 1:
                    delta_uncovered += 1
            
            for idx in only_new:
                if cover_count[idx] == 0:
                    delta_uncovered -= 1
            
            if delta_uncovered <= 0 or random.random() < np.exp(-delta_uncovered / T):
                for idx in only_old:
                    cover_count[idx] -= 1
                for idx in only_new:
                    cover_count[idx] += 1
                
                current_blocks[i] = new_block
                block_masks[i] = new_mask
                current_uncovered += delta_uncovered
                
                if random.random() < 0.1:
                    for j in range(len(current_blocks) - 1, -1, -1):
                        mask_j = block_masks[j]
                        if np.any(cover_count[list(mask_j)] == 1):
                            continue
                        cover_count[list(mask_j)] -= 1
                        del current_blocks[j]
                        del block_masks[j]
                        N -= 1
                        current_uncovered = np.sum(cover_count == 0)
                        break
                
                if current_uncovered == 0:
                    if N < len(best_blocks):
                        best_blocks = current_blocks[:]
                        best_uncovered = 0
                
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_blocks = current_blocks[:]
            
            steps += 1
            T *= cooling
        
        return best_blocks

    best_valid_blocks = None
    best_valid_N = float('inf')
    
    greedy_time = 10.0
    
    for i in range(5):
        if time.time() > start_time + greedy_time:
            break
        blocks = greedy_construct()
        N = len(blocks)
        if N < best_valid_N:
            best_valid_N = N
            best_valid_blocks = blocks
    
    if best_valid_blocks is None:
        return []
    
    while time.time() < deadline - 2:
        improved = local_search(best_valid_blocks)
        
        if len(improved) < len(best_valid_blocks):
            best_valid_blocks = improved
            best_valid_N = len(improved)
        
        if time.time() < deadline - 10:
            kick_blocks = best_valid_blocks[:]
            if len(kick_blocks) > 1:
                idx_to_remove = random.randint(0, len(kick_blocks)-1)
                del kick_blocks[idx_to_remove]
            
            repaired = local_search(kick_blocks)
            
            cover_count_check = np.zeros(num_tsubs, dtype=int)
            for b in repaired:
                cover_count_check[list(get_mask(b))] += 1
            
            if np.all(cover_count_check > 0):
                if len(repaired) < len(best_valid_blocks):
                    best_valid_blocks = repaired
                    best_valid_N = len(repaired)
    
    return [list(b) for b in best_valid_blocks]
