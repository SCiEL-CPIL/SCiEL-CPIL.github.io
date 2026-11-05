import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    points = list(range(v))
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute: for each block, which t-subsets it covers
    # and for each subset, which blocks cover it
    block_to_subsets = [[] for _ in range(num_blocks)]
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            block_to_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)
    
    # Convert to numpy arrays
    block_to_subsets_arr = [np.array(s, dtype=np.int32) for s in block_to_subsets]
    subset_to_blocks_arr = [np.array(s, dtype=np.int32) for s in subset_to_blocks]
    
    # Precompute initial gains: how many t-subsets each block covers
    initial_gains = np.array([len(s) for s in block_to_subsets], dtype=np.int32)
    
    def run_greedy_full(random_seed=None, noise_scale=0.3):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = initial_gains.copy()
        uncovered_count = num_subsets
        
        selected_blocks = []
        
        while uncovered_count > 0:
            noise = rng.uniform(0, noise_scale, size=num_blocks)
            scores = gains + noise
            best_idx = np.argmax(scores)
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_to_subsets_arr[best_idx]
            newly_covered = b_subs[uncovered[b_subs]]
            
            if len(newly_covered) == 0:
                continue
            
            # Update gains: for each newly covered subset, decrement gain of all blocks covering it
            if len(newly_covered) > 0:
                indices_to_decrement = np.concatenate([subset_to_blocks_arr[s] for s in newly_covered])
                counts = np.bincount(indices_to_decrement, minlength=num_blocks)
                gains -= counts
            
            for s in newly_covered:
                uncovered[s] = False
            
            uncovered_count -= len(newly_covered)
                    
        return selected_blocks
    
    def run_greedy_partial(uncovered_mask, random_seed=None, noise_scale=0.3):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            return []
            
        gains = np.zeros(num_blocks, dtype=np.int32)
        for s_idx in uncovered_indices:
            gains[subset_to_blocks_arr[s_idx]] += 1
            
        selected_blocks = []
        current_uncovered = uncovered_mask.copy()
        uncovered_count = len(uncovered_indices)
        
        while uncovered_count > 0:
            noise = rng.uniform(0, noise_scale, size=num_blocks)
            scores = gains + noise
            best_idx = np.argmax(scores)
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_to_subsets_arr[best_idx]
            newly_covered = b_subs[current_uncovered[b_subs]]
            
            if len(newly_covered) == 0:
                continue
            
            if len(newly_covered) > 0:
                indices_to_decrement = np.concatenate([subset_to_blocks_arr[s] for s in newly_covered])
                counts = np.bincount(indices_to_decrement, minlength=num_blocks)
                gains -= counts
            
            for s in newly_covered:
                current_uncovered[s] = False
            
            uncovered_count -= len(newly_covered)
                    
        return selected_blocks
    
    def minimize(blocks_indices, random_seed=None):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        
        if not current_blocks:
            return current_blocks
            
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            cover_count[block_to_subsets_arr[b_idx]] += 1
            
        improved = True
        while improved:
            improved = False
            for i in range(len(current_blocks)):
                b_idx = current_blocks[i]
                b_subs = block_to_subsets_arr[b_idx]
                if not np.any(cover_count[b_subs] == 1):
                    # This block is redundant
                    cover_count[b_subs] -= 1
                    current_blocks.pop(i)
                    improved = True
                    break
                    
        return current_blocks
    
    def local_search_swap(blocks_indices, random_seed=None, max_iters=200):
        """Try to improve by swapping blocks: replace one block with another that covers more uncovered subsets."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        if not current_blocks:
            return current_blocks
        
        # Compute cover count
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            cover_count[block_to_subsets_arr[b_idx]] += 1
        
        uncovered = (cover_count == 0).astype(bool)
        
        for _ in range(max_iters):
            # Pick a random block to try replacing
            pos = rng.randint(0, len(current_blocks) - 1)
            old_block = current_blocks[pos]
            old_subs = block_to_subsets_arr[old_block]
            
            # Temporarily remove old block
            cover_count[old_subs] -= 1
            uncovered = (cover_count == 0).astype(bool)
            
            # Find best replacement block
            best_block = -1
            best_gain = 0
            # Sample some candidate blocks for speed
            candidates = rng.sample(range(num_blocks), min(200, num_blocks))
            for cand in candidates:
                cand_subs = block_to_subsets_arr[cand]
                gain = np.sum(uncovered[cand_subs])
                if gain > best_gain:
                    best_gain = gain
                    best_block = cand
            
            if best_block == -1 or best_gain == 0:
                # Restore old block
                cover_count[old_subs] += 1
                continue
            
            # Apply swap
            current_blocks[pos] = best_block
            cover_count[old_subs] += 1  # restore
            cover_count[block_to_subsets_arr[best_block]] += 1
            
            # Check if we made progress (fewer uncovered)
            new_uncovered = (cover_count == 0).sum()
            if new_uncovered == 0:
                break
        
        return current_blocks
    
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    trial = 0
    while time.time() < deadline:
        # Strategy 1: Full Greedy + Minimize
        greedy_res = run_greedy_full(trial, noise_scale=0.2 + 0.1 * (trial % 3))
        min_res = minimize(greedy_res, trial)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
        
        # Strategy 2: Local search swap on best
        if best_blocks is not None and len(best_blocks) > 10:
            swapped = local_search_swap(best_blocks, trial + 500, max_iters=100)
            min_res2 = minimize(swapped, trial + 600)
            
            if len(min_res2) < best_len:
                best_len = len(min_res2)
                best_blocks = min_res2
        
        # Strategy 3: Perturb best + repair + minimize
        if best_blocks is not None and len(best_blocks) > 15 and time.time() < deadline - 20:
            perturbed = list(best_blocks)
            num_remove = min(2, len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), num_remove)
                blocks_to_remove = set(perturbed[i] for i in remove_indices)
                
                cover_count_temp = np.zeros(num_subsets, dtype=np.int32)
                for b_idx in perturbed:
                    if b_idx not in blocks_to_remove:
                        cover_count_temp[block_to_subsets_arr[b_idx]] += 1
                        
                uncovered_mask = (cover_count_temp == 0).astype(bool)
                
                repair_res = run_greedy_partial(uncovered_mask, trial + 1000, noise_scale=0.2)
                
                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)
                
                min_res3 = minimize(new_set, trial + 2000)
                
                if len(min_res3) < best_len:
                    best_len = len(min_res3)
                    best_blocks = min_res3

        trial += 1
        
    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        best_blocks = minimize(greedy_res, 0)
        
    return [list(all_blocks[i]) for i in best_blocks]
