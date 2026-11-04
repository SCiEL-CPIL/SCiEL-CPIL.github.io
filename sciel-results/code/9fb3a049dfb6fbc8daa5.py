import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block_to_subsets and subset_to_blocks
    # block_to_subsets[b_idx] is a list of subset indices
    # subset_to_blocks[s_idx] is a list of block indices
    block_to_subsets = [[] for _ in range(num_blocks)]
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            block_to_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)
            
    # Convert to numpy arrays for vectorized operations where possible
    # But for sparse updates, lists of indices are often faster in Python loops if we are careful
    # Let's use numpy arrays for the subset lists of each block to allow boolean masking
    block_to_subsets_arr = [np.array(s, dtype=np.int32) for s in block_to_subsets]
    subset_to_blocks_arr = [np.array(s, dtype=np.int32) for s in subset_to_blocks]

    def get_gains(cover_count, uncovered_mask=None):
        """
        Calculate the gain (number of newly covered subsets) for each block.
        If uncovered_mask is provided, it assumes cover_count is for the current state.
        Gain = number of subsets in block where cover_count == 0 (if uncovered_mask is None, we assume we are building from scratch or similar).
        Actually, for repair, we want blocks that cover subsets where cover_count == 0.
        """
        gains = np.zeros(num_blocks, dtype=np.int32)
        # This loop is expensive. 
        # Alternative: 
        # For each uncovered subset, increment gain for all blocks covering it.
        # This is O(num_uncovered * avg_blocks_per_subset).
        if uncovered_mask is None:
             # Standard greedy: gain is just total subsets in block (since all are uncovered initially)
             return np.array([len(s) for s in block_to_subsets], dtype=np.int32)
        
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            return gains
            
        # Vectorized update is hard here because subset_to_blocks are sparse lists.
        # We will iterate over uncovered subsets.
        for s_idx in uncovered_indices:
            blocks_covering = subset_to_blocks_arr[s_idx]
            gains[blocks_covering] += 1
            
        return gains

    def run_greedy(cover_count, random_seed=None, noise_scale=0.0):
        """
        Greedy repair.
        cover_count: current coverage counts.
        Returns list of new block indices added.
        """
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered_mask = (cover_count == 0).astype(bool)
        if not np.any(uncovered_mask):
            return []
            
        # Calculate initial gains
        gains = get_gains(cover_count, uncovered_mask)
        
        selected_blocks = []
        uncovered_count = np.sum(uncovered_mask)
        
        while uncovered_count > 0:
            if noise_scale > 0:
                noise = rng.uniform(0, noise_scale, size=num_blocks)
                scores = gains + noise
            else:
                scores = gains
                
            best_idx = np.argmax(scores)
            
            if gains[best_idx] == 0:
                # Should not happen if uncovered_count > 0 and we have valid blocks
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_to_subsets_arr[best_idx]
            # Find which of these subsets are currently uncovered
            newly_covered = b_subs[uncovered_mask[b_subs]]
            
            if len(newly_covered) == 0:
                continue
                
            # Update cover_count and uncovered_mask
            cover_count[newly_covered] += 1
            uncovered_mask[newly_covered] = False
            
            # Update gains for all blocks that cover these newly covered subsets
            # A block's gain decreases by 1 for each newly covered subset it covers
            for s in newly_covered:
                blocks_covering_s = subset_to_blocks_arr[s]
                gains[blocks_covering_s] -= 1
                
            uncovered_count -= len(newly_covered)
            
        return selected_blocks

    def minimize(blocks_indices, random_seed=None):
        """
        Remove redundant blocks.
        """
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
            
        changed = True
        while changed:
            changed = False
            redundant_blocks = []
            
            for i, b_idx in enumerate(current_blocks):
                b_subs = block_to_subsets_arr[b_idx]
                # If any subset in this block is covered ONLY by this block, it's not redundant
                if not np.any(cover_count[b_subs] == 1):
                    redundant_blocks.append(i)
                    
            if redundant_blocks:
                # Remove a random redundant block to break symmetry / explore
                remove_pos = redundant_blocks[rng.randint(0, len(redundant_blocks)-1)]
                b_idx_to_remove = current_blocks[remove_pos]
                
                cover_count[block_to_subsets_arr[b_idx_to_remove]] -= 1
                current_blocks.pop(remove_pos)
                changed = True
                
        return current_blocks

    def perturb_and_repair(best_blocks, num_remove, random_seed):
        """
        Remove num_remove random blocks, repair, minimize.
        """
        if len(best_blocks) <= num_remove:
            return None
            
        rng = random.Random(random_seed)
        
        # Select blocks to remove
        remove_indices = rng.sample(range(len(best_blocks)), num_remove)
        blocks_to_remove_set = set(best_blocks[i] for i in remove_indices)
        
        # Create new list without removed blocks
        remaining_blocks = [b for b in best_blocks if b not in blocks_to_remove_set]
        
        # Calculate cover count for remaining
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in remaining_blocks:
            cover_count[block_to_subsets_arr[b_idx]] += 1
            
        # Repair
        repair_res = run_greedy(cover_count, random_seed=random_seed+1, noise_scale=0.0)
        
        # Combine
        new_blocks = remaining_blocks + repair_res
        
        # Minimize
        min_res = minimize(new_blocks, random_seed=random_seed+2)
        
        return min_res

    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    trial = 0
    
    # Phase 1: Initial Greedy Searches
    while time.time() < deadline - 30.0: # Leave 30s for local search
        # Run greedy with some noise to find diverse starting points
        noise = 0.1 + (trial % 5) * 0.1
        # We need to simulate greedy from scratch
        # run_greedy expects cover_count. We start with zeros.
        cover_count_zero = np.zeros(num_subsets, dtype=np.int32)
        greedy_res = run_greedy(cover_count_zero, random_seed=trial, noise_scale=noise)
        
        min_res = minimize(greedy_res, random_seed=trial)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
            
        trial += 1
        
    # Phase 2: Local Search (Perturbation)
    if best_blocks is None:
        # Fallback if no time left for phase 1 (unlikely)
        cover_count_zero = np.zeros(num_subsets, dtype=np.int32)
        best_blocks = minimize(run_greedy(cover_count_zero, random_seed=0, noise_scale=0.0), random_seed=0)
        best_len = len(best_blocks)

    while time.time() < deadline:
        # Try different perturbation sizes
        # Smaller perturbations are more likely to succeed but explore less
        num_remove = [1, 2, 3, 5, 8][trial % 5]
        
        new_candidate = perturb_and_repair(best_blocks, num_remove, trial + 10000)
        
        if new_candidate is not None and len(new_candidate) < best_len:
            best_len = len(new_candidate)
            best_blocks = new_candidate
            
        trial += 1

    return [list(all_blocks[i]) for i in best_blocks]
