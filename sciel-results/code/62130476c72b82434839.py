import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin
    
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block_to_subsets and subset_to_blocks
    block_to_subsets = [[] for _ in range(num_blocks)]
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            block_to_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)
            
    # Convert to numpy arrays for faster access
    block_to_subsets_arr = [np.array(s, dtype=np.int32) for s in block_to_subsets]
    subset_to_blocks_arr = [np.array(s, dtype=np.int32) for s in subset_to_blocks]
    
    # Precompute subset membership as a set for fast lookup in swap checks
    # Only for frequently used blocks - use a lazy cache approach
    _subset_set_cache = {}
    def get_subset_set(b_idx):
        if b_idx not in _subset_set_cache:
            _subset_set_cache[b_idx] = set(block_to_subsets_arr[b_idx].tolist())
        return _subset_set_cache[b_idx]
    
    def run_greedy_full(random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
        uncovered_count = num_subsets
        
        selected_blocks = []
        
        while uncovered_count > 0:
            # Use a faster selection: pick among top candidates
            # First find blocks with max gain, then add small noise
            max_gain = gains.max()
            if max_gain == 0:
                break
            # Get all blocks with gain close to max (within 2)
            candidates = np.where(gains >= max_gain - 2)[0]
            if len(candidates) == 0:
                break
            # Add noise only among candidates
            noise = rng.uniform(0, 0.5, size=len(candidates))
            best_local = noise.argmax()
            best_idx = candidates[best_local]
            
            selected_blocks.append(best_idx)
            
            b_subs = block_to_subsets_arr[best_idx]
            newly_covered = b_subs[uncovered[b_subs]]
            
            if len(newly_covered) == 0:
                continue
                
            # Vectorized update
            uncovered[b_subs] = False
            for s in newly_covered:
                gains[subset_to_blocks_arr[s]] -= 1
            
            uncovered_count -= len(newly_covered)
                    
        return selected_blocks

    def run_greedy_partial(uncovered_mask, random_seed=None):
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
            max_gain = gains.max()
            if max_gain == 0:
                break
            candidates = np.where(gains >= max_gain - 2)[0]
            if len(candidates) == 0:
                break
            noise = rng.uniform(0, 0.5, size=len(candidates))
            best_local = noise.argmax()
            best_idx = candidates[best_local]
            
            selected_blocks.append(best_idx)
            
            b_subs = block_to_subsets_arr[best_idx]
            newly_covered = b_subs[current_uncovered[b_subs]]
            
            if len(newly_covered) == 0:
                continue
                
            current_uncovered[b_subs] = False
            for s in newly_covered:
                gains[subset_to_blocks_arr[s]] -= 1
            
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
            
        while True:
            redundant_blocks = []
            
            for i, b_idx in enumerate(current_blocks):
                b_subs = block_to_subsets_arr[b_idx]
                if not np.any(cover_count[b_subs] == 1):
                    redundant_blocks.append(i)
                    
            if not redundant_blocks:
                break
                
            remove_pos = redundant_blocks[rng.randint(0, len(redundant_blocks)-1)]
            b_idx_to_remove = current_blocks[remove_pos]
            
            cover_count[block_to_subsets_arr[b_idx_to_remove]] -= 1
            current_blocks.pop(remove_pos)
            
        return current_blocks

    def try_swap(current_set, old_block, new_block, coverage_counts):
        """Try to swap old_block with new_block in current_set, maintaining validity.
        Returns True if swap was successful, False otherwise."""
        if old_block == new_block:
            return False
        
        # Find unique subsets covered by old_block
        b_subs = block_to_subsets_arr[old_block]
        unique_subs = b_subs[coverage_counts[b_subs] == 1]
        
        if len(unique_subs) == 0:
            # Block is not critical, can swap freely
            coverage_counts[b_subs] -= 1
            new_subs = block_to_subsets_arr[new_block]
            coverage_counts[new_subs] += 1
            # Replace in list
            idx = current_set.index(old_block)
            current_set[idx] = new_block
            return True
        
        # Check if new_block covers all unique subsets using cached set
        new_subs = block_to_subsets_arr[new_block]
        new_subs_set = get_subset_set(new_block)
        
        for s in unique_subs:
            if s not in new_subs_set:
                return False
        
        # Swap is valid
        coverage_counts[b_subs] -= 1
        coverage_counts[new_subs] += 1
        idx = current_set.index(old_block)
        current_set[idx] = new_block
        return True

    def minimize_with_swap(blocks_indices, random_seed=None, time_limit=5.0):
        """Minimize by deletion and swapping."""
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
            
        start = time.time()
        current_set_set = set(current_blocks)
        
        # First do deletion pass
        while True:
            redundant_blocks = []
            
            for i, b_idx in enumerate(current_blocks):
                b_subs = block_to_subsets_arr[b_idx]
                if not np.any(cover_count[b_subs] == 1):
                    redundant_blocks.append(i)
                    
            if not redundant_blocks:
                break
                
            if time.time() - start > time_limit:
                break
                
            remove_pos = redundant_blocks[rng.randint(0, len(redundant_blocks)-1)]
            b_idx_to_remove = current_blocks[remove_pos]
            
            cover_count[block_to_subsets_arr[b_idx_to_remove]] -= 1
            current_blocks.pop(remove_pos)
            current_set_set.discard(b_idx_to_remove)
            
        # Then do swap pass
        if time.time() - start < time_limit:
            max_swaps = 30
            swaps_done = 0
            
            while swaps_done < max_swaps and time.time() - start < time_limit:
                # Pick a random block to try swapping
                if len(current_blocks) == 0:
                    break
                    
                old_idx = rng.randint(0, len(current_blocks) - 1)
                old_block = current_blocks[old_idx]
                
                # Try a few random new blocks
                for _ in range(8):
                    new_block = rng.randint(0, num_blocks - 1)
                    
                    if new_block in current_set_set:
                        continue
                        
                    if try_swap(current_blocks, old_block, new_block, cover_count):
                        swaps_done += 1
                        current_set_set.discard(old_block)
                        current_set_set.add(new_block)
                        # After swap, try deletion again
                        redundant_blocks = []
                        for i, b_idx in enumerate(current_blocks):
                            b_subs = block_to_subsets_arr[b_idx]
                            if not np.any(cover_count[b_subs] == 1):
                                redundant_blocks.append(i)
                        
                        if redundant_blocks:
                            remove_pos = redundant_blocks[rng.randint(0, len(redundant_blocks)-1)]
                            b_idx_to_remove = current_blocks[remove_pos]
                            cover_count[block_to_subsets_arr[b_idx_to_remove]] -= 1
                            current_blocks.pop(remove_pos)
                            current_set_set.discard(b_idx_to_remove)
                        break
        
        return current_blocks

    best_blocks = None
    best_len = float('inf')
    
    trial = 0
    while time.time() < deadline:
        # Strategy 1: Full Greedy + Minimize with Swap
        greedy_res = run_greedy_full(trial)
        min_res = minimize_with_swap(greedy_res, trial, time_limit=4.0)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
            
        # Strategy 2: Perturb Best + Repair + Minimize with Swap
        if best_blocks is not None and len(best_blocks) > 10:
            perturbed = list(best_blocks)
            # Remove 3-5 random blocks for more aggressive perturbation
            num_remove = min(5, len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), num_remove)
                blocks_to_remove = set(perturbed[i] for i in remove_indices)
                
                # Build uncovered mask
                cover_count_temp = np.zeros(num_subsets, dtype=np.int32)
                for b_idx in perturbed:
                    if b_idx not in blocks_to_remove:
                        cover_count_temp[block_to_subsets_arr[b_idx]] += 1
                        
                uncovered_mask = (cover_count_temp == 0).astype(bool)
                
                # Repair
                repair_res = run_greedy_partial(uncovered_mask, trial + 1000)
                
                # Combine
                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)
                
                # Minimize with swap
                min_res2 = minimize_with_swap(new_set, trial + 2000, time_limit=3.0)
                
                if len(min_res2) < best_len:
                    best_len = len(min_res2)
                    best_blocks = min_res2

        trial += 1
        
        # Check time
        if time.time() >= deadline:
            break
        
    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        best_blocks = minimize_with_swap(greedy_res, 0, time_limit=5.0)
        
    return [list(all_blocks[i]) for i in best_blocks]
