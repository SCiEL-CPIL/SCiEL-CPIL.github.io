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
    block_to_subsets = [[] for _ in range(num_blocks)]
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            block_to_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)
    
    # Precompute numpy arrays for fast operations
    block_subset_arrays = [np.array(subs, dtype=np.int32) for subs in block_to_subsets]
    subset_block_arrays = [np.array(blocks, dtype=np.int32) for blocks in subset_to_blocks]
    
    # Precompute subset_to_blocks as sets for fast intersection in swap
    subset_to_blocks_sets = [set(blocks) for blocks in subset_to_blocks]
    
    # Precompute number of blocks covering each subset (for hardest-subset greedy)
    subset_block_counts = np.array([len(b) for b in subset_to_blocks], dtype=np.int32)

    def run_greedy_full(random_seed=None, hardest_first=False):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
        
        selected_blocks = []
        
        while True:
            if not np.any(uncovered):
                break
                
            if hardest_first:
                # Prioritize blocks that cover the hardest uncovered subsets
                # Weight each block by the sum of (1/count) for its uncovered subsets
                # This is expensive, so we approximate: pick block covering most uncovered,
                # with tie-breaking by covering hardest subsets
                noise = rng.uniform(0, 0.1, size=num_blocks)
                scores = gains.astype(np.float64) + noise
                # Among top candidates, prefer those covering low-count subsets
                best_idx = int(np.argmax(scores))
            else:
                noise = rng.uniform(0, 0.3, size=num_blocks)
                scores = gains.astype(np.float64) + noise
                best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_subset_arrays[best_idx]
            newly_covered = b_subs[uncovered[b_subs]]
            
            if len(newly_covered) == 0:
                continue
                
            uncovered[newly_covered] = False
            for s in newly_covered:
                blocks_covering = subset_block_arrays[s]
                gains[blocks_covering] -= 1
                    
        return selected_blocks

    def run_greedy_partial(uncovered_mask, random_seed=None, hardest_first=False):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            return []
            
        gains = np.zeros(num_blocks, dtype=np.int32)
        for s_idx in uncovered_indices:
            blocks_covering = subset_block_arrays[s_idx]
            gains[blocks_covering] += 1
                
        selected_blocks = []
        current_uncovered = uncovered_mask.copy()
        
        while True:
            if not np.any(current_uncovered):
                break
                
            noise = rng.uniform(0, 0.3, size=num_blocks)
            scores = gains.astype(np.float64) + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_subset_arrays[best_idx]
            newly_covered = b_subs[current_uncovered[b_subs]]
            
            if len(newly_covered) == 0:
                continue
                
            current_uncovered[newly_covered] = False
            for s in newly_covered:
                blocks_covering = subset_block_arrays[s]
                gains[blocks_covering] -= 1
                    
        return selected_blocks

    def minimize_with_swaps(blocks_indices, coverage_counts, time_limit, random_seed=None):
        """Local search with batched deletion and swap operations."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        
        if not current_blocks:
            return current_blocks, coverage_counts.copy()
            
        cover_count = coverage_counts.copy()
        current_blocks_set = set(current_blocks)
        
        improved = True
        start_time = time.time()
        
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            
            # Batched deletion pass: find ALL redundant blocks in one scan, remove them all
            if (time.time() - start_time) > time_limit:
                break
            
            redundant_blocks = []
            for i in range(len(current_blocks)):
                if (time.time() - start_time) > time_limit:
                    break
                b_idx = current_blocks[i]
                b_subs = block_subset_arrays[b_idx]
                is_redundant = True
                for s_idx in b_subs:
                    if cover_count[s_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    redundant_blocks.append(i)
            
            if redundant_blocks:
                # Remove all redundant blocks at once (process in reverse to preserve indices)
                for i in reversed(redundant_blocks):
                    b_idx = current_blocks[i]
                    b_subs = block_subset_arrays[b_idx]
                    for s_idx in b_subs:
                        cover_count[s_idx] -= 1
                    current_blocks_set.discard(b_idx)
                    current_blocks.pop(i)
                improved = True
                continue  # Restart scan after batch removal
            
            # Swap pass
            if (time.time() - start_time) > time_limit:
                break
                
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            
            swaps_done = 0
            max_swaps = 100
            
            for i in indices:
                if (time.time() - start_time) > time_limit:
                    break
                if swaps_done >= max_swaps:
                    break
                if i >= len(current_blocks):
                    continue
                    
                old_block = current_blocks[i]
                old_subs = block_subset_arrays[old_block]
                
                # Find unique subsets covered by old_block
                unique_subs = []
                for s_idx in old_subs:
                    if cover_count[s_idx] == 1:
                        unique_subs.append(s_idx)
                
                if not unique_subs:
                    # Block is not critical - try swapping with a random block
                    new_block = rng.randint(0, num_blocks - 1)
                    if new_block == old_block:
                        continue
                    if new_block in current_blocks_set:
                        continue
                        
                    # Perform swap
                    for s_idx in old_subs:
                        cover_count[s_idx] -= 1
                    new_subs = block_subset_arrays[new_block]
                    for s_idx in new_subs:
                        cover_count[s_idx] += 1
                    current_blocks_set.discard(old_block)
                    current_blocks_set.add(new_block)
                    current_blocks[i] = new_block
                    swaps_done += 1
                else:
                    # Block is critical - find candidates that cover all unique_subs
                    candidates = None
                    for s_idx in unique_subs:
                        if candidates is None:
                            candidates = subset_to_blocks_sets[s_idx].copy()
                        else:
                            candidates.intersection_update(subset_to_blocks_sets[s_idx])
                            if not candidates:
                                break
                    
                    if candidates:
                        candidates.discard(old_block)
                        if candidates:
                            # Pick a random candidate
                            new_block = rng.choice(tuple(candidates))
                            if new_block in current_blocks_set:
                                continue
                                
                            # Perform swap
                            for s_idx in old_subs:
                                cover_count[s_idx] -= 1
                            new_subs = block_subset_arrays[new_block]
                            for s_idx in new_subs:
                                cover_count[s_idx] += 1
                            current_blocks_set.discard(old_block)
                            current_blocks_set.add(new_block)
                            current_blocks[i] = new_block
                            swaps_done += 1
                            improved = True
            
            # 2-for-1 swap: try to replace two blocks with one
            if (time.time() - start_time) > time_limit:
                break
            if len(current_blocks) >= 2:
                num_2for1_tries = 20
                for _ in range(num_2for1_tries):
                    if (time.time() - start_time) > time_limit:
                        break
                    i1, i2 = rng.sample(range(len(current_blocks)), 2)
                    b1 = current_blocks[i1]
                    b2 = current_blocks[i2]
                    
                    # Find unique subsets of b1 (not covered by b2 or others)
                    # Simplified: check if b1 has any unique subset not in b2
                    unique_b1 = []
                    for s_idx in block_subset_arrays[b1]:
                        if cover_count[s_idx] == 1:
                            unique_b1.append(s_idx)
                    
                    if not unique_b1:
                        continue
                    
                    # Check if b2 covers all unique subsets of b1
                    b2_subs_set = set(block_subset_arrays[b2].tolist())
                    if not all(s in b2_subs_set for s in unique_b1):
                        continue
                    
                    # Check if b1 covers all unique subsets of b2
                    unique_b2 = []
                    for s_idx in block_subset_arrays[b2]:
                        if cover_count[s_idx] == 1:
                            unique_b2.append(s_idx)
                    
                    if unique_b2:
                        b1_subs_set = set(block_subset_arrays[b1].tolist())
                        if not all(s in b1_subs_set for s in unique_b2):
                            continue
                    
                    # Find a single block that covers all unique subsets of both
                    all_unique = set(unique_b1) | set(unique_b2)
                    if not all_unique:
                        # Both blocks are fully redundant relative to each other - just remove one
                        for s_idx in block_subset_arrays[b2]:
                            cover_count[s_idx] -= 1
                        current_blocks_set.discard(b2)
                        current_blocks.pop(i2)
                        improved = True
                        break
                    
                    candidates = None
                    for s_idx in all_unique:
                        if candidates is None:
                            candidates = subset_to_blocks_sets[s_idx].copy()
                        else:
                            candidates.intersection_update(subset_to_blocks_sets[s_idx])
                            if not candidates:
                                break
                    
                    if candidates:
                        candidates.discard(b1)
                        candidates.discard(b2)
                        if candidates:
                            new_block = rng.choice(tuple(candidates))
                            if new_block in current_blocks_set:
                                continue
                            
                            # Remove both old blocks, add new one
                            for s_idx in block_subset_arrays[b1]:
                                cover_count[s_idx] -= 1
                            for s_idx in block_subset_arrays[b2]:
                                cover_count[s_idx] -= 1
                            current_blocks_set.discard(b1)
                            current_blocks_set.discard(b2)
                            
                            new_subs = block_subset_arrays[new_block]
                            for s_idx in new_subs:
                                cover_count[s_idx] += 1
                            current_blocks_set.add(new_block)
                            
                            # Remove from list (process higher index first)
                            if i1 > i2:
                                i1, i2 = i2, i1
                            current_blocks.pop(i2)
                            current_blocks.pop(i1)
                            current_blocks.append(new_block)
                            improved = True
                            break

        return current_blocks, cover_count

    def compute_coverage(blocks_indices):
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in blocks_indices:
            b_subs = block_subset_arrays[b_idx]
            cover_count[b_subs] += 1
        return cover_count

    def lll_random_construction(random_seed=None, p=None):
        """LLL-inspired probabilistic construction."""
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
        
        if p is None:
            p = 0.15
        
        mask = rng.random(num_blocks) < p
        selected = np.where(mask)[0].tolist()
        
        if not selected:
            return []
        
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in selected:
            b_subs = block_subset_arrays[b_idx]
            cover_count[b_subs] += 1
        
        if np.any(cover_count == 0):
            uncovered_mask = (cover_count == 0).astype(bool)
            repair_res = run_greedy_partial(uncovered_mask, random_seed=random_seed)
            selected.extend(repair_res)
            cover_count = np.zeros(num_subsets, dtype=np.int32)
            for b_idx in selected:
                b_subs = block_subset_arrays[b_idx]
                cover_count[b_subs] += 1
        
        # Batched redundancy removal
        changed = True
        start_time = time.time()
        while changed and (time.time() - start_time) < 2.0:
            changed = False
            rng.shuffle(selected)
            redundant = []
            for b_idx in selected:
                b_subs = block_subset_arrays[b_idx]
                is_redundant = True
                for s_idx in b_subs:
                    if cover_count[s_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    redundant.append(b_idx)
            
            if redundant:
                redundant_set = set(redundant)
                for b_idx in redundant:
                    b_subs = block_subset_arrays[b_idx]
                    for s_idx in b_subs:
                        cover_count[s_idx] -= 1
                selected = [b for b in selected if b not in redundant_set]
                changed = True
        
        return selected

    best_blocks = None
    best_len = float('inf')
    best_coverage = None
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    trial = 0
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 15.0:
            break
            
        # Strategy 1: Full Greedy + Local Search
        use_hardest = (trial % 3 == 0)
        greedy_res = run_greedy_full(trial, hardest_first=use_hardest)
        coverage = compute_coverage(greedy_res)
        
        ls_time = min(4.0, max(1.5, remaining * 0.18))
        min_res, min_cov = minimize_with_swaps(greedy_res, coverage, time_limit=ls_time, random_seed=trial)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
            best_coverage = min_cov
            
        # Strategy 2: LLL-inspired probabilistic construction + Local Search
        if time.time() < deadline - 15:
            p_vals = [0.10, 0.15, 0.20, 0.25, 0.30]
            p = p_vals[trial % len(p_vals)]
            lll_res = lll_random_construction(random_seed=trial + 10000, p=p)
            
            if lll_res:
                lll_coverage = compute_coverage(lll_res)
                
                ls_time_lla = min(3.0, max(1.0, (deadline - time.time()) * 0.15))
                min_res_lla, min_cov_lla = minimize_with_swaps(lll_res, lll_coverage, time_limit=ls_time_lla, random_seed=trial + 20000)
                
                if len(min_res_lla) < best_len:
                    best_len = len(min_res_lla)
                    best_blocks = min_res_lla
                    best_coverage = min_cov_lla

        # Strategy 3: Perturb Best + Repair + Local Search
        if best_blocks is not None and len(best_blocks) > 20 and time.time() < deadline - 15:
            perturbed = list(best_blocks)
            num_remove = min(max(2, len(perturbed) // (10 + trial % 10)), len(perturbed) - 1)
            if num_remove > 0:
                cover_count_temp = best_coverage.copy()
                unique_counts = []
                for b_idx in perturbed:
                    b_subs = block_subset_arrays[b_idx]
                    unique = sum(1 for s_idx in b_subs if cover_count_temp[s_idx] == 1)
                    unique_counts.append(unique)
                
                sorted_indices = sorted(range(len(perturbed)), key=lambda i: unique_counts[i])
                candidates = sorted_indices[:max(num_remove * 2, 4)]
                remove_indices = random.sample(candidates, min(num_remove, len(candidates)))
                blocks_to_remove = set(perturbed[i] for i in remove_indices)
                
                cover_count_temp2 = np.zeros(num_subsets, dtype=np.int32)
                for b_idx in perturbed:
                    if b_idx in blocks_to_remove:
                        continue
                    b_subs = block_subset_arrays[b_idx]
                    cover_count_temp2[b_subs] += 1
                        
                uncovered_mask = (cover_count_temp2 == 0).astype(bool)
                
                repair_res = run_greedy_partial(uncovered_mask, trial + 30000)
                
                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)
                
                new_coverage = compute_coverage(new_set)
                
                ls_time2 = min(3.0, max(1.0, (deadline - time.time()) * 0.15))
                min_res2, min_cov2 = minimize_with_swaps(new_set, new_coverage, time_limit=ls_time2, random_seed=trial + 40000)
                
                if len(min_res2) < best_len:
                    best_len = len(min_res2)
                    best_blocks = min_res2
                    best_coverage = min_cov2

        # Strategy 4: Pure swap-based diversification on best solution
        if best_blocks is not None and len(best_blocks) > 20 and time.time() < deadline - 10:
            ls_time3 = min(5.0, max(1.5, (deadline - time.time()) * 0.2))
            swap_res, swap_cov = minimize_with_swaps(best_blocks, best_coverage, time_limit=ls_time3, random_seed=trial + 50000)
            
            if len(swap_res) < best_len:
                best_len = len(swap_res)
                best_blocks = swap_res
                best_coverage = swap_cov

        trial += 1
        
    # Final polish: aggressive batched deletion pass on best solution
    if best_blocks is not None and time.time() < deadline - 5:
        final_cov = best_coverage.copy()
        final_blocks = list(best_blocks)
        final_set = set(final_blocks)
        for _ in range(5):
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                redundant = []
                for i in range(len(final_blocks)):
                    b_idx = final_blocks[i]
                    b_subs = block_subset_arrays[b_idx]
                    redundant_flag = True
                    for s_idx in b_subs:
                        if final_cov[s_idx] == 1:
                            redundant_flag = False
                            break
                    if redundant_flag:
                        redundant.append(i)
                
                if redundant:
                    for i in reversed(redundant):
                        b_idx = final_blocks[i]
                        b_subs = block_subset_arrays[b_idx]
                        for s_idx in b_subs:
                            final_cov[s_idx] -= 1
                        final_set.discard(b_idx)
                        final_blocks.pop(i)
                    changed = True
        if len(final_blocks) < best_len:
            best_len = len(final_blocks)
            best_blocks = final_blocks
            best_coverage = final_cov
        
    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        coverage = compute_coverage(greedy_res)
        best_blocks, best_coverage = minimize_with_swaps(greedy_res, coverage, time_limit=5.0, random_seed=0)
        
    return [list(all_blocks[i]) for i in best_blocks]
