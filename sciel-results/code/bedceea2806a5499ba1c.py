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
    
    # Precompute which t-subsets each block covers
    block_to_subsets = []
    for block in all_blocks:
        subs = [subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_to_subsets.append(np.array(subs, dtype=np.int32))
    
    # For each subset, which blocks cover it
    subset_to_blocks = [[] for _ in range(num_subsets)]
    for b_idx, subs in enumerate(block_to_subsets):
        for s_idx in subs:
            subset_to_blocks[s_idx].append(b_idx)
    
    # Convert to numpy arrays
    subset_to_blocks_arr = [np.array(blocks, dtype=np.int32) for blocks in subset_to_blocks]
    subset_to_blocks_set = [set(blocks) for blocks in subset_to_blocks]
    
    # Block sizes (number of t-subsets covered)
    block_sizes = np.array([len(s) for s in block_to_subsets], dtype=np.int32)
    
    # Sort blocks by size for initial greedy ordering
    block_order = np.argsort(-block_sizes)
    
    def greedy_cover(uncovered_mask, random_seed=0):
        """Greedy set cover with random tie-breaking."""
        rng = np.random.default_rng(random_seed)
        uncovered = uncovered_mask.copy()
        gains = np.zeros(num_blocks, dtype=np.int32)
        
        # Initialize gains: for each uncovered subset, increment gain for all blocks covering it
        uncovered_indices = np.where(uncovered)[0]
        for s_idx in uncovered_indices:
            gains[subset_to_blocks_arr[s_idx]] += 1
        
        selected = []
        while np.any(uncovered):
            if np.max(gains) == 0:
                break
            
            # Add small noise for diversification
            noise = rng.uniform(0, 0.1, size=num_blocks)
            scores = gains.astype(np.float64) + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
            
            selected.append(best_idx)
            
            # Update uncovered and gains
            b_subs = block_to_subsets[best_idx]
            newly_covered = b_subs[uncovered[b_subs]]
            
            if len(newly_covered) == 0:
                continue
            
            uncovered[newly_covered] = False
            
            for s_idx in newly_covered:
                gains[subset_to_blocks_arr[s_idx]] -= 1
        
        return selected
    
    def compute_coverage(blocks_list):
        """Compute coverage count for each t-subset."""
        cover = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in blocks_list:
            cover[block_to_subsets[b_idx]] += 1
        return cover
    
    def local_search(blocks_list, cover, time_limit, random_seed=0):
        """Local search: deletion + swap with weighted scoring."""
        rng = random.Random(random_seed)
        current = list(blocks_list)
        cover = cover.copy()
        current_set = set(current)
        
        # Clause weights: subsets with low coverage are more "critical"
        weights = np.ones(num_subsets, dtype=np.float64)
        for s in range(num_subsets):
            if cover[s] == 1:
                weights[s] = 2.0
            elif cover[s] == 2:
                weights[s] = 1.5
        
        improved = True
        start = time.time()
        stall = 0
        max_stall = 25
        
        while improved and (time.time() - start) < time_limit:
            improved = False
            stall += 1
            
            if stall > max_stall:
                # Reset weights
                weights = np.ones(num_subsets, dtype=np.float64)
                for s in range(num_subsets):
                    if cover[s] == 1:
                        weights[s] = 2.0
                stall = 0
            
            # Deletion pass
            indices = list(range(len(current)))
            rng.shuffle(indices)
            
            for i in indices:
                if i >= len(current):
                    continue
                if time.time() - start > time_limit:
                    break
                
                b_idx = current[i]
                b_subs = block_to_subsets[b_idx]
                
                # Check if redundant
                redundant = True
                for s_idx in b_subs:
                    if cover[s_idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    for s_idx in b_subs:
                        cover[s_idx] -= 1
                        if cover[s_idx] == 1:
                            weights[s_idx] = 2.0
                        elif cover[s_idx] == 2:
                            weights[s_idx] = 1.5
                    current_set.discard(b_idx)
                    current.pop(i)
                    improved = True
                    stall = 0
                    break
            
            # Swap pass
            if time.time() - start > time_limit:
                break
            
            indices = list(range(len(current)))
            rng.shuffle(indices)
            swaps = 0
            max_swaps = 60
            
            for i in indices:
                if time.time() - start > time_limit:
                    break
                if swaps >= max_swaps:
                    break
                if i >= len(current):
                    continue
                
                old_b = current[i]
                old_subs = block_to_subsets[old_b]
                
                # Find subsets uniquely covered by this block
                unique_subs = [s for s in old_subs if cover[s] == 1]
                
                if unique_subs:
                    # Must find a block covering all unique subsets
                    candidates = None
                    for s_idx in unique_subs:
                        if candidates is None:
                            candidates = set(subset_to_blocks_set[s_idx])
                        else:
                            candidates &= subset_to_blocks_set[s_idx]
                            if not candidates:
                                break
                    
                    if candidates:
                        candidates.discard(old_b)
                        candidates -= current_set
                        
                        if candidates:
                            # Score by weighted coverage
                            best_score = -1
                            best_cand = None
                            for cand in list(candidates)[:50]:
                                cand_subs = block_to_subsets[cand]
                                score = sum(weights[s] for s in cand_subs)
                                if score > best_score:
                                    best_score = score
                                    best_cand = cand
                            
                            if best_cand is not None:
                                # Perform swap
                                for s_idx in old_subs:
                                    cover[s_idx] -= 1
                                    if cover[s_idx] == 1:
                                        weights[s_idx] = 2.0
                                new_subs = block_to_subsets[best_cand]
                                for s_idx in new_subs:
                                    cover[s_idx] += 1
                                    if cover[s_idx] == 2:
                                        weights[s_idx] = 1.5
                                current_set.discard(old_b)
                                current_set.add(best_cand)
                                current[i] = best_cand
                                swaps += 1
                                improved = True
                                stall = 0
                else:
                    # No unique subsets - try random swap
                    if rng.random() < 0.5:
                        new_b = rng.randint(0, num_blocks - 1)
                        if new_b in current_set or new_b == old_b:
                            continue
                        
                        # Evaluate
                        old_score = sum(weights[s] for s in old_subs)
                        new_score = sum(weights[s] for s in block_to_subsets[new_b])
                        
                        if new_score > old_score or rng.random() < 0.2:
                            for s_idx in old_subs:
                                cover[s_idx] -= 1
                                if cover[s_idx] == 1:
                                    weights[s_idx] = 2.0
                            new_subs = block_to_subsets[new_b]
                            for s_idx in new_subs:
                                cover[s_idx] += 1
                                if cover[s_idx] == 2:
                                    weights[s_idx] = 1.5
                            current_set.discard(old_b)
                            current_set.add(new_b)
                            current[i] = new_b
                            swaps += 1
                            improved = True
                            stall = 0
        
        return current, cover
    
    # Main search loop
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_blocks = None
    best_len = float('inf')
    best_cover = None
    
    trial = 0
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 20:
            break
        
        # Greedy with different random seeds
        seed = trial * 7919 + 13
        greedy_res = greedy_cover(np.ones(num_subsets, dtype=bool), random_seed=seed)
        
        if len(greedy_res) < best_len:
            best_len = len(greedy_res)
            best_blocks = greedy_res
            best_cover = compute_coverage(greedy_res)
        
        # Local search on greedy result
        ls_time = min(5.0, max(2.0, remaining * 0.15))
        ls_res, ls_cover = local_search(greedy_res, best_cover if best_blocks == greedy_res else compute_coverage(greedy_res), time_limit=ls_time, random_seed=seed + 100)
        
        if len(ls_res) < best_len:
            best_len = len(ls_res)
            best_blocks = ls_res
            best_cover = ls_cover
        
        # Perturbation + repair
        if best_blocks is not None and len(best_blocks) > 20 and time.time() < deadline - 25:
            perturbed = list(best_blocks)
            num_remove = min(max(2, len(perturbed) // (10 + trial % 6)), len(perturbed) - 2)
            if num_remove > 0 and len(perturbed) > num_remove + 1:
                remove_idx = random.sample(range(len(perturbed)), num_remove)
                remove_set = set(perturbed[i] for i in remove_idx)
                
                temp_cover = np.zeros(num_subsets, dtype=np.int32)
                for b in perturbed:
                    if b not in remove_set:
                        temp_cover[block_to_subsets[b]] += 1
                
                uncovered = (temp_cover == 0)
                repair = greedy_cover(uncovered, random_seed=seed + 200)
                
                new_set = [b for b in perturbed if b not in remove_set] + repair
                
                new_cover = compute_coverage(new_set)
                ls_time2 = min(4.0, max(1.5, (deadline - time.time()) * 0.12))
                min_res, min_cov = local_search(new_set, new_cover, time_limit=ls_time2, random_seed=seed + 300)
                
                if len(min_res) < best_len:
                    best_len = len(min_res)
                    best_blocks = min_res
                    best_cover = min_cov
        
        # Pure local search on best
        if best_blocks is not None and time.time() < deadline - 15:
            ls_time3 = min(4.0, max(1.5, (deadline - time.time()) * 0.12))
            swap_res, swap_cov = local_search(best_blocks, best_cover, time_limit=ls_time3, random_seed=seed + 400)
            if len(swap_res) < best_len:
                best_len = len(swap_res)
                best_blocks = swap_res
                best_cover = swap_cov
        
        trial += 1
    
    # Final deletion pass
    if best_blocks is not None and time.time() < deadline - 5:
        final_blocks = list(best_blocks)
        final_cover = best_cover.copy()
        final_set = set(final_blocks)
        
        for _ in range(3):
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                for i in range(len(final_blocks)):
                    b_idx = final_blocks[i]
                    b_subs = block_to_subsets[b_idx]
                    redundant = True
                    for s in b_subs:
                        if final_cover[s] == 1:
                            redundant = False
                            break
                    if redundant:
                        for s in b_subs:
                            final_cover[s] -= 1
                        final_set.discard(b_idx)
                        final_blocks.pop(i)
                        changed = True
                        break
        
        if len(final_blocks) < best_len:
            best_len = len(final_blocks)
            best_blocks = final_blocks
            best_cover = final_cover
    
    if best_blocks is None:
        greedy_res = greedy_cover(np.ones(num_subsets, dtype=bool), random_seed=0)
        best_blocks = greedy_res
        best_cover = compute_coverage(greedy_res)
    
    return [list(all_blocks[i]) for i in best_blocks]
