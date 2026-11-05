import numpy as np
import itertools
import time
import random
from math import comb

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
            
    # Convert to numpy arrays for speed
    block_subset_arrays = [np.array(subs, dtype=np.int32) for subs in block_to_subsets]
    subset_block_arrays = [np.array(blocks, dtype=np.int32) for blocks in subset_to_blocks]
    
    # Precompute sets for subset_to_blocks for fast intersection
    subset_to_blocks_sets = [set(blocks) for blocks in subset_to_blocks]
    
    def compute_coverage(blocks_indices):
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in blocks_indices:
            cover_count[block_subset_arrays[b_idx]] += 1
        return cover_count

    def run_greedy_full(random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
        selected_blocks = []
        
        while np.any(uncovered):
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            # Add noise to break ties and encourage exploration
            noise_scale = 0.5
            noise = rng.uniform(0.0, noise_scale, size=num_blocks)
            scores = gains.astype(np.float64) + noise
            best_idx = int(np.argmax(scores))
            
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

    def run_greedy_hardest(random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered = np.ones(num_subsets, dtype=bool)
        selected_set = set()
        selected_blocks = []
        
        while np.any(uncovered):
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            
            # Find hardest subset (lowest degree among non-selected blocks)
            # Sample to save time
            sample_size = min(20, len(uncovered_indices))
            candidates = uncovered_indices[rng.choice(len(uncovered_indices), size=sample_size, replace=False)]
            
            best_subset = -1
            best_degree = float('inf')
            best_blocks_for_subset = None
            
            for s_idx in candidates:
                covering = subset_block_arrays[s_idx]
                non_selected = 0
                for b in covering:
                    if b not in selected_set:
                        non_selected += 1
                if non_selected == 0:
                    continue
                if non_selected < best_degree:
                    best_degree = non_selected
                    best_subset = s_idx
                    best_blocks_for_subset = covering
            
            if best_subset == -1:
                # Fallback
                s_idx = uncovered_indices[0]
                covering = subset_block_arrays[s_idx]
                non_sel = [b for b in covering if b not in selected_set]
                if not non_sel:
                    break
                best_b = max(non_sel, key=lambda b: np.sum(uncovered[block_subset_arrays[b]]))
            else:
                covering = best_blocks_for_subset
                non_sel = [b for b in covering if b not in selected_set]
                if not non_sel:
                    break
                best_b = max(non_sel, key=lambda b: np.sum(uncovered[block_subset_arrays[b]]))
            
            selected_blocks.append(best_b)
            selected_set.add(best_b)
            b_subs = block_subset_arrays[best_b]
            newly = b_subs[uncovered[b_subs]]
            uncovered[newly] = False
            
        return selected_blocks

    def lll_random_construction(random_seed=None, p=0.15):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        mask = rng.random(num_blocks) < p
        selected = np.where(mask)[0].tolist()
        
        if not selected:
            return []
        
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in selected:
            cover_count[block_subset_arrays[b_idx]] += 1
            
        if np.any(cover_count == 0):
            uncovered_mask = (cover_count == 0).astype(bool)
            # Simple repair: greedy on uncovered
            uncovered_indices = np.where(uncovered_mask)[0]
            # Use a simple greedy repair
            current_uncovered = uncovered_mask.copy()
            # Re-use gains logic for repair
            gains = np.zeros(num_blocks, dtype=np.int32)
            for s_idx in uncovered_indices:
                blocks_covering = subset_block_arrays[s_idx]
                gains[blocks_covering] += 1
            
            while np.any(current_uncovered):
                if np.max(gains) == 0:
                    break
                best_idx = int(np.argmax(gains))
                selected.append(best_idx)
                b_subs = block_subset_arrays[best_idx]
                newly = b_subs[current_uncovered[b_subs]]
                current_uncovered[newly] = False
                for s in newly:
                    blocks_covering = subset_block_arrays[s]
                    gains[blocks_covering] -= 1
            
            # Recompute coverage
            cover_count = np.zeros(num_subsets, dtype=np.int32)
            for b_idx in selected:
                cover_count[block_subset_arrays[b_idx]] += 1
        
        # Remove redundant blocks
        changed = True
        start_time = time.time()
        while changed and (time.time() - start_time) < 1.0:
            changed = False
            rng.shuffle(selected)
            for b_idx in selected[:]:
                if b_idx not in selected:
                    continue
                b_subs = block_subset_arrays[b_idx]
                if not np.any(cover_count[b_subs] == 1):
                    cover_count[b_subs] -= 1
                    selected.remove(b_idx)
                    changed = True
                    
        return selected

    def minimize_with_swaps(blocks_indices, coverage_counts, time_limit, random_seed=None):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        if not current_blocks:
            return current_blocks, coverage_counts.copy()
            
        cover_count = coverage_counts.copy()
        current_blocks_set = set(current_blocks)
        start_time = time.time()
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            
            # 1. Deletion Pass
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            for i in indices:
                if i >= len(current_blocks):
                    continue
                if (time.time() - start_time) > time_limit:
                    break
                b_idx = current_blocks[i]
                b_subs = block_subset_arrays[b_idx]
                if not np.any(cover_count[b_subs] == 1):
                    cover_count[b_subs] -= 1
                    current_blocks_set.discard(b_idx)
                    current_blocks.pop(i)
                    improved = True
                    break
            
            # 2. Swap Pass (1-for-1)
            if (time.time() - start_time) > time_limit:
                break
                
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            swaps_done = 0
            max_swaps = 50
            
            for i in indices:
                if (time.time() - start_time) > time_limit:
                    break
                if swaps_done >= max_swaps:
                    break
                if i >= len(current_blocks):
                    continue
                    
                old_block = current_blocks[i]
                old_subs = block_subset_arrays[old_block]
                
                unique_mask = cover_count[old_subs] == 1
                unique_subs = old_subs[unique_mask]
                
                if len(unique_subs) == 0:
                    # Not critical: random swap for diversification
                    new_block = rng.randint(0, num_blocks - 1)
                    if new_block == old_block or new_block in current_blocks_set:
                        continue
                    # Check if swap improves "overlap"
                    # Heuristic: Maximize overlap of new block with current solution
                    new_subs = block_subset_arrays[new_block]
                    overlap = np.sum(cover_count[new_subs] > 1)
                    # Only accept if overlap is high? Or just random?
                    # Let's just do it to explore.
                    cover_count[old_subs] -= 1
                    cover_count[new_subs] += 1
                    current_blocks_set.discard(old_block)
                    current_blocks_set.add(new_block)
                    current_blocks[i] = new_block
                    swaps_done += 1
                else:
                    # Critical: find candidates covering all unique subsets
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
                            # Select best candidate: Maximize overlap with current solution
                            best_cand = None
                            best_overlap = -1
                            # Sample if too many candidates
                            cand_list = list(candidates)
                            if len(cand_list) > 20:
                                cand_list = rng.sample(cand_list, 20)
                            
                            for cand in cand_list:
                                new_subs = block_subset_arrays[cand]
                                overlap = np.sum(cover_count[new_subs] > 1)
                                if overlap > best_overlap:
                                    best_overlap = overlap
                                    best_cand = cand
                            
                            if best_cand is not None and best_cand not in current_blocks_set:
                                cover_count[old_subs] -= 1
                                new_subs = block_subset_arrays[best_cand]
                                cover_count[new_subs] += 1
                                current_blocks_set.discard(old_block)
                                current_blocks_set.add(best_cand)
                                current_blocks[i] = best_cand
                                swaps_done += 1
                                improved = True

            # 3. 2-Swap Pass (2-for-1)
            if (time.time() - start_time) > time_limit:
                break
                
            # Try to replace 2 blocks with 1
            # Sample pairs
            num_pairs = 20
            for _ in range(num_pairs):
                if (time.time() - start_time) > time_limit:
                    break
                if len(current_blocks) < 2:
                    break
                i1, i2 = rng.sample(range(len(current_blocks)), 2)
                b1 = current_blocks[i1]
                b2 = current_blocks[i2]
                
                # Find unique subsets of b1 and b2
                # Note: unique is relative to the *entire* set
                # If we remove b1 and b2, what subsets become uncovered?
                # Those are subsets where cover_count == 1 and covered by b1 or b2.
                # Actually, simpler:
                # Calculate the set of subsets that MUST be covered by the new block.
                # These are subsets s such that:
                # 1. s is in b1 or b2
                # 2. cover_count[s] == 1 (if s in b1) OR cover_count[s] == 2 (if s in both? No)
                
                # Let's compute the "must cover" set for the pair {b1, b2}
                # If we remove b1 and b2, subset s is uncovered if:
                # - s in b1 and cover_count[s] == 1
                # - s in b2 and cover_count[s] == 1
                # - s in b1 and s in b2 and cover_count[s] == 2 (impossible, count is sum)
                
                # Actually, if s is in both b1 and b2, cover_count[s] >= 2.
                # If we remove both, count drops by 2. If it was 2, it becomes 0 (uncovered).
                # If it was 3, it becomes 1 (covered).
                
                must_cover = set()
                b1_subs = block_subset_arrays[b1]
                b2_subs = block_subset_arrays[b2]
                
                for s in b1_subs:
                    if cover_count[s] == 1:
                        must_cover.add(s)
                    elif cover_count[s] == 2 and s in b2_subs:
                        # If s is in both, and count is 2, removing both makes it 0
                        must_cover.add(s)
                
                for s in b2_subs:
                    if cover_count[s] == 1:
                        must_cover.add(s)
                    elif cover_count[s] == 2 and s in b1_subs:
                        must_cover.add(s)
                
                if not must_cover:
                    continue
                
                # Find a block that covers all must_cover subsets
                # Intersect subset_to_blocks_sets for all s in must_cover
                candidates = None
                for s in must_cover:
                    if candidates is None:
                        candidates = subset_to_blocks_sets[s].copy()
                    else:
                        candidates.intersection_update(subset_to_blocks_sets[s])
                        if not candidates:
                            break
                
                if candidates:
                    # Remove b1 and b2 from candidates
                    candidates.discard(b1)
                    candidates.discard(b2)
                    if candidates:
                        # Pick one
                        new_block = rng.choice(tuple(candidates))
                        if new_block in current_blocks_set:
                            # If new block is already in the set, we can't add it again.
                            # But we can remove b1 and b2.
                            # Wait, if new_block is already in the set, we just remove b1 and b2.
                            # This reduces block count by 2.
                            # But we must ensure new_block covers must_cover.
                            # It does.
                            # So we just remove b1 and b2.
                            cover_count[b1_subs] -= 1
                            cover_count[b2_subs] -= 1
                            current_blocks_set.discard(b1)
                            current_blocks_set.discard(b2)
                            # Remove from list
                            # Indices i1, i2 are now stale
                            # Remove the one with higher index first
                            idxs = sorted([i1, i2], reverse=True)
                            for idx in idxs:
                                if idx < len(current_blocks):
                                    current_blocks.pop(idx)
                            improved = True
                            break # Restart loop
                        else:
                            # New block not in set. Add it, remove b1, b2.
                            # Count change: -1 for b1, -1 for b2, +1 for new.
                            # Net change: -1 block.
                            cover_count[b1_subs] -= 1
                            cover_count[b2_subs] -= 1
                            new_subs = block_subset_arrays[new_block]
                            cover_count[new_subs] += 1
                            
                            current_blocks_set.discard(b1)
                            current_blocks_set.discard(b2)
                            current_blocks_set.add(new_block)
                            
                            # Update list
                            # Remove b1, b2, add new_block
                            # This is tricky with indices.
                            # Just rebuild list?
                            # Or swap: replace b1 with new_block, remove b2.
                            current_blocks[i1] = new_block
                            # Remove b2 (at i2)
                            # If i2 < i1, i2 is still valid?
                            # We sorted indices for removal above. Here we are modifying.
                            # Let's just use a set for membership and a list for order.
                            # The list order doesn't matter for validity, only for iteration.
                            # We can just replace b1 with new_block and remove b2.
                            if i2 < i1:
                                current_blocks.pop(i2)
                                # i1 is now i1-1
                                current_blocks[i1-1] = new_block
                            else:
                                current_blocks.pop(i2)
                                current_blocks[i1] = new_block
                            
                            improved = True
                            break # Restart loop

        # Final Deletion
        changed = True
        while changed and (time.time() - start_time) < time_limit:
            changed = False
            idx_list = list(range(len(current_blocks)))
            rng.shuffle(idx_list)
            for i in idx_list:
                if i >= len(current_blocks):
                    continue
                b_idx = current_blocks[i]
                b_subs = block_subset_arrays[b_idx]
                if not np.any(cover_count[b_subs] == 1):
                    cover_count[b_subs] -= 1
                    current_blocks_set.discard(b_idx)
                    current_blocks.pop(i)
                    changed = True
                    break

        return current_blocks, cover_count

    best_blocks = None
    best_len = float('inf')
    best_coverage = None
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    trial = 0
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 10.0:
            break
            
        # Strategy 1: Greedy
        greedy_res = run_greedy_full(trial)
        coverage = compute_coverage(greedy_res)
        ls_time = min(3.0, max(1.0, remaining * 0.15))
        min_res, min_cov = minimize_with_swaps(greedy_res, coverage, time_limit=ls_time, random_seed=trial)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
            best_coverage = min_cov
            
        # Strategy 2: Hardest
        if time.time() < deadline - 10:
            hardest_res = run_greedy_hardest(trial + 1000)
            if hardest_res:
                hardest_cov = compute_coverage(hardest_res)
                ls_time_h = min(3.0, max(1.0, (deadline - time.time()) * 0.15))
                min_res_h, min_cov_h = minimize_with_swaps(hardest_res, hardest_cov, time_limit=ls_time_h, random_seed=trial + 2000)
                
                if len(min_res_h) < best_len:
                    best_len = len(min_res_h)
                    best_blocks = min_res_h
                    best_coverage = min_cov_h

        # Strategy 3: LLL
        if time.time() < deadline - 10:
            p_vals = [0.1, 0.15, 0.2, 0.25]
            p = p_vals[trial % len(p_vals)]
            lll_res = lll_random_construction(random_seed=trial + 3000, p=p)
            
            if lll_res:
                lll_coverage = compute_coverage(lll_res)
                ls_time_lla = min(3.0, max(1.0, (deadline - time.time()) * 0.15))
                min_res_lla, min_cov_lla = minimize_with_swaps(lll_res, lll_coverage, time_limit=ls_time_lla, random_seed=trial + 4000)
                
                if len(min_res_lla) < best_len:
                    best_len = len(min_res_lla)
                    best_blocks = min_res_lla
                    best_coverage = min_cov_lla

        # Strategy 4: Polish Best
        if best_blocks is not None and time.time() < deadline - 5:
            ls_time3 = min(4.0, max(1.0, (deadline - time.time()) * 0.2))
            swap_res, swap_cov = minimize_with_swaps(best_blocks, best_coverage, time_limit=ls_time3, random_seed=trial + 5000)
            
            if len(swap_res) < best_len:
                best_len = len(swap_res)
                best_blocks = swap_res
                best_coverage = swap_cov

        trial += 1
        
    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        coverage = compute_coverage(greedy_res)
        best_blocks, best_coverage = minimize_with_swaps(greedy_res, coverage, time_limit=5.0, random_seed=0)
        
    # Final safety
    seen = set()
    unique_blocks = []
    for b in best_blocks:
        if b not in seen:
            seen.add(b)
            unique_blocks.append(b)
    
    return [list(all_blocks[i]) for i in unique_blocks]
