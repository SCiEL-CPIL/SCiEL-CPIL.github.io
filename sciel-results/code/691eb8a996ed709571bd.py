import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (kill at 150s)
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_t_subsets)
    t_subset_to_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_subsets)
    
    # Precompute coverage: which t-subsets does each block cover?
    block_to_tsubs = []
    block_to_tsubs_set = []
    for block in all_k_subsets:
        covered = []
        for tsub in itertools.combinations(block, t):
            idx = t_subset_to_index[tsub]
            covered.append(idx)
        block_to_tsubs.append(covered)
        block_to_tsubs_set.append(set(covered))
        
    # Precompute inverse: which blocks cover each t-subset?
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, tsubs in enumerate(block_to_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
            
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # Precompute scarcity weights for greedy bias
    # weight[b] = sum(1 / degree(t) for t in block)
    tsub_degrees = [len(lst) for lst in tsub_to_blocks]
    block_scarcity = np.zeros(num_blocks, dtype=np.float64)
    for b_idx, tsubs in enumerate(block_to_tsubs):
        w = 0.0
        for s_idx in tsubs:
            if tsub_degrees[s_idx] > 0:
                w += 1.0 / tsub_degrees[s_idx]
        block_scarcity[b_idx] = w

    # Helper: Weighted Greedy Construction
    def greedy_construction(randomize_ties=False, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        uncovered = np.ones(num_tsubs, dtype=bool)
        gains = np.full(num_blocks, len(block_to_tsubs[0]), dtype=np.int32)
        
        selected_block_indices = []
        
        while uncovered.any():
            if time.time() > deadline:
                return None # Timeout
            
            # Calculate score: gain * large_factor + scarcity
            # We only need to consider blocks with gain > 0
            # To be fast, we can just argmax gains, then break ties with scarcity
            # But gains change.
            
            # Efficient max gain search
            # Since gains are small ints, we can find max gain and then pick among those with max gain
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all blocks with max_gain
            # This can be done with np.where(gains == max_gain)
            candidates = np.where(gains == max_gain)[0]
            
            if len(candidates) == 1:
                b = candidates[0]
            else:
                # Tie-breaking: Pick block with highest scarcity among candidates
                # Or randomize
                if randomize_ties:
                    # With some probability, pick random from candidates
                    # With some probability, pick highest scarcity
                    if random.random() < 0.5:
                        b = random.choice(candidates)
                    else:
                        # Find max scarcity among candidates
                        cand_scarcity = block_scarcity[candidates]
                        b = candidates[np.argmax(cand_scarcity)]
                else:
                    # Deterministic: Highest scarcity
                    cand_scarcity = block_scarcity[candidates]
                    b = candidates[np.argmax(cand_scarcity)]
            
            selected_block_indices.append(b)
            
            # Update uncovered and gains
            for s_idx in block_to_tsubs[b]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    gains[tsub_to_blocks_np[s_idx]] -= 1
                    
        if uncovered.any():
            return None # Failed to cover
            
        return selected_block_indices

    # Helper: Local Optimization
    def local_optimize(blocks):
        if not blocks:
            return []
            
        selected_set = set(blocks)
        selected_list = list(blocks)
        
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected_list:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Step A: Redundancy Removal
            redundant_blocks = []
            for b in selected_list:
                is_red = True
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        is_red = False
                        break
                if is_red:
                    redundant_blocks.append(b)
            
            if redundant_blocks:
                for b in redundant_blocks:
                    selected_list.remove(b)
                    selected_set.remove(b)
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
                improved = True
                continue # Restart
            
            # Step B: 1-for-1 Swaps
            # Calculate unique t-subsets for each block
            unique_tsubs_map = {}
            unique_counts = {}
            for b in selected_list:
                u_list = []
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        u_list.append(s)
                unique_tsubs_map[b] = u_list
                unique_counts[b] = len(u_list)
            
            if not unique_counts:
                break
                
            # Sort blocks by unique count to try removing "easy" ones first
            sorted_blocks = sorted(selected_list, key=lambda x: unique_counts[x])
            
            swap_made = False
            for b_out in sorted_blocks:
                if time.time() > deadline:
                    break
                s_uniq = unique_tsubs_map[b_out]
                if not s_uniq:
                    continue # Should have been removed as redundant
                
                # Find replacement
                # Candidates must cover all s_uniq
                # Start with candidates of the first unique subset
                first_s = s_uniq[0]
                candidates = tsub_to_blocks_np[first_s]
                
                best_candidate = None
                # We want a candidate that covers all s_uniq and is not in selected_set
                # Ideally, we want a candidate that maximizes "extra" coverage? No, just any valid one.
                # To improve, we might prefer one that allows further reductions?
                # For now, just find any.
                
                for c in candidates:
                    if c in selected_set:
                        continue
                    c_set = block_to_tsubs_set[c]
                    covers_all = True
                    for s in s_uniq:
                        if s not in c_set:
                            covers_all = False
                            break
                    if covers_all:
                        best_candidate = c
                        break # Take first one found? Or random?
                        # Taking first one found might be biased. Let's collect a few and pick random.
                        # But for speed, first found is okay.
                
                if best_candidate is not None:
                    # Perform swap
                    selected_list.remove(b_out)
                    selected_set.remove(b_out)
                    selected_list.append(best_candidate)
                    selected_set.add(best_candidate)
                    
                    for s in block_to_tsubs[b_out]:
                        cover_count[s] -= 1
                    for s in block_to_tsubs[best_candidate]:
                        cover_count[s] += 1
                    
                    swap_made = True
                    improved = True
                    break # Restart loop after one swap
            
            if not swap_made:
                # Step C: 2-for-1 Swaps (Systematic with random fallback)
                # Try to find two blocks whose union of unique subsets is covered by a third block
                
                found_2for1 = False
                n_sel = len(selected_list)
                
                # First pass: systematic pairs (limited to avoid O(n^2) explosion)
                max_sys_pairs = min(500, n_sel * (n_sel - 1) // 2)
                pairs_tried = 0
                for i in range(n_sel):
                    if time.time() > deadline:
                        break
                    if pairs_tried >= max_sys_pairs:
                        break
                    for j in range(i + 1, n_sel):
                        if pairs_tried >= max_sys_pairs or time.time() > deadline:
                            break
                        pairs_tried += 1
                        b1, b2 = selected_list[i], selected_list[j]
                        
                        # Calculate required subsets: those that would become uncovered
                        # if we remove both b1 and b2
                        b2_set = block_to_tsubs_set[b2]
                        required_subsets = []
                        for s in block_to_tsubs[b1]:
                            cnt = cover_count[s]
                            if s in b2_set:
                                cnt -= 1
                            if cnt == 0:
                                required_subsets.append(s)
                        for s in block_to_tsubs[b2]:
                            if s in b2_set and s not in block_to_tsubs[b1]:
                                # This is in b2 but not b1
                                if cover_count[s] == 1:  # Only covered by b2
                                    required_subsets.append(s)
                        
                        if not required_subsets:
                            continue
                        
                        # Find a block c that covers all required_subsets
                        first_req = required_subsets[0]
                        candidates = tsub_to_blocks_np[first_req]
                        
                        for c in candidates:
                            if c == b1 or c == b2 or c in selected_set:
                                continue
                            c_set = block_to_tsubs_set[c]
                            covers_all = all(s in c_set for s in required_subsets)
                            if covers_all:
                                # Perform 2-for-1 swap
                                selected_list.remove(b1)
                                selected_list.remove(b2)
                                selected_set.remove(b1)
                                selected_set.remove(b2)
                                selected_list.append(c)
                                selected_set.add(c)
                                
                                for s in block_to_tsubs[b1]:
                                    cover_count[s] -= 1
                                for s in block_to_tsubs[b2]:
                                    cover_count[s] -= 1
                                for s in block_to_tsubs[c]:
                                    cover_count[s] += 1
                                
                                found_2for1 = True
                                improved = True
                                break
                        
                        if found_2for1:
                            break
                    if found_2for1:
                        break
                
                # If systematic didn't find anything, try random pairs
                if not found_2for1 and time.time() < deadline:
                    random_pairs = 2000
                    for _ in range(random_pairs):
                        if time.time() > deadline:
                            break
                        if n_sel < 2:
                            break
                        b1, b2 = random.sample(selected_list, 2)
                        
                        b2_set = block_to_tsubs_set[b2]
                        required_subsets = []
                        for s in block_to_tsubs[b1]:
                            cnt = cover_count[s]
                            if s in b2_set:
                                cnt -= 1
                            if cnt == 0:
                                required_subsets.append(s)
                        for s in block_to_tsubs[b2]:
                            if s not in block_to_tsubs[b1]:
                                if cover_count[s] == 1:
                                    required_subsets.append(s)
                        
                        if not required_subsets:
                            continue
                        
                        first_req = required_subsets[0]
                        candidates = tsub_to_blocks_np[first_req]
                        
                        for c in candidates:
                            if c == b1 or c == b2 or c in selected_set:
                                continue
                            c_set = block_to_tsubs_set[c]
                            if all(s in c_set for s in required_subsets):
                                selected_list.remove(b1)
                                selected_list.remove(b2)
                                selected_set.remove(b1)
                                selected_set.remove(b2)
                                selected_list.append(c)
                                selected_set.add(c)
                                
                                for s in block_to_tsubs[b1]:
                                    cover_count[s] -= 1
                                for s in block_to_tsubs[b2]:
                                    cover_count[s] -= 1
                                for s in block_to_tsubs[c]:
                                    cover_count[s] += 1
                                
                                found_2for1 = True
                                improved = True
                                break
                        
                        if found_2for1:
                            break
                
                # Step D: 2-for-2 Swaps (replace two blocks with two different blocks)
                if not found_2for1 and time.time() < deadline:
                    found_2for2 = False
                    # Try random pairs and look for two replacements
                    random_pairs_2for2 = 1000
                    for _ in range(random_pairs_2for2):
                        if time.time() > deadline:
                            break
                        if n_sel < 2:
                            break
                        b1, b2 = random.sample(selected_list, 2)
                        
                        b2_set = block_to_tsubs_set[b2]
                        required_subsets = []
                        for s in block_to_tsubs[b1]:
                            cnt = cover_count[s]
                            if s in b2_set:
                                cnt -= 1
                            if cnt == 0:
                                required_subsets.append(s)
                        for s in block_to_tsubs[b2]:
                            if s not in block_to_tsubs[b1]:
                                if cover_count[s] == 1:
                                    required_subsets.append(s)
                        
                        if not required_subsets:
                            continue
                        
                        # Split required into two groups and find two blocks
                        # Heuristic: use first half and second half
                        mid = len(required_subsets) // 2
                        req1 = required_subsets[:mid] if mid > 0 else [required_subsets[0]]
                        req2 = required_subsets[mid:] if mid > 0 else [required_subsets[-1]]
                        
                        # Find block c1 for req1
                        c1_found = False
                        c1 = None
                        if req1:
                            for c in tsub_to_blocks_np[req1[0]]:
                                if c == b1 or c == b2 or c in selected_set:
                                    continue
                                if all(s in block_to_tsubs_set[c] for s in req1):
                                    c1 = c
                                    c1_found = True
                                    break
                        
                        if c1_found:
                            # Find block c2 for req2, c2 != c1
                            c2_found = False
                            c2 = None
                            if req2:
                                for c in tsub_to_blocks_np[req2[0]]:
                                    if c == b1 or c == b2 or c == c1 or c in selected_set:
                                        continue
                                    if all(s in block_to_tsubs_set[c] for s in req2):
                                        c2 = c
                                        c2_found = True
                                        break
                            
                            if c2_found:
                                # Verify together they cover all required
                                c1_set = block_to_tsubs_set[c1]
                                c2_set = block_to_tsubs_set[c2]
                                if all(s in c1_set or s in c2_set for s in required_subsets):
                                    selected_list.remove(b1)
                                    selected_list.remove(b2)
                                    selected_set.remove(b1)
                                    selected_set.remove(b2)
                                    selected_list.extend([c1, c2])
                                    selected_set.update([c1, c2])
                                    
                                    for s in block_to_tsubs[b1]:
                                        cover_count[s] -= 1
                                    for s in block_to_tsubs[b2]:
                                        cover_count[s] -= 1
                                    for s in block_to_tsubs[c1]:
                                        cover_count[s] += 1
                                    for s in block_to_tsubs[c2]:
                                        cover_count[s] += 1
                                    
                                    found_2for2 = True
                                    improved = True
                                    break
                        
                        if found_2for2:
                            break
                
                if not improved:
                    break
                    
        return selected_list

    best_solution = None
    best_len = float('inf')
    
    # Multi-start strategy
    # Run greedy + local search multiple times
    
    # First run deterministic
    blocks = greedy_construction(randomize_ties=False, random_seed=42)
    if blocks is not None:
        optimized_blocks = local_optimize(blocks)
        if len(optimized_blocks) < best_len:
            best_len = len(optimized_blocks)
            best_solution = optimized_blocks
    
    # Second run with slight randomization for diversity
    if time.time() < deadline - 20:
        blocks2 = greedy_construction(randomize_ties=True, random_seed=100)
        if blocks2 is not None:
            optimized_blocks2 = local_optimize(blocks2)
            if len(optimized_blocks2) < best_len:
                best_len = len(optimized_blocks2)
                best_solution = optimized_blocks2
    
    # Subsequent runs with randomization
    for i in range(1, 40):
        if time.time() > deadline - 15:
            break
            
        blocks = greedy_construction(randomize_ties=True, random_seed=42 + i)
        
        if blocks is None:
            continue
            
        optimized_blocks = local_optimize(blocks)
        
        if len(optimized_blocks) < best_len:
            best_len = len(optimized_blocks)
            best_solution = optimized_blocks
            
    # If we have time, do ILS on the best solution
    if best_solution is not None and time.time() < deadline - 10:
        current_best = list(best_solution)
        
        # Perturb and Repair with multiple perturbation strategies
        ild_iterations = 0
        while time.time() < deadline - 5 and ild_iterations < 20:
            ild_iterations += 1
            
            # Choose perturbation strategy based on iteration
            strategy = ild_iterations % 3
            
            perturbed = list(current_best)
            
            if strategy == 0:
                # Remove 5 random blocks
                if len(perturbed) > 10:
                    remove_indices = random.sample(range(len(perturbed)), 5)
                    remove_indices.sort(reverse=True)
                    for idx in remove_indices:
                        perturbed.pop(idx)
            elif strategy == 1:
                # Remove 3 blocks and add 2 random blocks (swap-like perturbation)
                if len(perturbed) > 8:
                    remove_indices = random.sample(range(len(perturbed)), 3)
                    remove_indices.sort(reverse=True)
                    for idx in remove_indices:
                        perturbed.pop(idx)
                    # Add 2 random blocks
                    random_blocks = random.sample(range(num_blocks), 2)
                    for rb in random_blocks:
                        if rb not in perturbed:
                            perturbed.append(rb)
            else:
                # Remove 7 random blocks (larger perturbation)
                if len(perturbed) > 12:
                    remove_indices = random.sample(range(len(perturbed)), 7)
                    remove_indices.sort(reverse=True)
                    for idx in remove_indices:
                        perturbed.pop(idx)
            
            # Repair: Greedy to cover gaps
            cover_count = np.zeros(num_tsubs, dtype=np.int32)
            for b in perturbed:
                for s in block_to_tsubs[b]:
                    cover_count[s] += 1
            
            uncovered_mask = cover_count == 0
            
            selected_set = set(perturbed)
            
            # Calculate gains for all blocks relative to current uncovered
            gains = np.zeros(num_blocks, dtype=np.int32)
            for b_idx in range(num_blocks):
                if b_idx in selected_set:
                    continue
                gain = 0
                for s in block_to_tsubs[b_idx]:
                    if uncovered_mask[s]:
                        gain += 1
                gains[b_idx] = gain
            
            while uncovered_mask.any():
                if time.time() > deadline - 2:
                    break
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                # Pick one with highest scarcity
                cand_scarcity = block_scarcity[candidates]
                b = candidates[np.argmax(cand_scarcity)]
                
                selected_set.add(b)
                for s in block_to_tsubs[b]:
                    if uncovered_mask[s]:
                        uncovered_mask[s] = False
                        gains[tsub_to_blocks_np[s]] -= 1
            
            # Local Optimize the result
            result_blocks = list(selected_set)
            optimized = local_optimize(result_blocks)
            
            if len(optimized) < best_len:
                best_len = len(optimized)
                best_solution = optimized
                current_best = optimized

    if best_solution is None:
        return []
        
    result = [list(all_k_subsets[i]) for i in best_solution]
    return result
