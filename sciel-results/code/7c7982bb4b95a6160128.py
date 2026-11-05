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
            
    # Convert to numpy arrays for faster access in greedy
    block_to_subsets_arr = [np.array(s, dtype=np.int32) if s else np.array([], dtype=np.int32) for s in block_to_subsets]
    subset_to_blocks_arr = [np.array(s, dtype=np.int32) if s else np.array([], dtype=np.int32) for s in subset_to_blocks]
    
    def run_greedy_full(random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
        uncovered_count = num_subsets
        
        selected_blocks = []
        selected_set = set()
        
        while uncovered_count > 0:
            # Add small noise to break ties and allow different paths
            noise = rng.uniform(0, 0.1, size=num_blocks)
            scores = gains.astype(np.float64) + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            if best_idx in selected_set:
                # Should not happen if gains are updated correctly, but safety check
                gains[best_idx] = 0
                continue

            selected_blocks.append(best_idx)
            selected_set.add(best_idx)
            
            b_subs = block_to_subsets_arr[best_idx]
            newly_covered = b_subs[uncovered[b_subs]]
            
            if len(newly_covered) == 0:
                continue
            
            # Update uncovered and gains
            for s in newly_covered:
                uncovered[s] = False
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
        
        # Iterative deletion pass
        changed = True
        while changed:
            changed = False
            # Shuffle to remove order bias
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            
            for i in indices:
                b_idx = current_blocks[i]
                b_subs = block_to_subsets_arr[b_idx]
                if len(b_subs) == 0:
                    continue
                if not np.any(cover_count[b_subs] == 1):
                    # Redundant
                    cover_count[b_subs] -= 1
                    # Remove from list
                    # Using pop(i) is O(N), but N is small (~300)
                    current_blocks.pop(i)
                    changed = True
                    break # Restart loop as indices are now invalid
                    
        return current_blocks

    def local_search_improve(current_blocks, time_limit, random_seed=None):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_set = list(current_blocks)
        if not current_set:
            return current_set
        
        # Calculate coverage counts
        coverage_counts = np.zeros(num_subsets, dtype=np.int32)
        for b in current_set:
            coverage_counts[block_to_subsets_arr[b]] += 1
            
        current_set_set = set(current_set)
        best_len = len(current_set)
        best_set = list(current_set)
        
        start_time = time.time()
        stall_count = 0
        max_stalls = 50
        
        while time.time() - start_time < time_limit:
            if stall_count > max_stalls:
                break
                
            # 1. Deletion Pass
            changed = False
            indices = list(range(len(current_set)))
            rng.shuffle(indices)
            for i in indices:
                b = current_set[i]
                b_subs = block_to_subsets_arr[b]
                if not np.any(coverage_counts[b_subs] == 1):
                    # Remove
                    coverage_counts[b_subs] -= 1
                    current_set.pop(i)
                    current_set_set.discard(b)
                    changed = True
                    break
            
            if changed:
                if len(current_set) < best_len:
                    best_len = len(current_set)
                    best_set = list(current_set)
                continue # Restart loop with new state
            
            # 2. Swap Pass: Find a block with unique subsets and try to replace it
            # To speed up, we can sample a few blocks rather than checking all, 
            # or check all if the set is small. 
            # Let's iterate through blocks.
            
            swap_made = False
            indices = list(range(len(current_set)))
            rng.shuffle(indices)
            
            for i in indices:
                if time.time() - start_time > time_limit:
                    break
                    
                old_block = current_set[i]
                b_subs = block_to_subsets_arr[old_block]
                
                # Find unique subsets for this block
                unique_subs = []
                for s in b_subs:
                    if coverage_counts[s] == 1:
                        unique_subs.append(s)
                
                if not unique_subs:
                    # This block is redundant, deletion pass should have caught it, 
                    # but maybe state changed? If so, delete.
                    coverage_counts[b_subs] -= 1
                    current_set.pop(i)
                    current_set_set.discard(old_block)
                    swap_made = True
                    break
                
                # We need a replacement block that covers ALL unique_subs
                # Find candidates: intersection of subset_to_blocks[s] for s in unique_subs
                
                # Optimization: sort unique_subs by length of their block lists
                unique_subs.sort(key=lambda s: len(subset_to_blocks[s]))
                
                # Start with the smallest list
                min_list = subset_to_blocks[unique_subs[0]]
                if not min_list:
                    continue
                
                candidates = set(min_list)
                
                for s in unique_subs[1:]:
                    candidates.intersection_update(subset_to_blocks[s])
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                # Remove current block and any block already in the set
                candidates.discard(old_block)
                candidates.difference_update(current_set_set)
                
                if candidates:
                    # Pick a candidate. Heuristic: Pick one that maximizes gain (covers most other uncovered or low-count subsets)
                    # For simplicity and speed, pick a random one, or the one with highest total coverage of "critical" subsets
                    # Let's pick the one that covers the most subsets with count == 1 (other than the ones we are fixing, but we are fixing those)
                    # Actually, any valid replacement works. Let's pick one that maximizes the number of subsets it covers with count > 1 
                    # to potentially create new redundancies, or just random.
                    # Let's use a heuristic: score = sum(1 for s in block_to_subsets[cand] if coverage_counts[s] > 1)
                    # This might be expensive. Let's just pick randomly from candidates to keep it fast.
                    # Or pick the one with the highest "potential" to be redundant later?
                    # Let's stick to random selection from candidates for diversity.
                    
                    new_block = rng.choice(list(candidates))
                    
                    # Perform swap
                    # Remove old
                    coverage_counts[block_to_subsets_arr[old_block]] -= 1
                    current_set[i] = new_block
                    current_set_set.discard(old_block)
                    current_set_set.add(new_block)
                    # Add new
                    coverage_counts[block_to_subsets_arr[new_block]] += 1
                    
                    swap_made = True
                    break
            
            if swap_made:
                if len(current_set) < best_len:
                    best_len = len(current_set)
                    best_set = list(current_set)
                stall_count = 0
            else:
                stall_count += 1
                
                # If stuck, try a random perturbation: remove a random block and repair
                if stall_count % 5 == 0:
                    if current_set:
                        # Remove a random block
                        rem_idx = rng.randrange(len(current_set))
                        rem_block = current_set[rem_idx]
                        coverage_counts[block_to_subsets_arr[rem_block]] -= 1
                        current_set.pop(rem_idx)
                        current_set_set.discard(rem_block)
                        
                        # Repair: Find uncovered subsets and add blocks to cover them
                        uncovered = np.where(coverage_counts == 0)[0]
                        if len(uncovered) > 0:
                            # Greedy repair
                            # Simple approach: for each uncovered subset, pick a random block covering it
                            # This is O(U * K) which is fine for small U
                            # But we need to update counts.
                            # Let's use a quick greedy for repair
                            remaining_uncovered = set(uncovered)
                            while remaining_uncovered:
                                # Pick a random uncovered subset
                                s = rng.choice(list(remaining_uncovered))
                                # Pick a random block covering it
                                blocks_covering = subset_to_blocks[s]
                                if not blocks_covering:
                                    break # Should not happen
                                # Prefer blocks not in set? Or just any?
                                # Let's pick one that covers most remaining uncovered
                                best_b = None
                                max_gain = -1
                                # Sample a few blocks to avoid full scan
                                sample_size = min(10, len(blocks_covering))
                                candidates = rng.sample(blocks_covering, sample_size)
                                for b in candidates:
                                    gain = 0
                                    for s2 in block_to_subsets_arr[b]:
                                        if s2 in remaining_uncovered:
                                            gain += 1
                                    if gain > max_gain:
                                        max_gain = gain
                                        best_b = b
                                
                                if best_b is not None:
                                    if best_b not in current_set_set:
                                        current_set.append(best_b)
                                        current_set_set.add(best_b)
                                        coverage_counts[block_to_subsets_arr[best_b]] += 1
                                        # Update remaining_uncovered
                                        for s2 in block_to_subsets_arr[best_b]:
                                            if s2 in remaining_uncovered:
                                                remaining_uncovered.discard(s2)
                                    else:
                                        # If already in set, it should have covered s, so this is an error state
                                        # Just break
                                        pass
                                else:
                                    break
                            
                            # After repair, run deletion
                            changed = True
                            while changed:
                                changed = False
                                indices = list(range(len(current_set)))
                                rng.shuffle(indices)
                                for i in indices:
                                    b = current_set[i]
                                    b_subs = block_to_subsets_arr[b]
                                    if not np.any(coverage_counts[b_subs] == 1):
                                        coverage_counts[b_subs] -= 1
                                        current_set.pop(i)
                                        current_set_set.discard(b)
                                        changed = True
                                        break
                            
                            if len(current_set) < best_len:
                                best_len = len(current_set)
                                best_set = list(current_set)
                            
                            stall_count = 0
                            continue # Restart main loop
                
        return best_set

    # Main Logic
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    trial = 0
    while time.time() < deadline:
        # 1. Generate initial solution
        greedy_res = run_greedy_full(trial)
        
        # 2. Minimize
        min_res = minimize(greedy_res, trial)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
            
        # 3. Local Search on the minimized solution
        # Allocate remaining time to local search
        time_elapsed = time.time() - start_time
        time_remaining = deadline - time_elapsed
        ls_time = max(1.0, time_remaining * 0.4) # Use 40% of remaining time for LS
        
        ls_res = local_search_improve(min_res, ls_time, trial + 500)
        
        if len(ls_res) < best_len:
            best_len = len(ls_res)
            best_blocks = ls_res
            
        trial += 1
        
        # If we have a very good solution (close to lower bound?), we can spend more time polishing
        # Lower bound is 264. If we are below 300, we are in good territory.
        
    # Final polish on the best solution found
    if best_blocks is not None:
        # One last quick pass
        final_time = max(1.0, deadline - (time.time() - start_time) - 1.0)
        if final_time > 0.1:
            final_res = local_search_improve(best_blocks, final_time, 99999)
            if len(final_res) < best_len:
                best_len = len(final_res)
                best_blocks = final_res

    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        best_blocks = minimize(greedy_res, 0)
        
    return [list(all_blocks[i]) for i in best_blocks]
