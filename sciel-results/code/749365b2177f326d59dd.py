import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # 1. Precomputation
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Compute masks for each block using NumPy for speed
    block_masks = np.zeros(num_blocks, dtype=np.uint64)
    # Also store the list of subset indices for each block for faster iteration later
    block_subsets = [[] for _ in range(num_blocks)]
    
    for i, block in enumerate(all_blocks):
        mask = 0
        subs = []
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
            subs.append(idx)
        block_masks[i] = mask
        block_subsets[i] = subs
        
    # Inverted index: subset_idx -> list of block indices
    subset_to_blocks = [[] for _ in range(num_t_subsets)]
    for i in range(num_blocks):
        for idx in block_subsets[i]:
            subset_to_blocks[idx].append(i)

    def count_bits(x):
        return x.bit_count()

    def greedy_construct(randomize=False, rng=None):
        """
        Greedy set cover.
        If randomize is True, adds slight noise to break ties and explore different solutions.
        """
        if rng is None:
            rng = random
        uncovered = (1 << num_t_subsets) - 1
        chosen = []
        used = [False] * num_blocks
        
        # Precompute subset masks as list of 1<<idx for fast AND
        subset_masks = [1 << i for i in range(num_t_subsets)]
        
        while uncovered != 0:
            best_gain = -1
            best_idx = -1
            second_best_gain = -1
            second_best_idx = -1
            
            for i in range(num_blocks):
                if used[i]:
                    continue
                gain = count_bits(block_masks[i] & uncovered)
                if gain > best_gain:
                    second_best_gain = best_gain
                    second_best_idx = best_idx
                    best_gain = gain
                    best_idx = i
                elif gain > second_best_gain:
                    second_best_gain = gain
                    second_best_idx = i
                    
            if best_idx == -1:
                break  # Should not happen if uncovered != 0 and valid blocks exist
                
            # Randomized selection: with some probability pick second best
            if randomize and second_best_idx != -1 and second_best_gain > 0:
                # Probability of picking second best decreases with gap
                gap = best_gain - second_best_gain
                if gap == 0:
                    prob_second = 0.5
                elif gap == 1:
                    prob_second = 0.3
                elif gap == 2:
                    prob_second = 0.15
                else:
                    prob_second = 0.05
                if rng.random() < prob_second:
                    best_idx = second_best_idx
                    best_gain = second_best_gain
            
            chosen.append(best_idx)
            used[best_idx] = True
            uncovered &= ~block_masks[best_idx]
            
        return chosen

    def prune_solution(chosen_indices):
        """
        Removes blocks that are not essential.
        Returns the pruned list of indices.
        """
        if not chosen_indices:
            return []
        
        # Count coverage of each subset
        counts = [0] * num_t_subsets
        chosen_set = set(chosen_indices)
        
        for idx in chosen_indices:
            for s_idx in block_subsets[idx]:
                counts[s_idx] += 1
        
        # Iteratively remove non-essential blocks
        changed = True
        current_list = list(chosen_indices)
        
        while changed:
            changed = False
            new_list = []
            for idx in current_list:
                # Check if block idx is essential
                essential = False
                for s_idx in block_subsets[idx]:
                    if counts[s_idx] == 1:
                        essential = True
                        break
                
                if not essential:
                    # Remove block idx
                    for s_idx in block_subsets[idx]:
                        counts[s_idx] -= 1
                    changed = True
                else:
                    new_list.append(idx)
            
            current_list = new_list
        
        return current_list

    def try_swap_improvement(chosen_indices, max_iterations=20, rng=None):
        """
        Attempts to improve the solution by swapping blocks.
        Returns the new list if improved, else the original.
        """
        if rng is None:
            rng = random
        current_solution = list(chosen_indices)
        improved = True
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            # Compute counts for current solution
            counts = [0] * num_t_subsets
            for idx in current_solution:
                for s_idx in block_subsets[idx]:
                    counts[s_idx] += 1
            
            # Try to remove and replace each block
            for i in range(len(current_solution)):
                idx = current_solution[i]
                
                # Find essential subsets for block idx
                essential_subs = [s for s in block_subsets[idx] if counts[s] == 1]
                
                if not essential_subs:
                    # Block is redundant, remove it
                    for s in block_subsets[idx]:
                        counts[s] -= 1
                    current_solution.pop(i)
                    improved = True
                    break  # Restart loop as list changed
                
                # Find candidates that cover ALL essential_subs
                # Pick the essential sub with fewest covering blocks for efficiency
                essential_subs_sorted = sorted(essential_subs, key=lambda s: len(subset_to_blocks[s]))
                first_essential = essential_subs_sorted[0]
                pool = subset_to_blocks[first_essential]
                
                best_candidate = -1
                best_score = -1
                
                # Score function: maximize coverage of already-covered subsets,
                # minimize coverage of uncovered subsets
                for cand_idx in pool:
                    if cand_idx in current_solution:
                        continue
                    
                    # Check if cand covers all essential_subs
                    cand_mask = block_masks[cand_idx]
                    covers_all = True
                    for s in essential_subs:
                        if not (cand_mask & (1 << s)):
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    
                    # Score: count how many subsets in cand are already covered (counts > 0 after removing idx)
                    # After removing idx, a subset s has count = counts[s] - (1 if s in idx else 0)
                    # We want to maximize the number of s in cand where (counts[s] - (1 if s in idx else 0)) > 0
                    # This means the subset remains covered after the swap
                    
                    idx_subs_set = set(block_subsets[idx])
                    cand_subs = block_subsets[cand_idx]
                    
                    still_covered = 0
                    newly_covered = 0
                    for s in cand_subs:
                        if s in idx_subs_set:
                            # s is in both idx and cand
                            # After swap: count remains counts[s]
                            if counts[s] > 1:
                                still_covered += 1
                        else:
                            # s is in cand but not in idx
                            # After swap: count becomes counts[s] + 1
                            if counts[s] >= 1:
                                still_covered += 1
                            else:
                                newly_covered += 1
                    
                    # Primary: maximize still_covered (redundancy)
                    # Secondary: minimize newly_covered (new uncovered risk)
                    score = still_covered * 10 - newly_covered
                    
                    if score > best_score:
                        best_score = score
                        best_candidate = cand_idx
                
                if best_candidate != -1:
                    # Perform swap
                    current_solution[i] = best_candidate
                    improved = True
                    break  # Restart scanning with new solution
            
        return current_solution

    def local_search_2opt(chosen_indices, max_iterations=10, rng=None):
        """
        Try to remove 2 blocks and add 1 block that covers all their essential subsets.
        """
        if rng is None:
            rng = random
        current_solution = list(chosen_indices)
        improved = True
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            n = len(current_solution)
            if n < 2:
                break
            
            # Compute counts
            counts = [0] * num_t_subsets
            for idx in current_solution:
                for s_idx in block_subsets[idx]:
                    counts[s_idx] += 1
            
            # Try all pairs
            for i in range(n):
                if improved:
                    break
                for j in range(i + 1, n):
                    idx1 = current_solution[i]
                    idx2 = current_solution[j]
                    
                    # Find essential subsets for idx1 and idx2
                    essential1 = [s for s in block_subsets[idx1] if counts[s] == 1]
                    essential2 = [s for s in block_subsets[idx2] if counts[s] == 1]
                    
                    # Also find subsets that become uncovered when both are removed
                    # These are subsets where count == 2 and both idx1 and idx2 cover them
                    both_essential = set()
                    for s in block_subsets[idx1]:
                        if counts[s] == 2:
                            both_essential.add(s)
                    for s in block_subsets[idx2]:
                        if counts[s] == 2:
                            both_essential.add(s)
                    
                    all_essential = set(essential1) | set(essential2) | both_essential
                    if not all_essential:
                        # Both blocks are redundant
                        for s in block_subsets[idx1]:
                            counts[s] -= 1
                        for s in block_subsets[idx2]:
                            counts[s] -= 1
                        # Remove in reverse order to preserve indices
                        current_solution.pop(j)
                        current_solution.pop(i)
                        improved = True
                        break
                    
                    # Find a single block that covers all essential subsets
                    # Use the rarest essential subset to limit candidates
                    essential_list = list(all_essential)
                    essential_list.sort(key=lambda s: len(subset_to_blocks[s]))
                    pool = subset_to_blocks[essential_list[0]]
                    
                    best_candidate = -1
                    best_score = -1
                    
                    for cand_idx in pool:
                        if cand_idx in current_solution:
                            continue
                        
                        # Check if cand covers all essential subsets
                        cand_mask = block_masks[cand_idx]
                        covers_all = True
                        for s in all_essential:
                            if not (cand_mask & (1 << s)):
                                covers_all = False
                                break
                        if not covers_all:
                            continue
                        
                        # Score: maximize redundancy
                        cand_subs = block_subsets[cand_idx]
                        idx1_set = set(block_subsets[idx1])
                        idx2_set = set(block_subsets[idx2])
                        
                        still_covered = 0
                        newly_covered = 0
                        for s in cand_subs:
                            # After removing idx1 and idx2, adding cand:
                            # count_new = counts[s] - (1 if s in idx1) - (1 if s in idx2) + (1 if s in cand)
                            old_count = counts[s]
                            in_idx1 = 1 if s in idx1_set else 0
                            in_idx2 = 1 if s in idx2_set else 0
                            in_cand = 1
                            new_count = old_count - in_idx1 - in_idx2 + in_cand
                            if new_count > 0:
                                if old_count - in_idx1 - in_idx2 > 0:
                                    still_covered += 1
                                else:
                                    newly_covered += 1
                        
                        score = still_covered * 10 - newly_covered
                        
                        if score > best_score:
                            best_score = score
                            best_candidate = cand_idx
                    
                    if best_candidate != -1:
                        # Perform the exchange: remove idx1, idx2, add best_candidate
                        # Remove in reverse order
                        current_solution.pop(j)
                        current_solution.pop(i)
                        current_solution.append(best_candidate)
                        improved = True
                        break
            
        return current_solution

    # Main Loop: Try multiple random greedy constructions
    best_solution = None
    best_len = float('inf')
    
    # Initial greedy (deterministic)
    init_sol = greedy_construct(randomize=False)
    pruned_init = prune_solution(init_sol)
    if len(pruned_init) < best_len:
        best_len = len(pruned_init)
        best_solution = pruned_init
    
    # Try randomized greedy + swap improvements
    # Time budget check
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    cycles = 0
    rng = random.Random(42)
    while time.time() < deadline - 10.0:  # Keep 10s for final safety
        # Randomized Greedy
        sol = greedy_construct(randomize=True, rng=rng)
        sol = prune_solution(sol)
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
        
        # Try to improve with swaps
        improved_sol = try_swap_improvement(sol, max_iterations=15, rng=rng)
        
        if len(improved_sol) < best_len:
            best_len = len(improved_sol)
            best_solution = improved_sol
        
        # Try 2-opt local search
        if time.time() < deadline - 15.0:
            improved_2opt = local_search_2opt(best_solution, max_iterations=5, rng=rng)
            if len(improved_2opt) < best_len:
                best_len = len(improved_2opt)
                best_solution = improved_2opt
        
        cycles += 1
        # If we hit 131, we are done (lower bound)
        if best_len <= 131:
            break
    
    # Final polish: try more aggressive local search on best solution
    if time.time() < deadline - 5.0 and best_solution is not None:
        final_sol = try_swap_improvement(best_solution, max_iterations=30, rng=rng)
        if len(final_sol) < best_len:
            best_len = len(final_sol)
            best_solution = final_sol
        
        final_2opt = local_search_2opt(best_solution, max_iterations=10, rng=rng)
        if len(final_2opt) < best_len:
            best_len = len(final_2opt)
            best_solution = final_2opt
    
    # Convert indices to lists of points
    result_blocks = [list(all_blocks[i]) for i in best_solution]
    
    return result_blocks

# Quick test
if __name__ == "__main__":
    # Test with v=19, k=7, t=4
    blocks = construct(19, 7, 4)
    print(f"Number of blocks: {len(blocks)}")
    # Verify coverage
    v, k, t = 19, 7, 4
    all_t = list(itertools.combinations(range(v), 4))
    covered = set()
    for b in blocks:
        for sub in itertools.combinations(b, 4):
            covered.add(sub)
    is_valid = all(t_sub in covered for t_sub in all_t)
    print(f"Valid: {is_valid}")
