import itertools
import time
import math

def construct(v, k, t):
    points = range(v)
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    all_mask = (1 << n_t_subsets) - 1
    
    # 2. Generate all k-subsets (candidate blocks)
    k_subsets = list(itertools.combinations(points, k))
    n_k_subsets = len(k_subsets)
    
    # 3. Compute bitmask for each k-subset
    # Also build coverers: list of block indices for each t-subset
    masks = [0] * n_k_subsets
    coverers = [[] for _ in range(n_t_subsets)]
    
    for i, block in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
            coverers[idx].append(i)
        masks[i] = mask
        
    # 4. Greedy Set Cover to get initial Upper Bound (UB)
    uncovered = all_mask
    selected_greedy = []
    used_blocks = [False] * n_k_subsets
    
    # Optimize Greedy: 
    # Instead of scanning all blocks every time, we can just scan all. 
    # 75k blocks * ~300 steps * bit_count(11k bits) is ~22.5M bit_counts.
    # Python int bit_count is fast in C. This should take < 10s.
    
    while uncovered:
        best_gain = -1
        best_idx = -1
        for i in range(n_k_subsets):
            if used_blocks[i]:
                continue
            gain = (masks[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if gain == uncovered.bit_count():
                    break
        if best_idx == -1:
            break
        selected_greedy.append(best_idx)
        used_blocks[best_idx] = True
        uncovered &= ~masks[best_idx]
        
    current_best = len(selected_greedy)
    best_solution = selected_greedy[:]
    
    # 5. Branch and Bound with Time Limit
    start_time = time.time()
    time_limit = 100 # seconds
    
    # Precompute max gain per block for lower bound estimation
    # Max possible gain is C(k, t)
    max_gain_bound = math.comb(k, t)
    
    # To speed up B&B, we maintain a set of used blocks.
    # We pick the "hardest" t-subset: one with the fewest available covering blocks.
    # To find this efficiently, we can maintain counts, but updating counts is expensive.
    # Alternative: Just pick the first uncovered t-subset.
    # Better: Pick the uncovered t-subset with the smallest number of *total* coverers?
    # The number of total coverers is static (364). So that doesn't help distinguish.
    # We need the number of *available* coverers.
    
    # Let's try a simple B&B first.
    
    def branch_and_bound(uncovered_mask, current_depth, current_path, used_set):
        nonlocal current_best, best_solution
        
        if time.time() - start_time > time_limit:
            return False # Timeout signal
        
        if not uncovered_mask:
            if current_depth < current_best:
                current_best = current_depth
                best_solution = current_path[:]
            return True
        
        # Lower Bound Pruning
        # Simple LB: ceil(bits / max_gain)
        bits = uncovered_mask.bit_count()
        lb = (bits + max_gain_bound - 1) // max_gain_bound
        if current_depth + lb >= current_best:
            return True
        
        # Select target t-subset
        # Find the first set bit
        target_bit = uncovered_mask & -uncovered_mask
        target_idx = target_bit.bit_length() - 1
        
        # Get candidates from coverers[target_idx]
        candidates = []
        for i in coverers[target_idx]:
            if not used_set[i]:
                # Check if it actually covers the target (it should by definition of coverers)
                # And check gain
                gain = (masks[i] & uncovered_mask).bit_count()
                if gain > 0:
                    candidates.append((gain, i))
        
        if not candidates:
            return True # Dead end
        
        # Sort candidates by gain descending
        candidates.sort(reverse=True)
        
        for gain, i in candidates:
            if current_depth + 1 >= current_best:
                break
            
            new_uncovered = uncovered_mask & ~masks[i]
            
            # Additional Pruning: 
            # If the new uncovered set has bits, and depth+1+LB >= best, skip
            new_bits = new_uncovered.bit_count()
            new_lb = (new_bits + max_gain_bound - 1) // max_gain_bound
            if current_depth + 1 + new_lb >= current_best:
                continue
            
            used_set[i] = True
            # Recurse
            if not branch_and_bound(new_uncovered, current_depth + 1, current_path + [i], used_set):
                return False # Timeout propagated
            used_set[i] = False
            
        return True

    # Initialize used_set for B&B
    # We can start from the greedy solution and try to improve?
    # Or start from scratch?
    # Starting from scratch with B&B is risky if the tree is huge.
    # But we have a good UB from Greedy.
    
    # Let's try to run B&B.
    # To make it more effective, we can use the greedy solution to order blocks?
    # No, standard B&B is fine.
    
    # One optimization: If the greedy solution is already very good, B&B might not find anything better quickly.
    # We can also try "Local Search" on the greedy solution to tighten UB before B&B.
    
    def local_search(blocks_list):
        improved = True
        while improved:
            improved = False
            for i in range(len(blocks_list)):
                # Try to remove block i
                block_mask = masks[blocks_list[i]]
                # Check if union of others covers all
                union_others = 0
                for j in range(len(blocks_list)):
                    if j != i:
                        union_others |= masks[blocks_list[j]]
                
                if union_others == all_mask:
                    blocks_list.pop(i)
                    improved = True
                    break # Restart loop as indices changed
                
                # Try to swap block i with another block?
                # If removing i leaves a gap, can we fill the gap with a block j that is already in the set? No.
                # Can we swap i with a block k NOT in the set, such that k covers the gap AND allows removing another block?
                # This is complex. Let's stick to removal for now, or simple 1-for-1 swap if it reduces "uniqueness".
                # Actually, a better local search:
                # For each block i, calculate the set of t-subsets covered ONLY by i (Unique_i).
                # If Unique_i is empty, remove i.
                # If Unique_i is not empty, look for a block k (not in set) such that Unique_i is a subset of mask(k).
                # If found, swap i with k? No, that keeps count same.
                # We want to reduce count.
                # So, if we remove i, we get a gap G.
                # If we can cover G with 0 blocks, remove i.
                # If we can cover G with 1 block k, and k is already in the set? No, k is not in set.
                # If we add k and remove i, count is same.
                # To reduce count, we must find a redundancy.
                
        return blocks_list

    # Run Local Search on Greedy Result
    best_solution = local_search(best_solution)
    current_best = len(best_solution)
    
    # Run B&B
    # We need a used_set. Since B&B explores different paths, we need a mutable set.
    # But B&B is recursive.
    # We start with an empty used_set? No, we want to find a solution *better* than current_best.
    # We can prune any branch that reaches depth >= current_best.
    
    # To help B&B, we can order the blocks in `coverers` by their "quality" (e.g., how many t-subsets they cover, or random).
    # They are already generated in some order.
    
    # Let's run B&B for the remaining time.
    used_set = [False] * n_k_subsets
    branch_and_bound(all_mask, 0, [], used_set)
    
    return [list(k_subsets[i]) for i in best_solution]
