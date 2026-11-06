# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from math import gcd
    
    # Set a safety deadline. The evaluator kills at 90s, but we want to return well before.
    deadline = time.time() + 55.0
    
    nums = list(range(2, 257))
    n = len(nums)
    
    # Precompute conflicts. 
    # conflict[i] is a set of indices j such that lcm(nums[i], nums[j]) <= 256.
    conflict = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = gcd(nums[i], nums[j])
            if (nums[i] // g) * nums[j] <= 256:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Convert conflicts to bitsets for faster operations
    conflict_bits = [0] * n
    for i in range(n):
        mask = 0
        for j in conflict[i]:
            mask |= (1 << j)
        conflict_bits[i] = mask

    weights = [1.0 / nums[i] for i in range(n)]
    
    # Initial heuristic to get a good lower bound
    # Reuse the logic from the original program but simplified for speed
    def is_valid_add(selected, i):
        if len(selected) < len(conflict[i]):
            return not selected.intersection(conflict[i])
        else:
            return not conflict[i].intersection(selected)

    def local_search(init_set, max_iters=10):
        best = set(init_set)
        best_score = sum(weights[i] for i in best)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            improved = False
            
            # 1. Try adding one element
            best_add = -1
            best_add_val = 0.0
            for i in range(n):
                if i in best:
                    continue
                if is_valid_add(best, i):
                    val = weights[i]
                    if val > best_add_val:
                        best_add_val = val
                        best_add = i
            
            if best_add != -1:
                best.add(best_add)
                best_score += best_add_val
                improved = True
            
            # 2. Try remove-one-add-one (swap)
            if time.time() < deadline - 10.0:
                best_swap = None
                best_swap_gain = 0.0
                for i in best:
                    if time.time() > deadline - 5.0:
                        break
                    for j in range(n):
                        if j in best:
                            continue
                        if nums[j] >= nums[i]:
                            continue
                        inter = conflict[j] & best
                        if i in inter:
                            if len(inter) > 1:
                                continue
                        else:
                            if len(inter) > 0:
                                continue
                        gain = weights[j] - weights[i]
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap = (i, j)
                            
                if best_swap is not None:
                    i, j = best_swap
                    best.remove(i)
                    best.add(j)
                    best_score += best_swap_gain
                    improved = True
            
            if not improved:
                break
        return best, best_score

    def greedy_construct(exclude_indices):
        current_set = set()
        for i in range(n):
            if i in exclude_indices:
                continue
            if is_valid_add(current_set, i):
                current_set.add(i)
        return current_set

    best_set = set()
    best_score = 0.0
    
    # 1. Standard Greedy
    init_set = greedy_construct(set())
    ls_set, ls_score = local_search(init_set, max_iters=5)
    if ls_score > best_score:
        best_set = ls_set
        best_score = ls_score

    # 2. Exclusion Prefixes
    for p in range(1, 11):
        if time.time() > deadline - 20.0:
            break
        exclude = set(range(p))
        init_set = greedy_construct(exclude)
        ls_set, ls_score = local_search(init_set, max_iters=5)
        if ls_score > best_score:
            best_set = ls_set
            best_score = ls_score

    # 3. Branch and Bound
    # Order nodes by weight descending to find good solutions early and prune better?
    # Or keep index order for bitset simplicity?
    # Let's stick to index order for bitset masks, but we can sort the "available" set?
    # Actually, standard B&B with bitsets works well if we just iterate.
    
    # We will use a recursive search with a stack to avoid recursion limit issues
    # State: (idx, available_mask, current_score, current_set_list)
    # We want to maximize current_score + sum(weights[j] for j in available_mask)
    
    # Precompute suffix sums of weights? No, available mask changes.
    # We can approximate upper bound by sum of weights in available_mask.
    # To make this fast, we can precompute weight sums for masks? Impossible (2^255).
    # So we must calculate sum(weights[j] for j in available_mask) on the fly.
    # This is O(n) per node. With 255 nodes, and potentially many nodes visited, this might be slow.
    
    # Optimization: Only visit nodes if the upper bound is promising.
    # We can use a greedy upper bound: 
    # Greedily pick the highest weight nodes from available_mask that don't conflict.
    # This is O(k^2) where k is size of available.
    
    def greedy_upper_bound(available_mask):
        # Returns sum of weights of a greedy independent set from available_mask
        # This is a valid upper bound? No, it's a lower bound on the max weight independent set of the subgraph.
        # We need an UPPER bound.
        # The sum of all weights in available_mask is a valid upper bound.
        # Let's use that. It's O(n) to compute.
        s = 0.0
        m = available_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += weights[idx]
            m ^= lsb
        return s

    # To speed up sum calculation, we can keep a running sum?
    # When we remove a node, we subtract its weight.
    
    import sys
    sys.setrecursionlimit(10000)
    
    best_bb_score = best_score
    best_bb_set = list(best_set)
    
    # We will use a stack-based DFS
    # (idx, available_mask, current_score, current_set)
    # We process nodes in increasing index order.
    # If we include i, we remove i and its neighbors from available.
    # If we exclude i, we just remove i.
    
    # To prune, we need sum(weights in available_mask).
    # Let's store the current sum of available weights in the state.
    
    def branch_and_bound():
        nonlocal best_bb_score, best_bb_set
        start_time = time.time()
        
        # Initial state: all nodes available, score 0
        # But we can start with the heuristic solution?
        # No, B&B searches for the global optimum.
        # We use the heuristic score as the initial lower bound (cutoff).
        
        stack = []
        # (idx, available_mask, current_score, current_set, sum_available_weights)
        # idx is the next index to consider.
        # available_mask contains indices >= idx that are compatible with current_set.
        
        # Initialize
        full_mask = (1 << n) - 1
        # We don't have a running sum for full mask yet.
        total_weight = sum(weights)
        stack.append((0, full_mask, 0.0, set(), total_weight))
        
        while stack:
            if time.time() > deadline - 2.0:
                break
            
            idx, avail_mask, curr_score, curr_set, sum_avail = stack.pop()
            
            if idx >= n:
                if curr_score > best_bb_score:
                    best_bb_score = curr_score
                    best_bb_set = list(curr_set)
                continue
            
            # Pruning
            if curr_score + sum_avail <= best_bb_score:
                continue
            
            # Try to include idx if it is in avail_mask
            if (avail_mask >> idx) & 1:
                # Include idx
                new_score = curr_score + weights[idx]
                new_set = curr_set | {idx}
                # Remove idx and its conflicts from avail_mask
                # Conflicts are in conflict_bits[idx]
                # We also need to mask out indices < idx? 
                # No, avail_mask already only contains indices >= idx?
                # Let's ensure avail_mask only has bits >= idx.
                # In the initialization, it's full.
                # When we pop idx, the next idx is idx+1.
                # So we should mask out idx and all < idx from avail_mask before pushing?
                # Actually, if we push (idx+1, ...), the avail_mask should not contain idx or < idx.
                
                # Mask for indices > idx
                mask_gt_idx = full_mask ^ ((1 << (idx + 1)) - 1)
                
                # New available: (avail_mask & ~conflict_bits[idx] & ~(1 << idx)) & mask_gt_idx
                new_avail = (avail_mask & ~conflict_bits[idx] & ~(1 << idx)) & mask_gt_idx
                
                # Update sum_avail
                # We need to subtract weights of removed nodes.
                # Removed: idx and conflicts[idx] that were in avail_mask.
                # This is expensive to calculate exactly.
                # Approximation: Just calculate sum of new_avail? O(n).
                # Let's try to do it efficiently.
                # We can just recalculate sum_avail for the new state if the depth is small?
                # Or keep it simple: recalculate sum_avail in the loop if we store it?
                # Let's just recalculate sum_avail for the child.
                
                new_sum_avail = 0.0
                m = new_avail
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    new_sum_avail += weights[j]
                    m ^= lsb
                
                stack.append((idx + 1, new_avail, new_score, new_set, new_sum_avail))
            
            # Try to exclude idx
            # New available: avail_mask without idx
            new_avail_excl = (avail_mask & ~(1 << idx)) & (full_mask ^ ((1 << (idx + 1)) - 1))
            
            # Update sum_avail
            # Subtract weights[idx] if it was in avail
            new_sum_avail_excl = sum_avail
            if (avail_mask >> idx) & 1:
                new_sum_avail_excl -= weights[idx]
                
            stack.append((idx + 1, new_avail_excl, curr_score, curr_set, new_sum_avail_excl))
            
    # Run B&B
    branch_and_bound()
    
    if best_bb_score > best_score:
        best_set = set(best_bb_set)
        best_score = best_bb_score
        
    return [nums[i] for i in best_set]
# EVOLVE-BLOCK-END
