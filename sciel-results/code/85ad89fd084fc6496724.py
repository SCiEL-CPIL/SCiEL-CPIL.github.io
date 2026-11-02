import itertools
import random
import time

def construct(v, k, t):
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    m = len(all_t_subsets)
    
    best_blocks = None
    best_size = float('inf')
    
    # Time limit management
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # Helper to get t-subsets of a block
    def get_block_t_subsets(block):
        return list(itertools.combinations(block, t))
        
    # Helper to check if a block covers a list of t-subsets
    def block_covers_subsets(block, subsets):
        block_set = set(block)
        for s in subsets:
            for x in s:
                if x not in block_set:
                    return False
        return True

    # Greedy Construction: Max Coverage Set Cover
    def greedy_construct():
        uncovered = [1] * m
        blocks = []
        
        # Precompute t-subsets for potential blocks? No, too many.
        # Instead, iteratively pick the block that covers the most currently uncovered subsets.
        # To speed up, we can use a heuristic: start with a random seed and grow it, 
        # but the "Max Coverage" greedy is standard.
        
        # Optimization: 
        # 1. Start with a random k-subset.
        # 2. Calculate coverage.
        # 3. Pick the next block that maximizes new coverage.
        # To make it faster, we can restrict candidates to blocks that share many elements with existing blocks or are random.
        
        # Let's use a randomized greedy:
        # At each step, generate a set of candidate blocks (e.g., 50 random k-subsets).
        # Pick the one with the highest marginal gain.
        
        # Even better: The "seed and grow" approach in the original code is actually quite fast 
        # if optimized. The bottleneck is the inner loop of finding the best x to add.
        # Let's stick to a robust greedy that is known to work well:
        # 1. While uncovered > 0:
        #    a. Pick a random uncovered subset S.
        #    b. Try to find a block B containing S that covers max new subsets.
        #       - Generate candidates: Random k-sets containing S.
        #       - Pick best.
        #    c. Add B, update uncovered.
        
        while True:
            min_cov_val = min(uncovered)
            if min_cov_val > 0:
                break
            
            # Pick a random uncovered subset
            candidates_idx = [i for i in range(m) if uncovered[i] == 1]
            if not candidates_idx:
                break
            idx = random.choice(candidates_idx)
            seed_subset = all_t_subsets[idx]
            
            # Generate candidate blocks containing seed_subset
            best_block = None
            best_gain = -1
            
            # Number of candidates to try
            num_candidates = 20
            
            # Pre-compute points not in seed
            remaining_points = [x for x in range(v) if x not in seed_subset]
            
            for _ in range(num_candidates):
                # Randomly pick k-t points from remaining
                k_t = k - t
                if k_t > len(remaining_points):
                    continue
                extra = random.sample(remaining_points, k_t)
                block = list(seed_subset) + extra
                block.sort()
                
                # Calculate gain
                gain = 0
                for t_sub in itertools.combinations(block, t):
                    i = subset_to_idx[t_sub]
                    if uncovered[i] == 1:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is None:
                # Fallback: random block
                best_block = sorted(random.sample(range(v), k))
            
            # Update coverage
            for t_sub in itertools.combinations(best_block, t):
                i = subset_to_idx[t_sub]
                uncovered[i] -= 1
                
            blocks.append(best_block)
            
        return blocks, uncovered

    # Local Search: Remove Redundant + Swap (Improved)
    def local_search(blocks, cov):
        changed = True
        while changed:
            changed = False
            
            # 1. Remove redundant blocks
            i = 0
            while i < len(blocks):
                block = blocks[i]
                is_essential = False
                for t_sub in itertools.combinations(block, t):
                    idx = subset_to_idx[t_sub]
                    if cov[idx] == 1:
                        is_essential = True
                        break
                
                if not is_essential:
                    # Remove block
                    for t_sub in itertools.combinations(block, t):
                        idx = subset_to_idx[t_sub]
                        cov[idx] -= 1
                    blocks.pop(i)
                    changed = True
                else:
                    i += 1
            
            if not changed:
                # 2. Try Swaps
                # For each block, if it has unique subsets, try to replace it
                # with a block that covers those unique subsets and maybe more.
                # To avoid O(N^2) checks, we sample.
                
                # Shuffle blocks to randomize order
                indices = list(range(len(blocks)))
                random.shuffle(indices)
                
                for i in indices:
                    if time.time() - start_time > time_limit:
                        return blocks, cov
                    
                    B = blocks[i]
                    
                    # Find critical (unique) subsets
                    critical = []
                    for t_sub in itertools.combinations(B, t):
                        idx = subset_to_idx[t_sub]
                        if cov[idx] == 1:
                            critical.append(t_sub)
                    
                    if not critical:
                        continue # Should have been removed, but safety check
                        
                    # Union of critical subsets
                    crit_union_set = set()
                    for s in critical:
                        crit_union_set.update(s)
                    
                    if len(crit_union_set) > k:
                        continue # Cannot cover all critical subsets with one block
                    
                    # Try to find a replacement B'
                    # B' must contain crit_union_set
                    # We can generate random B' containing crit_union_set
                    
                    # If crit_union_set size is k, B' is fixed
                    # If < k, we have freedom
                    
                    found_swap = False
                    # Try a few candidates
                    for _ in range(10):
                        if len(crit_union_set) == k:
                            B_prime = sorted(crit_union_set)
                        else:
                            remaining = [x for x in range(v) if x not in crit_union_set]
                            need = k - len(crit_union_set)
                            extra = random.sample(remaining, need)
                            B_prime = sorted(list(crit_union_set) + extra)
                        
                        # Check if B_prime is already in blocks
                        # Optimization: Use a set of tuples for O(1) lookup
                        # But blocks is a list. For small N, linear is ok.
                        # Let's create a set of tuples for current blocks to speed up duplicate check?
                        # It's expensive to maintain. Let's just check.
                        
                        if B_prime in blocks:
                            continue
                            
                        # Check if B_prime covers all critical subsets (it does by construction)
                        # But we must ensure it doesn't "uncover" anything else?
                        # No, we are replacing B with B'.
                        # The critical subsets were ONLY covered by B.
                        # So B' MUST cover them. It does.
                        # Any other subsets covered by B but not B' will still be covered by other blocks?
                        # Wait. If a subset S is covered by B and another block B2, and we remove B, S is still covered by B2.
                        # If S is covered ONLY by B, it is in 'critical'.
                        # So if B' covers 'critical', the coverage remains valid.
                        
                        # So the swap is always valid if B' contains crit_union_set.
                        
                        # Update cov
                        for t_sub in itertools.combinations(B, t):
                            idx = subset_to_idx[t_sub]
                            cov[idx] -= 1
                        for t_sub in itertools.combinations(B_prime, t):
                            idx = subset_to_idx[t_sub]
                            cov[idx] += 1
                            
                        blocks[i] = B_prime
                        changed = True
                        found_swap = True
                        break # Restart loop from beginning to catch new redundancies
                    
                    if found_swap:
                        break
                        
        return blocks, cov

    # Main Loop
    for _ in range(200):
        if time.time() - start_time > time_limit:
            break
        
        random.seed()
        
        blocks, cov = greedy_construct()
        blocks, cov = local_search(blocks, cov)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks[:]
            
    return best_blocks
