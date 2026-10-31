import itertools
import random
import time

def construct(v, k, t):
    # 1. Fast Greedy Initialization
    def greedy_cover():
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        # Precompute all t-subsets for each potential point addition? 
        # Or just iterate.
        
        # To make it deterministic and fast, we sort the uncovered set
        # But we want diversity, so we shuffle the "order" of consideration or just pick best.
        # Standard Greedy: Pick block that covers max uncovered.
        # Since we generate blocks, we pick the uncovered subset that is "best" to start with.
        # Heuristic: Pick random uncovered subset.
        
        while uncovered:
            # Pick a random uncovered subset to start
            start_subset = random.choice(list(uncovered))
            block = list(start_subset)
            
            # Extend block to size k
            while len(block) < k:
                best_point = None
                best_gain = -1
                # Calculate gain for each candidate point
                # Gain = number of new t-subsets covered
                # A new t-subset is formed by (t-1) elements from current block + new point
                current_t_minus_1 = list(itertools.combinations(block, t - 1))
                
                # Optimization: Instead of checking all v points, 
                # we can just check all points. v=19 is small.
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in current_t_minus_1:
                        new_subset = tuple(sorted(rest + (x,)))
                        if new_subset in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        
                if best_point is None or best_gain == 0:
                    # No gain, pick random
                    candidates = [x for x in range(v) if x not in block]
                    if candidates:
                        best_point = random.choice(candidates)
                    else:
                        break
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
                
        return blocks

    # 2. Optimizer with 2-for-1 Exchange
    def optimize(blocks, time_limit=115):
        current_blocks = [tuple(sorted(b)) for b in blocks]
        
        # Build coverage count
        def calc_coverage(blist):
            cov = {}
            for b in blist:
                for sub in itertools.combinations(b, t):
                    cov[sub] = cov.get(sub, 0) + 1
            return cov

        coverage_count = calc_coverage(current_blocks)
        
        def get_critical_subsets(b, cov):
            """Returns list of t-subsets of b that are covered only by b"""
            crits = []
            for sub in itertools.combinations(b, t):
                if cov[sub] == 1:
                    crits.append(sub)
            return crits

        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            improved = False
            
            # Phase 1: Delete redundant blocks
            i = 0
            while i < len(current_blocks):
                b = current_blocks[i]
                is_red = True
                for sub in itertools.combinations(b, t):
                    if coverage_count[sub] == 1:
                        is_red = False
                        break
                if is_red:
                    for sub in itertools.combinations(b, t):
                        coverage_count[sub] -= 1
                    current_blocks.pop(i)
                    improved = True
                    # Don't increment i
                else:
                    i += 1
                    
            if improved:
                continue # Restart loop

            # Phase 2: 2-for-1 Exchange
            # For each pair of blocks, check if they can be merged
            # To avoid O(N^2) being too slow, we limit checks or use heuristics
            # N is ~150. N^2 = 22500. 
            # For each pair, we compute P = Union of points in critical subsets.
            # This is fast.
            
            # We iterate through all pairs.
            n = len(current_blocks)
            found_exchange = False
            
            # Shuffle indices to randomize which pair we find first
            indices = list(range(n))
            random.shuffle(indices)
            
            for i_idx in range(n):
                if found_exchange: break
                i = indices[i_idx]
                b1 = current_blocks[i]
                crits1 = get_critical_subsets(b1, coverage_count)
                
                # If b1 has no critical subsets, it's already redundant (handled in Phase 1)
                # So crits1 is non-empty here.
                
                for j_idx in range(i_idx + 1, n):
                    j = indices[j_idx]
                    b2 = current_blocks[j]
                    crits2 = get_critical_subsets(b2, coverage_count)
                    
                    # Combine critical subsets
                    all_crits = crits1 + crits2
                    
                    # Find union of points in all_crits
                    P = set()
                    for sub in all_crits:
                        P.update(sub)
                    
                    if len(P) > k:
                        continue
                        
                    # Construct candidate block
                    # P must be subset of new block
                    # Add random points to reach size k
                    remaining = list(set(range(v)) - P)
                    random.shuffle(remaining)
                    
                    # Try a few random completions
                    for _ in range(10):
                        if len(remaining) < k - len(P):
                            break
                        extra = remaining[:k - len(P)]
                        new_block = tuple(sorted(list(P) + extra))
                        
                        # Check if new_block is valid (distinct from existing)
                        # And check if it covers all all_crits
                        # It covers all_crits by construction (contains P)
                        
                        # We need to ensure that removing b1 and b2 doesn't uncover anything else
                        # We already checked that crits1 and crits2 are the ONLY unique subsets.
                        # So if new_block covers crits1 U crits2, the move is valid.
                        
                        # However, we must ensure new_block is not already in current_blocks
                        if new_block in current_blocks:
                            continue
                            
                        # Perform exchange
                        # Remove b1, b2
                        for sub in itertools.combinations(b1, t):
                            coverage_count[sub] -= 1
                        for sub in itertools.combinations(b2, t):
                            coverage_count[sub] -= 1
                            
                        # Add new_block
                        for sub in itertools.combinations(new_block, t):
                            coverage_count[sub] = coverage_count.get(sub, 0) + 1
                            
                        # Replace b1, b2 with new_block
                        # Remove indices j and i (j > i)
                        current_blocks.pop(j)
                        current_blocks.pop(i)
                        current_blocks.append(new_block)
                        
                        improved = True
                        found_exchange = True
                        break
                        
                    if found_exchange:
                        break
                        
            if not improved:
                # No deletion, no exchange found in this pass
                # Try random swaps to escape local minima?
                # Or just break if time is low
                if time.time() - start_time > time_limit - 5:
                    break
                
                # Random Swap
                if len(current_blocks) > 1:
                    idx = random.randint(0, len(current_blocks) - 1)
                    old_block = current_blocks[idx]
                    crits_old = get_critical_subsets(old_block, coverage_count)
                    
                    # Try to find a random block that covers crits_old
                    # Generate random blocks until one covers crits_old
                    found_swap = False
                    for _ in range(50):
                        new_block = tuple(sorted(random.sample(range(v), k)))
                        # Check if new_block covers all crits_old
                        valid = True
                        for sub in crits_old:
                            if not all(x in new_block for x in sub):
                                valid = False
                                break
                        if valid and new_block != old_block:
                            # Check distinctness
                            if new_block in current_blocks:
                                continue
                            
                            # Perform swap
                            for sub in itertools.combinations(old_block, t):
                                coverage_count[sub] -= 1
                            for sub in itertools.combinations(new_block, t):
                                coverage_count[sub] = coverage_count.get(sub, 0) + 1
                            current_blocks[idx] = new_block
                            improved = True
                            found_swap = True
                            break
                            
            if not improved:
                # Stuck
                # If we are stuck, we might need to restart or accept
                # To avoid infinite loop, we break if no improvement for a long time
                # Or we can try a "kick" (add a random block, then delete)
                # For now, break if time is almost up
                if time.time() - start_time > time_limit - 2:
                    break

        return [list(b) for b in current_blocks]

    # Main Execution
    best_initial = None
    # Run greedy a few times to get a good start
    for _ in range(5):
        initial_blocks = greedy_cover()
        if best_initial is None or len(initial_blocks) < len(best_initial):
            best_initial = initial_blocks
            
    optimized_blocks = optimize(best_initial, time_limit=115)
    
    return optimized_blocks
