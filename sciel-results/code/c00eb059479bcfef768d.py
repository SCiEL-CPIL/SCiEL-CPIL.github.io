import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    # Keep a hard safety margin; evaluator kills at 150s.
    deadline = start_time + 140.0

    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    n_subsets = len(all_subsets)

    # For each block (tuple of k points), precompute the list of t-subset indices it covers
    # We'll build this on the fly as we generate blocks, but we need a helper.
    
    def block_triples(block_tuple):
        """Return list of indices of t-subsets contained in block_tuple."""
        return [subset_to_idx[sub] for sub in itertools.combinations(block_tuple, t)]

    def greedy_cover(seed_val):
        """
        Fast greedy covering. Returns (blocks_list, coverage_counts_array).
        coverage_counts_array[i] = number of blocks covering all_subsets[i].
        """
        rng = random.Random(seed_val)
        uncovered = [1] * n_subsets  # 1 if uncovered, 0 if covered
        blocks = []
        
        while any(uncovered):
            # Pick a random uncovered subset
            # To speed up, maintain a list of uncovered indices? 
            # For v=24, n_subsets=2024. Scanning is cheap.
            uncov_indices = [i for i, u in enumerate(uncovered) if u == 1]
            if not uncov_indices:
                break
            target_idx = rng.choice(uncov_indices)
            target = all_subsets[target_idx]
            
            block = list(target)
            rng.shuffle(block) # randomize order of initial points? No, block must contain target.
            # Actually, let's just start with target and add points.
            
            # To make it faster, we can pre-compute potential gains for each candidate point.
            # But the set of covered subsets changes as we add points.
            
            while len(block) < k:
                best_gain = -1
                best_point = -1
                # Candidates
                candidates = [x for x in range(v) if x not in block]
                rng.shuffle(candidates)
                
                # For efficiency, we can precompute the (t-1) subsets of the current block
                block_set = set(block)
                block_t1_subs = list(itertools.combinations(block, t - 1))
                
                for x in candidates:
                    gain = 0
                    for rest in block_t1_subs:
                        # New subset: rest + (x,)
                        # Must be sorted to match all_subsets format
                        new_sub = tuple(sorted(rest + (x,)))
                        idx = subset_to_idx.get(new_sub, -1)
                        if idx != -1 and uncovered[idx] == 1:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        if best_gain == len(block_t1_subs):
                            break # Max possible gain achieved
                
                if best_point == -1:
                    # Should not happen if uncovered exists, but fallback
                    best_point = candidates[0] if candidates else 0
                    if best_point in block:
                         # find any not in block
                         for x in range(v):
                             if x not in block:
                                 best_point = x
                                 break
                
                block.append(best_point)
                block.sort()
                
                # Update uncovered
                for sub in itertools.combinations(block, t):
                    idx = subset_to_idx[sub]
                    uncovered[idx] = 0

            blocks.append(block)
        
        # Compute coverage counts for the final set of blocks
        coverage_counts = [0] * n_subsets
        for b in blocks:
            for idx in block_triples(tuple(b)):
                coverage_counts[idx] += 1
        
        return blocks, coverage_counts

    def local_search(blocks, coverage_counts, time_limit, seed_val):
        """
        Simulated annealing / greedy local search to minimize block count.
        Returns (best_blocks, best_size).
        """
        rng = random.Random(seed_val)
        blocks = [list(b) for b in blocks]
        coverage_counts = list(coverage_counts)
        n_blocks = len(blocks)
        
        best_blocks = [list(b) for b in blocks]
        best_size = n_blocks
        
        # Precompute block triples indices for current blocks
        block_indices = [block_triples(tuple(b)) for b in blocks]
        
        # Temperature schedule
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999
        
        # We want to remove blocks.
        # A move: remove block i if coverage_counts[j] > 1 for all j in block_indices[i].
        # If we remove a block that makes some count 0, it's invalid.
        # We can use a "soft" penalty or just reject invalid moves.
        # To escape local minima where no single block can be removed, we might need to 
        # swap a block with a better one, or add/remove pairs.
        # Simpler: try to remove. If stuck, try to swap one block with a random new block 
        # that covers more "tight" subsets? 
        # Or just accept invalid moves with penalty? 
        # Let's stick to valid moves first: remove if safe.
        # To improve, we can also try to replace a block with a "better" block (covers more uncovered? no, all covered).
        # Actually, if all are covered, "better" means it allows more other blocks to be removed.
        
        # Strategy:
        # 1. Try to remove a random block. If valid, accept.
        # 2. If invalid, calculate "cost" (number of subsets that would become uncovered).
        #    Accept with probability exp(-cost / T) if we allow temporary invalidity?
        #    No, the evaluator requires valid. So we must maintain validity.
        #    Therefore, we can only accept moves that keep validity.
        #    This is a steepest descent / hill climbing on the space of valid coverings.
        #    To escape local minima, we need more complex moves (e.g., swap 2 blocks, or add 1 remove 1).
        
        # Let's implement:
        # Move type 1: Remove block i. Valid if min(coverage_counts[j] for j in block_indices[i]) > 1.
        # Move type 2: Swap block i with a new random block B. 
        #   Valid if:
        #   a) For all j in block_indices[i] \ block_triples(B), coverage_counts[j] - 1 > 0 (i.e., >= 1 after removal)
        #      Actually, if j is in i but not B, count decreases by 1. Must remain >= 1.
        #   b) For all j in block_triples(B) \ block_indices[i], count increases. No constraint.
        #   c) For all j in intersection, count unchanged.
        #   So condition: min(coverage_counts[j] for j in block_indices[i] if j not in set(block_triples(B))) > 1
        #   This is expensive to check for a random B.
        
        # Given time constraints, let's focus on fast removals and maybe simple swaps if time permits.
        # The seed program score was 1.0588 (approx 75-76 blocks?). 
        # Greedy usually gives ~78-80. Local search can shave off a few.
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            if n_blocks == 0:
                break
                
            move_type = rng.random()
            
            if move_type < 0.7:
                # Try removal
                i = rng.randrange(n_blocks)
                # Check validity
                valid = True
                for j in block_indices[i]:
                    if coverage_counts[j] == 1:
                        valid = False
                        break
                if valid:
                    # Apply removal
                    for j in block_indices[i]:
                        coverage_counts[j] -= 1
                    blocks.pop(i)
                    block_indices.pop(i)
                    n_blocks -= 1
                    if n_blocks < best_size:
                        best_size = n_blocks
                        best_blocks = [list(b) for b in blocks]
                else:
                    # Optional: accept invalid with penalty? No, must stay valid.
                    pass
            else:
                # Try swap: replace block i with a random block B
                i = rng.randrange(n_blocks)
                # Generate a random block B
                B = tuple(sorted(rng.sample(range(v), k)))
                B_indices = block_triples(B)
                
                # Check if swap is valid
                # Subsets in block_indices[i] but not in B_indices must have count > 1
                # We can optimize by checking counts directly.
                # Create a set of B_indices for fast lookup? B has C(7,3)=35 indices.
                B_set = set(B_indices)
                
                valid = True
                for j in block_indices[i]:
                    if j not in B_set:
                        if coverage_counts[j] == 1:
                            valid = False
                            break
                
                if valid:
                    # Apply swap
                    # Remove i
                    for j in block_indices[i]:
                        coverage_counts[j] -= 1
                    # Add B
                    for j in B_indices:
                        coverage_counts[j] += 1
                    
                    blocks[i] = list(B)
                    block_indices[i] = B_indices
                    # n_blocks unchanged
                    # Note: swap doesn't reduce size, but might enable future removals.
                    # We don't update best_size here unless we remove later.
                else:
                    pass
            
            # Cooling
            T *= cooling_rate
            if T < T_min:
                T = T_min

        return best_blocks, best_size

    # Main loop
    best_blocks = None
    best_size = float('inf')
    
    # Phase 1: Run several greedy restarts, keep the best
    # Phase 2: Run local search on the best greedy solution, and on other good ones if time permits.
    
    num_greedy = 5
    greedy_results = []
    
    for i in range(num_greedy):
        if time.time() > deadline - 20: # Leave time for local search
            break
        blocks, counts = greedy_cover(i)
        greedy_results.append((len(blocks), blocks, counts))
        
    # Sort by size
    greedy_results.sort(key=lambda x: x[0])
    
    # Local search on the top few
    for size, blocks, counts in greedy_results[:3]:
        if time.time() > deadline - 5:
            break
        
        # Allocate time for local search
        remaining = deadline - time.time() - 2
        if remaining < 5:
            continue
            
        # We can split remaining time among the candidates or just do one.
        # Let's do one long run on the best.
        ls_blocks, ls_size = local_search(blocks, counts, remaining, seed_val=1000)
        
        if ls_size < best_size:
            best_size = ls_size
            best_blocks = ls_blocks
            
        # If we have time, try local search on the second best?
        # For now, just take the best result from this loop.
        
    # If local search didn't run or failed, use best greedy
    if best_blocks is None:
        if greedy_results:
            best_blocks = greedy_results[0][1]
            best_size = greedy_results[0][0]
        else:
            # Fallback single greedy
            best_blocks, _ = greedy_cover(0)
            best_size = len(best_blocks)
            
    return best_blocks
