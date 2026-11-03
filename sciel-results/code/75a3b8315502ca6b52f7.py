import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin
    
    # Precomputation
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Map t-subset to its index
    # For speed, we represent a t-subset as a frozenset or sorted tuple for lookup
    # But we need to know which blocks cover which t-subsets.
    # Instead of listing all C(19,8) blocks (75582), we can generate them on the fly 
    # or precompute a mapping from t-subset to list of blocks covering it.
    # C(19,8) is small enough to precompute all blocks and their covered t-subsets.
    
    all_blocks = list(itertools.combinations(points, k))
    N_b = len(all_blocks)
    
    # block_cover[bi] = list of t-indices covered by block bi
    block_cover = []
    # t_to_blocks[ti] = list of block indices covering t-subset ti
    t_to_blocks = [[] for _ in range(N_t)]
    
    # Precompute block covers
    # Optimization: Use numpy for faster intersection checks later?
    # For now, lists are fine for 75k blocks * 56 t-subsets = ~4.2M entries.
    
    for bi, b in enumerate(all_blocks):
        covered = []
        for ts in itertools.combinations(b, t):
            ti = t_to_idx[ts]
            covered.append(ti)
        block_cover.append(covered)
        for ti in covered:
            t_to_blocks[ti].append(bi)

    def get_uncovered_counts(blocks):
        """
        Returns an array of counts for each t-subset.
        """
        counts = np.zeros(N_t, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                counts[ti] += 1
        return counts

    def is_valid(blocks):
        """
        Check if all t-subsets are covered.
        """
        counts = get_uncovered_counts(blocks)
        return np.all(counts > 0)

    def prune_redundant(blocks):
        """
        Remove blocks that do not uniquely cover any t-subset.
        """
        if not blocks:
            return []
        
        counts = get_uncovered_counts(blocks)
        current_blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for bi in current_blocks:
                is_redundant = True
                for ti in block_cover[bi]:
                    if counts[ti] == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    new_blocks.append(bi)
                else:
                    for ti in block_cover[bi]:
                        counts[ti] -= 1
                    changed = True
            current_blocks = new_blocks
        return current_blocks

    def stochastic_local_search(initial_blocks, rng, time_limit):
        """
        SLK: Stochastic Local Search with Kicker.
        Moves:
        1. Drop a redundant block.
        2. Swap a block with another block that covers its unique t-subsets 
           and potentially allows further drops.
        3. Random restarts if stuck? No, keep simple.
        """
        current_blocks = list(initial_blocks)
        counts = get_uncovered_counts(current_blocks)
        
        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        
        # Precompute a set for fast "in block" checks? 
        # block_cover[bi] is a list. Converting to set for membership is O(56).
        # We can maintain a list of sets for current blocks? 
        # Let's just use lists and linear scan if necessary, or convert on the fly.
        # 75k blocks, 56 t-subsets. 
        # For swap logic, we need to find candidates.
        
        while time.time() < time_limit:
            # 1. Try to drop redundant blocks
            changed = False
            new_blocks = []
            for bi in current_blocks:
                is_redundant = True
                for ti in block_cover[bi]:
                    if counts[ti] == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    new_blocks.append(bi)
                else:
                    for ti in block_cover[bi]:
                        counts[ti] -= 1
                    changed = True
            current_blocks = new_blocks
            if changed:
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = list(current_blocks)
                continue # Re-check drops before swapping

            # 2. Try a swap
            if not current_blocks:
                break
                
            # Pick a random block to potentially swap out
            idx = rng.randrange(len(current_blocks))
            bi_out = current_blocks[idx]
            
            # Find t-subsets uniquely covered by bi_out
            unique_ts = [ti for ti in block_cover[bi_out] if counts[ti] == 1]
            
            if not unique_ts:
                # Should have been dropped, but just in case
                continue
                
            # Find candidate blocks that cover ALL unique_ts
            # Start with candidates for the first unique t
            candidates = set(t_to_blocks[unique_ts[0]])
            for ti in unique_ts[1:]:
                candidates &= set(t_to_blocks[ti])
            
            candidates.discard(bi_out)
            
            if not candidates:
                continue
                
            # Score candidates
            # We want a candidate that:
            # 1. Covers the unique_ts (guaranteed by intersection)
            # 2. Maximizes the number of "safe" t-subsets it covers (count > 1) 
            #    OR minimizes the "damage" (t-subsets with count=1 that it covers 
            #    but were NOT unique to bi_out).
            
            # Let's use a score: 
            # Gain = number of t in candidate where count > 1 (safe to add)
            # Loss = number of t in candidate where count == 1 AND t is NOT in unique_ts 
            #       (these become count=2, so the other block loses uniqueness? 
            #        No, if count was 1 and we add candidate, count becomes 2.
            #        The block that uniquely covered t now has count 2 for t.
            #        This might make that other block redundant later. This is GOOD.
            #        Wait.
            #        If t has count 1, it is covered by exactly one block B_other.
            #        If we add candidate C which also covers t, count[t] becomes 2.
            #        B_other no longer uniquely covers t.
            #        This is a GOOD thing for potential drops.
            #        So we WANT to cover t-subsets with count 1 (other than the ones we are replacing? 
            #        Actually, the ones we are replacing (unique_ts) MUST be covered by C.
            #        For those, count goes 1 -> 2 (if we remove bi_out first? 
            #        Let's trace:
            #        Remove bi_out: count[unique_ts] becomes 0.
            #        Add C: count[unique_ts] becomes 1 (if C covers them, which it does).
            #        So for unique_ts, count remains 1.
            #        For other t in C:
            #           If count[t] was 1 (covered by B_other), now it becomes 2.
            #           B_other loses uniqueness. Good.
            #           If count[t] was > 1, now it becomes > 2. Neutral.
            #        
            #        So the "score" should be: Number of t in C (excluding unique_ts) 
            #        such that count[t] == 1.
            #        This maximizes the chance of creating new redundancies.
            
            best_cand = -1
            best_score = -1
            bi_out_set = set(block_cover[bi_out]) # For quick check? 
            # Actually, we know unique_ts are in bi_out.
            # We want t in C such that count[t] == 1 and t NOT in unique_ts.
            # Note: If t is in bi_out but NOT in unique_ts, then count[t] > 1 (since unique_ts are count 1).
            # So if C covers t (which is in bi_out, count>1), it's safe.
            # If C covers t (NOT in bi_out, count==1), it's "good" (creates redundancy elsewhere).
            
            # Let's iterate candidates
            for c in candidates:
                score = 0
                for ti in block_cover[c]:
                    if counts[ti] == 1 and ti not in unique_ts:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_cand = c
            
            if best_cand == -1:
                continue
                
            # Perform Swap
            # Remove bi_out
            for ti in block_cover[bi_out]:
                counts[ti] -= 1
            # Add best_cand
            for ti in block_cover[best_cand]:
                counts[ti] += 1
            current_blocks[idx] = best_cand
            
            # Check for new drops immediately? 
            # The main loop handles drops at the start of the next iteration.
            
        return best_blocks

    def probabilistic_construction(rng, time_limit):
        """
        Construct a covering by iteratively adding blocks.
        Use a weighting scheme to guide selection.
        Start with empty set.
        While not all covered:
           1. Identify uncovered t-subsets.
           2. For each uncovered t, look at blocks covering it.
           3. Score blocks based on how many OTHER uncovered t's they cover.
           4. Select block with highest score (or weighted random).
        """
        uncovered_mask = np.ones(N_t, dtype=bool)
        num_uncovered = N_t
        selected = []
        
        # Precompute list of blocks for each t-subset for fast access
        # t_to_blocks is already computed.
        
        while num_uncovered > 0:
            # Pick a random uncovered t-subset to focus on
            # This avoids O(N_b) scan every time if we just pick candidates for one t.
            # However, standard greedy picks the block covering the MOST uncovered t's.
            # That requires scanning all blocks.
            # "Approximate Greedy" picks a random uncovered t, then best block covering it.
            
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
                
            ti_target = rng.choice(uncovered_indices.tolist())
            candidates = t_to_blocks[ti_target]
            
            best_block = -1
            best_gain = -1
            
            # Evaluate candidates
            for bi in candidates:
                gain = 0
                for ti in block_cover[bi]:
                    if uncovered_mask[ti]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
            
            if best_block == -1:
                # Should not happen if problem is solvable
                break
                
            selected.append(best_block)
            for ti in block_cover[best_block]:
                if uncovered_mask[ti]:
                    uncovered_mask[ti] = False
                    num_uncovered -= 1
                    
        return selected

    # Main Execution
    best_solution = None
    best_len = float('inf')
    
    # Phase 1: Multiple Probabilistic Greedy Starts
    # We run several instances of the greedy constructor with different random seeds.
    # The randomness comes from the choice of the "target" t-subset.
    
    num_starts = 20
    for i in range(num_starts):
        if time.time() > deadline - 40.0:
            break
        rng = random.Random(42 + i)
        blocks = probabilistic_construction(rng, deadline - 40.0)
        blocks = prune_redundant(blocks)
        
        if is_valid(blocks): # Should be valid by construction
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_solution = blocks
                
    # Phase 2: Local Search on the Best Solution
    if best_solution:
        # Use remaining time for local search
        remaining_time = deadline - 5.0
        if time.time() < remaining_time:
            rng = random.Random(999)
            optimized = stochastic_local_search(best_solution, rng, remaining_time)
            if len(optimized) < best_len:
                best_len = len(optimized)
                best_solution = optimized

    # Fallback if no solution found (should not happen)
    if not best_solution:
        rng = random.Random(1)
        best_solution = probabilistic_construction(rng, deadline - 10.0)
        best_solution = prune_redundant(best_solution)

    return [list(all_blocks[bi]) for bi in best_solution]
