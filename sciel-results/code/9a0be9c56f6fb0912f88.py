import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 118.0  # Safety margin for 120s limit (evaluator kills at 150s, but keep buffer)

    # 1. Precomputation
    # Points are 0..v-1
    all_t_subsets = list(itertools.combinations(range(v), t))
    N_t = len(all_t_subsets)
    
    # Map t-subset to index
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # All possible blocks (k-subsets)
    all_blocks = list(itertools.combinations(range(v), k))
    N_blocks = len(all_blocks)
    
    # For each block, which t-subsets does it cover?
    # block_cover[b] = list of indices of t-subsets contained in block b
    block_cover = []
    for b in all_blocks:
        covered = [t_to_idx[s] for s in itertools.combinations(b, t)]
        block_cover.append(covered)
        
    # Inverse index: for each t-subset, which blocks cover it?
    # This is crucial for fast greedy selection and pruning.
    t_to_blocks = [[] for _ in range(N_t)]
    for bi, covered in enumerate(block_cover):
        for ti in covered:
            t_to_blocks[ti].append(bi)
            
    # Convert block_cover to numpy arrays for faster iteration in inner loops?
    # Actually, lists of lists are often faster in Python for small lists than numpy overhead.
    # But we can use numpy for the count vector.

    def greedy_cover(seed_rng):
        """
        Advanced Greedy Set Cover.
        Uses a scoring function: score(b) = sum( 1 / (freq(t) + 1) ) for t in b
        where freq(t) is the current number of blocks covering t.
        This prioritizes covering rare subsets.
        """
        # counts[t] = number of selected blocks covering t
        counts = np.zeros(N_t, dtype=np.int32)
        selected_blocks = []
        uncovered_mask = np.ones(N_t, dtype=bool)
        num_uncovered = N_t
        
        # Precompute the list of t-subsets for each block for faster access
        # We already have block_cover as list of lists of indices.
        
        while num_uncovered > 0:
            best_block = -1
            best_score = -1
            
            # We cannot iterate over ALL blocks every time (75k * 56 ops * ~300 steps = too slow?)
            # 75k * 56 = 4.2M ops per step. 300 steps = 1.2B ops. Too slow in Python.
            
            # Optimization: Only consider blocks that cover at least one uncovered t-subset.
            # We can maintain a list of "active" blocks or just iterate and skip if gain is 0.
            # Better: Pick a random uncovered t-subset, and only consider blocks covering it?
            # No, that's "Randomized Greedy" which is weaker.
            # 
            # Let's try a hybrid: 
            # 1. Identify a small set of "critical" uncovered t-subsets.
            # 2. For each critical t, look at its candidate blocks.
            # 3. Score those candidates.
            
            # Find the most "urgent" uncovered t-subsets.
            # Urgency can be defined by how few blocks cover it globally (static) or currently (dynamic).
            # Dynamic is better.
            
            # To avoid O(N_blocks) scan every iteration, we can sample a few uncovered t-subsets
            # and evaluate blocks covering them.
            
            # Let's pick up to 5 random uncovered t-subsets
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
                
            # Sample up to 3 random uncovered t-subsets to guide the search
            sample_size = min(3, len(uncovered_indices))
            sampled_ts = seed_rng.sample(uncovered_indices.tolist(), sample_size)
            
            # Get candidate blocks for these sampled t-subsets
            candidate_blocks = set()
            for ti in sampled_ts:
                candidate_blocks.update(t_to_blocks[ti])
                
            # If the set of candidates is too large, we might miss the global best.
            # But for C(19,8,5), each t-subset is covered by C(14,3)=364 blocks.
            # Union of 3 is at most 3*364 = 1092. This is small enough to evaluate.
            
            # Evaluate all candidates
            for bi in candidate_blocks:
                # Calculate score: sum of 1/(counts[t]+1) for t in block_cover[bi] where t is uncovered
                # We only care about uncovered ones for the "gain", but the score formula 
                # usually uses the current frequency to balance.
                # Standard Greedy: Maximize |block ∩ Uncovered|
                # Weighted Greedy: Maximize sum_{t in block} w(t) where w(t) = 1/freq(t)
                
                score = 0.0
                for ti in block_cover[bi]:
                    if uncovered_mask[ti]:
                        # If it's uncovered, freq is 0. 1/(0+1) = 1.
                        # If we want to prioritize rare ones, we should look at global rarity?
                        # No, dynamic freq is standard.
                        # If counts[ti] is 0, it's uncovered.
                        # Let's just count the number of newly covered subsets.
                        score += 1.0
                        # To break ties, we can add a small epsilon based on global rarity
                        # Or just stick to simple gain for speed.
                        
                if score > best_score:
                    best_score = score
                    best_block = bi
                    
            if best_block == -1:
                # Fallback: pick any block covering a random uncovered t
                ti = seed_rng.choice(uncovered_indices.tolist())
                best_block = seed_rng.choice(t_to_blocks[ti])
                best_score = 1 # At least 1

            # Select the block
            selected_blocks.append(best_block)
            
            # Update counts and uncovered mask
            for ti in block_cover[best_block]:
                if uncovered_mask[ti]:
                    uncovered_mask[ti] = False
                    num_uncovered -= 1
                counts[ti] += 1
                
        return selected_blocks

    def prune_redundant(blocks):
        """
        Iteratively remove blocks that do not uniquely cover any t-subset.
        A block is redundant if for every t-subset it covers, there is another block in the set covering it.
        """
        # Convert to list of block indices
        current_blocks = list(blocks)
        if not current_blocks:
            return current_blocks
            
        # Calculate counts
        counts = np.zeros(N_t, dtype=np.int32)
        for bi in current_blocks:
            for ti in block_cover[bi]:
                counts[ti] += 1
                
        changed = True
        while changed:
            changed = False
            # Try to remove each block
            # To speed up, we can check blocks in random order or order of "least utility"
            # Utility = number of t-subsets it uniquely covers (counts[ti]==1)
            
            # Calculate unique coverage for each block
            unique_counts = {}
            for bi in current_blocks:
                uc = 0
                for ti in block_cover[bi]:
                    if counts[ti] == 1:
                        uc += 1
                unique_counts[bi] = uc
                
            # Sort blocks by unique coverage (lowest first)
            # We want to remove blocks with 0 unique coverage first.
            # If a block has 0 unique coverage, it is safe to remove.
            
            removable = [bi for bi in current_blocks if unique_counts[bi] == 0]
            
            if removable:
                # Remove all removable blocks at once? 
                # Careful: removing one might make another redundant.
                # But if a block has 0 unique coverage, removing it doesn't uncover anything.
                # So we can remove all of them.
                for bi in removable:
                    current_blocks.remove(bi)
                    for ti in block_cover[bi]:
                        counts[ti] -= 1
                changed = True
            else:
                # No block is currently redundant.
                # Can we swap? 
                # The pruning step is just removal. Swaps are handled by the search loop.
                break
                
        return current_blocks

    # Main Search Loop: Multi-Start with Pruning
    best_solution = None
    best_len = float('inf')
    
    rng = random.Random(42)
    
    # Phase 1: Fast Greedy restarts to find a good baseline
    # Run ~20 fast restarts
    num_fast_restarts = 20
    for i in range(num_fast_restarts):
        if time.time() > deadline - 5.0:
            break
        local_rng = random.Random(rng.randint(0, 1000000))
        blocks = greedy_cover(local_rng)
        blocks = prune_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_solution = blocks
            
    # Phase 2: Local Search / Swap Optimization on the best solution
    # We use a "Swap" strategy:
    # 1. Pick a block B from the current set.
    # 2. Identify t-subsets covered ONLY by B (unique coverage).
    # 3. If no unique coverage, remove B (pruning).
    # 4. If unique coverage exists, find a new block B' that covers all unique t-subsets of B.
    # 5. Replace B with B'. This maintains validity.
    # 6. After replacement, run pruning again to see if size decreases.
    
    if best_solution is not None:
        current_blocks = list(best_solution)
        
        # Calculate initial counts
        counts = np.zeros(N_t, dtype=np.int32)
        for bi in current_blocks:
            for ti in block_cover[bi]:
                counts[ti] += 1
                
        # Local Search Loop
        # We allow "neutral" moves (same size) to escape local minima, 
        # but we must prune after every move to try to reduce size.
        
        max_local_steps = 5000
        steps = 0
        T = 1.0 # Temperature for accepting neutral/worse structural moves?
        # Actually, let's stick to strict improvement or neutral moves that lead to pruning.
        
        while steps < max_local_steps and time.time() < deadline - 2.0:
            steps += 1
            
            # Pick a random block to try to swap
            if not current_blocks:
                break
                
            bi_idx = rng.randrange(len(current_blocks))
            bi = current_blocks[bi_idx]
            
            # Find unique t-subsets covered by bi
            unique_ts = []
            for ti in block_cover[bi]:
                if counts[ti] == 1:
                    unique_ts.append(ti)
                    
            if not unique_ts:
                # Redundant block! Remove it.
                current_blocks.pop(bi_idx)
                for ti in block_cover[bi]:
                    counts[ti] -= 1
                # This is an improvement (size -1).
                # Continue loop to prune more?
                # The loop will pick another block next.
                continue
                
            # We need to find a block B' that covers all t in unique_ts.
            # And B' != bi.
            # To make the swap useful, B' should ideally cover other "weakly" covered t-subsets 
            # or allow future pruning.
            
            # Intersection of blocks covering each t in unique_ts
            if len(unique_ts) == 1:
                candidates = set(t_to_blocks[unique_ts[0]])
            else:
                # Start with the smallest candidate set
                min_len = float('inf')
                min_ti = unique_ts[0]
                for ti in unique_ts:
                    if len(t_to_blocks[ti]) < min_len:
                        min_len = len(t_to_blocks[ti])
                        min_ti = ti
                        
                candidates = set(t_to_blocks[min_ti])
                for ti in unique_ts[1:]:
                    candidates &= set(t_to_blocks[ti])
                    
            candidates.discard(bi)
            
            if not candidates:
                # No swap possible for this block.
                continue
                
            # Choose a candidate
            # Strategy: Pick a candidate that covers the most "rare" (low count) t-subsets?
            # Or just random?
            # Let's try a few random candidates and pick the one that minimizes "redundancy score"
            # Redundancy score of a set S: sum(1/counts[t]) for t in S?
            # Simpler: Just pick one random candidate.
            
            new_bi = rng.choice(list(candidates))
            
            # Perform Swap
            # Remove bi
            for ti in block_cover[bi]:
                counts[ti] -= 1
            # Add new_bi
            for ti in block_cover[new_bi]:
                counts[ti] += 1
            # Update list
            current_blocks[bi_idx] = new_bi
            
            # After swap, try to prune
            # Check if any block is now redundant
            pruned = False
            # Quick check: did the swap make new_bi redundant? Or others?
            # Let's run a lightweight prune: check unique counts for all blocks?
            # That's O(N_blocks * 56).
            
            # Optimization: Only check blocks that might have become redundant.
            # Removing bi decreased counts. Adding new_bi increased counts.
            # Blocks that were redundant before might not be anymore (if they shared coverage with bi? No, bi was removed).
            # Blocks that were NOT redundant might become redundant if their unique coverage was taken by new_bi?
            # No, new_bi ADDS coverage. It doesn't take unique coverage away.
            # However, if new_bi covers a t-subset that was uniquely covered by some OTHER block C, 
            # then C is no longer uniquely covering that t-subset. C might become redundant if it has no other unique t-subsets.
            
            # So, we need to check blocks that share t-subsets with new_bi.
            affected_blocks = set()
            for ti in block_cover[new_bi]:
                affected_blocks.update(t_to_blocks[ti])
            
            # Filter to only those in current_blocks
            affected_blocks = [b for b in affected_blocks if b in current_blocks]
            
            for b in affected_blocks:
                # Check if b is redundant
                is_redundant = True
                for ti in block_cover[b]:
                    if counts[ti] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove b
                    current_blocks.remove(b)
                    for ti in block_cover[b]:
                        counts[ti] -= 1
                    pruned = True
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_solution = list(current_blocks)
                        
            # If we pruned, great. If not, we are at the same size.
            # We accept the swap because it changes the structure, potentially enabling future prunes.
            # This is a "neutral" move in terms of size, but useful for exploration.
            
    # Final Fallback
    if best_solution is None:
        # Should not happen, but just in case
        local_rng = random.Random(123)
        best_solution = greedy_cover(local_rng)
        best_solution = prune_redundant(best_solution)
        
    # Convert indices back to blocks
    return [list(all_blocks[bi]) for bi in best_solution]
