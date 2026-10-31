<<<<<<< SEARCH
    # Main Optimization Loop
    temperature = 10.0
    cooling = 0.95
    min_temp = 0.01
    
    # Initial reduction
    current_blocks = reduce_redundant(current_blocks, cover_counts)
    if len(current_blocks) < best_size:
        best_blocks = list(current_blocks)
        best_size = len(best_blocks)
        
    random.seed(42)
    
    while time.time() < deadline - 5.0:
        # 1. Reduce redundant blocks
        current_blocks = reduce_redundant(current_blocks, cover_counts)
        
        # Update best
        if len(current_blocks) < best_size:
            best_blocks = list(current_blocks)
            best_size = len(best_blocks)
            # Reset temperature? Or just continue.
            
        # 2. Perturb
        # Remove 1 to 3 random blocks
        num_removals = random.randint(1, 3)
        if len(current_blocks) <= num_removals:
            break
            
        # Pick random indices to remove
        removals = random.sample(current_blocks, num_removals)
        
        # Apply removals
        for idx in removals:
            cover_counts -= Cover[idx]
        current_blocks = [b for b in current_blocks if b not in removals]
        
        # 3. Repair
        current_blocks = repair(current_blocks, cover_counts)
        
        # 4. Acceptance
        new_size = len(current_blocks)
        delta = new_size - best_size
        
        if delta < 0:
            # Improvement
            best_blocks = list(current_blocks)
            best_size = new_size
        elif temperature > min_temp:
            # Simulated Annealing acceptance
            # Probability of accepting worse solution
            # Delta is positive (worse)
            prob = np.exp(-delta / temperature)
            if random.random() < prob:
                # Accept, but we don't update best, we just stay in this state
                pass
            else:
                # Reject: revert to best state? 
                # In ILS, we usually keep the current state if accepted, 
                # but if rejected, we might want to go back to the last good state 
                # or just continue from the perturbed state (which is worse).
                # Standard ILS: If rejected, revert to the state BEFORE perturbation.
                # We need to store the state before perturbation.
                pass # Simplification: We will just restart from best if rejected often? 
                # Actually, let's just keep the current (worse) state for now and hope it leads to a better one?
                # No, that's random walk. 
                # Let's revert to best_blocks if rejected to maintain a good baseline.
                # But we need to reconstruct cover_counts for best_blocks.
                # This is expensive.
                
                # Alternative: Only accept improvements (Greedy ILS).
                # Or, store the previous state.
                
                # Let's try a simpler approach:
                # Always revert to best if not improved, but with some randomness in perturbation.
                # This is "Hill Climbing" with restarts.
                
                # Let's implement "Revert to Best" logic properly.
                # We need to be able to restore cover_counts quickly.
                # Since we only have one "best" state, we can just recompute counts for best_blocks.
                # But that's O(N_best * 35). Fast enough.
                
                # Let's just restart from best if rejected.
                current_blocks = list(best_blocks)
                cover_counts = np.zeros(n_tsubs, dtype=np.int32)
                for idx in current_blocks:
                    cover_counts += Cover[idx]
                
        temperature *= cooling
=======
    # Helper: Try to replace a block with another that maintains coverage
    def try_replace_one(blocks, counts, max_candidates=150):
        """
        Try to replace one block in the solution with a different block.
        Returns (new_blocks, new_counts, success) where success=True means
        the replacement was accepted (same or better size after reduction).
        """
        n = len(blocks)
        if n == 0:
            return blocks, counts, False
        
        # Pick a random block to try replacing
        pos = random.randrange(n)
        old_idx = blocks[pos]
        
        # Get t-subsets covered by old block
        old_mask = Cover[old_idx]
        
        # Compute overlap of all blocks with old block
        overlaps = Cover @ old_mask.astype(np.int32)
        
        # Get top candidates by overlap
        top_k = min(max_candidates, n_blocks)
        candidate_indices = np.argsort(overlaps)[::-1][:top_k]
        
        for cand_idx in candidate_indices:
            cand_idx = int(cand_idx)
            if cand_idx == old_idx or cand_idx in blocks:
                continue
            
            # Simulate replacement
            new_counts = counts - old_mask + Cover[cand_idx]
            
            # Check validity
            if np.any(new_counts == 0):
                continue
            
            # Try reduction
            new_blocks = list(blocks)
            new_blocks[pos] = cand_idx
            reduced = reduce_redundant(new_blocks, new_counts.copy())
            
            if len(reduced) < len(blocks):
                return reduced, new_counts, True
            elif len(reduced) == len(blocks) and random.random() < 0.2:
                # Same size, accept with small probability for diversity
                return reduced, new_counts, True
        
        return blocks, counts, False

    # Main Optimization Loop
    temperature = 10.0
    cooling = 0.95
    min_temp = 0.01
    
    # Initial reduction
    current_blocks = reduce_redundant(current_blocks, cover_counts)
    if len(current_blocks) < best_size:
        best_blocks = list(current_blocks)
        best_size = len(best_blocks)
        
    random.seed(42)
    
    while time.time() < deadline - 5.0:
        # 1. Reduce redundant blocks
        current_blocks = reduce_redundant(current_blocks, cover_counts)
        
        # Update best
        if len(current_blocks) < best_size:
            best_blocks = list(current_blocks)
            best_size = len(best_blocks)
        
        # 2. Perturb - choose strategy
        r = random.random()
        
        if r < 0.35 and len(current_blocks) > 1:
            # Strategy A: Remove 1-2 random blocks
            num_removals = random.randint(1, 2)
            if len(current_blocks) <= num_removals:
                break
            removals = random.sample(current_blocks, num_removals)
            for idx in removals:
                cover_counts -= Cover[idx]
            current_blocks = [b for b in current_blocks if b not in removals]
            
        elif r < 0.70 and len(current_blocks) > 1:
            # Strategy B: Try block replacement
            current_blocks, cover_counts, success = try_replace_one(
                current_blocks, cover_counts, max_candidates=100
            )
            if not success:
                # Fallback: remove one random block
                if len(current_blocks) > 1:
                    idx = random.choice(current_blocks)
                    cover_counts -= Cover[idx]
                    current_blocks = [b for b in current_blocks if b != idx]
            
        else:
            # Strategy C: Remove 2-4 random blocks (aggressive)
            num_removals = random.randint(2, min(4, max(1, len(current_blocks) - 1)))
            if len(current_blocks) <= num_removals:
                break
            removals = random.sample(current_blocks, num_removals)
            for idx in removals:
                cover_counts -= Cover[idx]
            current_blocks = [b for b in current_blocks if b not in removals]
        
        # 3. Repair
        current_blocks = repair(current_blocks, cover_counts)
        
        # 4. Acceptance
        new_size = len(current_blocks)
        delta = new_size - best_size
        
        if delta < 0:
            # Improvement
            best_blocks = list(current_blocks)
            best_size = new_size
        elif temperature > min_temp:
            # Simulated Annealing acceptance
            prob = np.exp(-delta / temperature)
            if random.random() < prob:
                pass
            else:
                # Reject: revert to best state
                current_blocks = list(best_blocks)
                cover_counts = np.zeros(n_tsubs, dtype=np.int32)
                for idx in current_blocks:
                    cover_counts += Cover[idx]
                
        temperature *= cooling
>>>>>>> REPLACE

