<<<<<<< SEARCH
    # 4. Lagrangian Relaxation Setup
    # Initial weights for Lagrangian multipliers
    # Heuristic: Start with weights inversely proportional to frequency
    t_counts = np.bincount(block_t_indices.ravel(), minlength=n_t)
    t_counts[t_counts == 0] = 1
    lambdas = 1.0 / t_counts.astype(np.float64)
    
    best_solution = None
    best_size = float('inf')
    best_indices = []
    
    # Parameters for Subgradient Optimization
    # We will run a series of "phases" where we adjust learning rate and penalty
    max_phases = 20
    base_lr = 0.5
    base_penalty = 0.1
    
    for phase in range(max_phases):
        if time.time() > deadline - 10.0:
            break
            
        lr = base_lr * (0.9 ** phase)
        penalty = base_penalty * (1.1 ** phase)
        
        # In each phase, we perform a "greedy" selection guided by current lambdas
        # Score for block b: sum(lambda[s] for s in b) - penalty
        # We select blocks with positive score, or top K if none positive, until coverage is met or time runs out.
        
        # To make it more robust, we can add random noise to lambdas to diversify
        rng = random.Random(42 + phase)
        noise = rng.random(n_t) * 0.01
        current_lambdas = lambdas + noise
        
        # Compute scores for all blocks
        # scores[b] = sum(current_lambdas[block_t_indices[b]])
        scores = np.sum(current_lambdas[block_t_indices], axis=1)
        
        # We want to select blocks that maximize score - penalty.
        # However, we must ensure we eventually cover everything.
        # Strategy: Sort blocks by score descending. Add them if score > penalty.
        # If we run out of high-score blocks but haven't covered everything, we must add lower score blocks.
        
        order = np.argsort(scores)[::-1]
        selected = []
        covered_mask = np.zeros(n_t, dtype=bool)
        
        # Heuristic: Start with a few high-scoring blocks, then switch to "cover remaining" mode
        # or just greedily pick best score > threshold, then fill gaps.
        
        # Let's try a hybrid: Pick top M blocks by score, then check coverage.
        # If not covered, add blocks that cover most uncovered subsets (standard greedy) but weighted by lambdas.
        
        # Actually, a simpler Lagrangian heuristic is:
        # 1. Sort by score.
        # 2. Iterate through sorted blocks. Add block if it covers at least one uncovered subset AND (score > penalty OR we are in "must cover" mode).
        # Let's stick to: Add if score > penalty. If we finish list and not covered, fallback to standard greedy on uncovered.
        
        for b_idx in order:
            if time.time() > deadline - 5.0:
                break
                
            # Check if this block covers any currently uncovered subset
            # This check is expensive if done naively for every block.
            # Optimization: Only consider block if it has high score.
            # If scores are sorted, we can break early if score < penalty and we have a reasonable number of blocks?
            # No, we need coverage.
            
            # Let's just add blocks with score > penalty.
            if scores[b_idx] < penalty:
                # If we stop here, we might not have full coverage.
                # We will handle "filling gaps" after this loop.
                pass 
            else:
                selected.append(b_idx)
                covered_mask[block_t_indices[b_idx]] = True
                
        # Check if we have full coverage
        if not covered_mask.all():
            # Fill gaps using standard greedy on the remaining uncovered subsets
            # This part is similar to the original greedy but guided by the fact that we already have some blocks.
            uncovered_indices = np.where(~covered_mask)[0]
            if len(uncovered_indices) > 0:
                # Calculate gains for remaining blocks
                # Gain = number of new subsets covered
                # We can reuse the precomputed matrix.
                # For each block, gain = sum(1 for s in block if not covered[s])
                # Since n_k is ~75k and n_t is ~11628, this is a matrix-vector product if we represent coverage as vector.
                # But we need to avoid double counting? No, we just want to know if it adds value.
                
                # Efficient way:
                # gains = (1 - covered_mask)[block_t_indices].sum(axis=1)
                gains = np.sum((1 - covered_mask)[block_t_indices], axis=1)
                
                # Only consider blocks not already selected?
                # It's okay to consider all, but selected ones will have gain 0 (or low).
                # We should avoid adding duplicates.
                selected_set = set(selected)
                
                while not covered_mask.all():
                    if time.time() > deadline - 2.0:
                        break
                    
                    # Find best block
                    # Mask out already selected? Not strictly necessary if we check gain > 0, but good for speed.
                    # Actually, if a block is already selected, its gain is 0 relative to *new* coverage.
                    # So we can just pick max gain.
                    
                    max_gain = np.max(gains)
                    if max_gain == 0:
                        # Should not happen if problem is solvable
                        break
                        
                    candidates = np.where(gains == max_gain)[0]
                    # Pick random candidate to diversify
                    best_idx = rng.choice(candidates)
                    
                    if best_idx not in selected_set:
                        selected.append(best_idx)
                        selected_set.add(best_idx)
                        covered_mask[block_t_indices[best_idx]] = True
                        # Update gains for other blocks?
                        # This is O(N) per step. If we need many steps, this is slow.
                        # However, the "Lagrangian" phase should have covered most.
                        # To speed up, we can just recompute gains for all blocks every few steps?
                        # Or just accept the O(N) cost if the number of fill steps is small.
                        # Let's update gains incrementally:
                        # For any block b, if it shares a subset s with best_idx that was just covered, its gain decreases.
                        # This is complex. Let's just recompute if time allows, or just pick next max from current static gains?
                        # Static gains will be wrong.
                        # Let's recompute gains. It's a vectorized op: ~1ms.
                        gains = np.sum((1 - covered_mask)[block_t_indices], axis=1)
                        # Zero out selected blocks to avoid re-picking (though gain would be 0 anyway)
                        gains[list(selected_set)] = 0

        if covered_mask.all():
            current_size = len(selected)
            if current_size < best_size:
                best_size = current_size
                best_indices = selected.copy()
                best_solution = [k_subsets[i] for i in best_indices]
                
        # Update Lagrangian Multipliers (Subgradient)
        # Gradient of Lagrangian dual: g_s = 1 if s is uncovered, 0 if covered?
        # Actually, the subgradient for min sum(x) s.t. A x >= 1 is:
        # g_s = 1 - sum_{b: s in b} x_b
        # If x_b is 1 (selected), and s is covered by multiple, sum > 1, g < 0.
        # If s is uncovered, sum = 0, g = 1.
        
        # Calculate coverage count for selected blocks
        if selected:
            coverage_count = np.zeros(n_t, dtype=np.int32)
            for bi in selected:
                coverage_count[block_t_indices[bi]] += 1
        else:
            coverage_count = np.zeros(n_t, dtype=np.int32)
            
        # Subgradient: 1 - coverage_count
        # If uncovered (count=0), grad = 1.
        # If covered (count>=1), grad <= 0.
        grad = 1.0 - coverage_count.astype(np.float64)
        
        # Normalize gradient?
        # Standard subgradient step: lambda += step_size * grad
        # Step size often decays.
        
        # We want lambdas to stay positive.
        lambdas = np.maximum(0, lambdas + lr * grad)
        
        # Optional: Decay lambdas slightly to prevent explosion?
        # Or just let them be.
=======
    # 4. Iterative Weighted Greedy Set Cover
    # This approach runs multiple instances of the greedy algorithm with different
    # random seeds or weight perturbations to escape local optima.
    
    best_solution = None
    best_size = float('inf')
    best_indices = []
    
    max_runs = 50
    run_count = 0
    
    while time.time() < deadline - 10.0:
        if run_count >= max_runs:
            break
        run_count += 1
        
        rng = random.Random(42 + run_count)
        
        # Initialize weights. We can start with uniform or random.
        # Random weights help diversify the greedy path.
        weights = rng.random(n_t)
        
        # Greedy Selection
        selected = []
        covered_mask = np.zeros(n_t, dtype=bool)
        
        # Precompute block weights based on initial weights
        # We will recompute scores dynamically or use a static approximation?
        # Dynamic is better for quality.
        
        # To speed up, we can pre-compute the list of t-subsets for each block.
        # block_t_indices is already (n_k, n_t_in_k)
        
        while not covered_mask.all():
            if time.time() > deadline - 5.0:
                break
            
            # Calculate score for each block: sum of weights of uncovered t-subsets in the block
            # uncovered_mask = ~covered_mask
            # gains[b] = sum(weights[s] for s in block_t_indices[b] if not covered_mask[s])
            
            # Vectorized calculation:
            # We need to zero out covered elements in weights before summing.
            # active_weights = weights * (~covered_mask)
            active_weights = weights * (~covered_mask)
            
            scores = np.sum(active_weights[block_t_indices], axis=1)
            
            # Find max score
            max_score = np.max(scores)
            if max_score == 0:
                # Should not happen unless problem is infeasible or covered_mask logic is wrong
                break
                
            # Get candidates with max score
            candidates = np.where(scores == max_score)[0]
            
            # Pick one randomly
            best_idx = rng.choice(candidates)
            
            selected.append(best_idx)
            covered_mask[block_t_indices[best_idx]] = True
            
            # Optimization: If time is tight, we might want to break early if we have a good solution?
            # No, we need full coverage.

        if covered_mask.all():
            current_size = len(selected)
            if current_size < best_size:
                best_size = current_size
                best_indices = selected.copy()
                # We don't construct best_solution immediately to save time, just store indices
                # But we need to check if we should update best_solution?
                # The local search at the end will handle the final list construction.
                # However, we need to know if we have a valid solution to potentially stop early?
                # The contract says "minimized". We should run as many as possible.
                
                # Early stopping heuristic: If we found a solution close to the known lower bound?
                # LB is 264. If we find < 300, it's very good.
                # But we don't know LB inside the function generally.
                # We'll just keep running.

>>>>>>> REPLACE

