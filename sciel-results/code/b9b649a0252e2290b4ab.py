    stuck_count = 0
    while time.time() < deadline - 2:
        improved = False
        # ... Hill Climbing Pass ...
        if improved:
            stuck_count = 0
            # Update best
        else:
            stuck_count += 1
            if stuck_count > 5:
                # Perform a random 2-swap perturbation
                idx1, idx2 = rng.sample(range(m), 2)
                # ... pick random c1, c2 ...
                # Apply swap
                # Recalculate score
                # If better than current, accept and reset stuck_count? 
                # Or just accept to escape.
                # Let's accept if it's not significantly worse, or just accept always to explore.
                # Actually, if we accept a worse move, we might fall into a deep hole.
                # Let's only accept if it's better than the *current* state, 
                # but since we are stuck, no single swap is better.
                # So we must accept a worse move to escape.
                # This is effectively Simulated Annealing with T=0 (greedy) failing, 
                # so we use a "kick".
                
                # Let's just do a random restart of the family if stuck for too long?
                # No, restart is expensive.
                
                # Let's try a specific heuristic: 
                # Replace the two least "central" sets with two random ones.
                # Or just random 2-swap.
                
                # Apply 2-swap.
                # If new_score > old_score, accept.
                # Else, keep it for a few iterations (annealing).
                
                # For simplicity and robustness, let's just do a random restart 
                # of the *search* (not the family) or a larger perturbation.
                
                # Actually, the vectorized search is so fast that we can just 
                # do multiple random restarts of the *initialization* if we don't improve.
                
                # But we already have a good family.
                # Let's stick to: 
                # 1. Hill Climb.
                # 2. If stuck, do 2-swap. If better, accept. If worse, revert and try another 2-swap.
                # 3. If 10 failed 2-swaps, try 3-swap.
                # 4. If 20 failed, restart with a new random initial family.
                
                pass
