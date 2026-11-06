<<<<<<< SEARCH
    # Local search: try removing comparators
    # Strategy: Repeatedly scan from end to start, try removing each comparator.
    # If removal keeps it valid, keep the removal and restart scan from end.
    # This is a greedy "remove if possible" heuristic.
    
    # Optimization: 
    # 1. Vectorized verification is the bottleneck.
    # 2. We can potentially cache intermediate states? No, too much memory.
    # 3. We can try to be smarter about which comparator to remove.
    #    Usually, redundant comparators are later in the network or in specific patterns.
    #    However, simple backward scan is robust.
    
    # To speed up, we can try to limit the search space or use a better heuristic.
    # Let's stick to the backward scan but ensure we don't exceed time.
    
    improved = True
    while improved and (time.time() < deadline - 5):
        improved = False
        # Iterate backwards to avoid index issues when removing
        for idx in range(len(current_network) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            
            # Try removing comparator at idx
            trial = current_network[:idx] + current_network[idx+1:]
            
            # Optimization: If we have removed many comparators, the network is smaller.
            # Verification cost is proportional to number of comparators.
            
            if _fast_verify(trial):
                current_network = trial
                current_len = len(current_network)
                improved = True
                # Restart scan from end to find more redundancies introduced by removal?
                # Or just continue? Restarting is safer to ensure we don't miss
                # a comparator that became redundant due to a previous removal.
                break
        
        # If we didn't break (no improvement in this pass), we are done.
        # But wait, the 'improved' flag is set to True inside the loop.
        # If we break due to time, we exit the while loop.
        
        # If we completed the inner for loop without finding a removal, improved is False, and we exit while.
=======
    # Local search: try removing and swapping comparators
    # Strategy: Alternate between removal and swap passes
    # Removal: scan from end, remove if valid
    # Swap: try swapping adjacent comparators to enable further removals
    
    improved = True
    use_swap = False
    while improved and (time.time() < deadline - 5):
        improved = False
        
        if use_swap:
            # Swap pass: try swapping adjacent pairs
            for idx in range(len(current_network) - 1):
                if time.time() > deadline - 5:
                    break
                
                # Try swapping comparators at idx and idx+1
                trial = current_network[:]
                trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
                
                if _fast_verify(trial):
                    current_network = trial
                    improved = True
                    break
        else:
            # Removal pass: scan from end, remove if valid
            for idx in range(len(current_network) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                
                # Try removing comparator at idx
                trial = current_network[:idx] + current_network[idx+1:]
                
                if _fast_verify(trial):
                    current_network = trial
                    current_len = len(current_network)
                    improved = True
                    break
        
        # Alternate between removal and swap passes
        use_swap = not use_swap
        
        # If we found an improvement, continue; otherwise, we're done
        if not improved:
            break
>>>>>>> REPLACE

