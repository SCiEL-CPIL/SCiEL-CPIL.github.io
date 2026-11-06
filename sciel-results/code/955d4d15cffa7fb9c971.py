        reduced, _ = _greedy_remove_pass(perturbed, deadline)
        
        # Only accept if valid and better
        if len(reduced) < best_len and _verify(reduced):
            best = reduced
            best_len = len(best)
