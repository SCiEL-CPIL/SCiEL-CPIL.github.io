    # Strategy 3: Random restarts
    best_S = S1 if score1 >= score2 else S2
    best_score = max(score1, score2)
    
    # Try local search with random restarts from best_S
    while time.time() < deadline - 5.0:  # Stop 5s before deadline
        # Perturb best_S
        perturbed_S = set(best_S)
        # Remove a few random elements
        num_remove = random.randint(1, 10)
        if len(perturbed_S) > num_remove:
            to_remove = random.sample(list(perturbed_S), num_remove)
            for x in to_remove:
                perturbed_S.remove(x)
        else:
            # If set is small, just clear it and start with random small set?
            # Or just break
            break
            
        S_new, score_new = greedy_optimize(perturbed_S)
        
        if score_new > best_score:
            best_S = S_new
            best_score = score_new
