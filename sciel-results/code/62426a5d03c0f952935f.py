<<<<<<< SEARCH
def local_search(chosen_list, chosen_mask, max_iterations=100):
    """Local search: try removing a point and refilling to maximality."""
    best = list(chosen_list)
    best_mask = chosen_mask
    best_len = len(best)
    
    for iteration in range(max_iterations):
        if time.time() > DEADLINE:
            break
        
        improved = False
        # Try removing each point in the current best set
        for i in range(len(best)):
            if time.time() > DEADLINE:
                break
            
            removed = best[i]
            new_mask = best_mask & ~(1 << removed)
            new_list = [p for j, p in enumerate(best) if j != i]
            
            # Complete to maximality with a fresh random order
            rng = random.Random(iteration * 10000 + i * 7 + 42)
            new_list, new_mask = complete_to_maximal(new_list, new_mask, rng)
            
            if len(new_list) < best_len:
                best = new_list
                best_mask = new_mask
                best_len = len(best)
                improved = True
                break  # Restart with new best
        
        if not improved:
            break
    
    return best, best_mask
=======
def local_search(chosen_list, chosen_mask, max_iterations=100):
    """Local search: try removing a point and refilling to maximality."""
    best = list(chosen_list)
    best_mask = chosen_mask
    best_len = len(best)
    
    for iteration in range(max_iterations):
        if time.time() > DEADLINE:
            break
        
        improved = False
        
        # Compute conflict scores for ordering removals
        # Points with lower conflict scores are less "essential" and more likely to be removable
        conflict_scores = []
        for p in best:
            score = 0
            cm = best_mask
            while cm:
                lb = cm & (-cm)
                a = lb.bit_length() - 1
                if a != p and (conflict[p][a] & best_mask):
                    score += 1
                cm ^= lb
            conflict_scores.append((score, p))
        
        # Try removing points with lowest conflict score first
        conflict_scores.sort(key=lambda x: x[0])
        
        for _, p_to_remove in conflict_scores:
            if time.time() > DEADLINE:
                break
            
            new_mask = best_mask & ~(1 << p_to_remove)
            new_list = [p for p in best if p != p_to_remove]
            
            # Try multiple completion strategies
            best_completion = None
            best_completion_mask = 0
            best_completion_len = float('inf')
            
            for attempt in range(8):
                rng = random.Random(iteration * 10000 + hash(p_to_remove) % 10000 + attempt * 7 + 42)
                strategy = "random" if attempt < 4 else "conflict_order"
                cl, cm = complete_to_maximal(list(new_list), new_mask, rng, strategy)
                if len(cl) < best_completion_len:
                    best_completion = cl
                    best_completion_mask = cm
                    best_completion_len = len(cl)
            
            if best_completion is not None and best_completion_len < best_len:
                best = best_completion
                best_mask = best_completion_mask
                best_len = len(best)
                improved = True
                break  # Restart with new best
        
        if not improved:
            break
    
    return best, best_mask
>>>>>>> REPLACE

