def _construct_greedy_heuristic_bitset(rng):
    chosen = []
    chosen_set = set()
    blocked = 0
    
    candidates = list(range(TOTAL))
    rng.shuffle(candidates)
    
    # We will iterate through candidates, but we can re-evaluate "best" candidate each step?
    # That would be O(N^3) or similar.
    # Let's stick to a randomized greedy but with a "look-ahead" of 1 step for the first few points?
    # Or just use the random order but if we get stuck, restart?
    
    # Actually, the "block max" heuristic is:
    # Among all valid p, pick the one that maximizes popcount( new_blocked ).
    
    for _ in range(TOTAL): # Max iterations
        if time.time() > DEADLINE:
            break
            
        valid_p = []
        for p in candidates:
            if p in chosen_set or (blocked >> p) & 1:
                continue
            valid_p.append(p)
            
        if not valid_p:
            break
            
        # Pick best from valid_p
        best_p = None
        max_new_blocks = -1
        
        # If valid_p is large, sample?
        # For N=16, TOTAL=256.
        # Let's just check all valid_p.
        for p in valid_p:
            new_blocked = blocked
            for q in chosen:
                new_blocked |= ISO_MASK[p][q]
            # Count newly blocked
            newly = new_blocked & ~blocked
            # Popcount
            cnt = bin(newly).count('1')
            if cnt > max_new_blocks:
                max_new_blocks = cnt
                best_p = p
                
        if best_p is None:
            break
            
        chosen.append(best_p)
        chosen_set.add(best_p)
        
        # Update blocked
        for q in chosen:
            if q == best_p:
                continue
            blocked |= ISO_MASK[best_p][q]
            
        # Remove best_p from candidates? 
        # Or just keep list and check membership.
        
    return chosen
