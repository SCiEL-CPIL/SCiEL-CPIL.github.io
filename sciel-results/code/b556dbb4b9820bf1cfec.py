def _greedy_random(conflict_masks, rng):
    S = []
    forbidden = 0
    # We need to track which points are in S to avoid re-adding
    # and to compute new forbidden set efficiently?
    # Actually, we just need to iterate candidates.
    
    while forbidden != FULL_MASK:
        # Find candidates
        # Candidates are points p such that (forbidden >> p) & 1 == 0 and p not in S
        # It's faster to maintain a list of available points or just iterate 0..255
        
        candidates = []
        for p in range(V):
            if (forbidden >> p) & 1:
                continue
            if p in S: # This check is O(N) if S is list. Use a set or bitmask for S too?
                continue
            candidates.append(p)
        
        if not candidates:
            break # Should not happen if we stop when full
            
        # Score each candidate
        best_score = -1
        best_candidates = []
        
        for p in candidates:
            # Calculate new forbidden
            new_forbidden = forbidden | (1 << p)
            for q in S:
                new_forbidden |= conflict_masks[p][q]
            
            score = bin(new_forbidden).count('1')
            
            if score > best_score:
                best_score = score
                best_candidates = [p]
            elif score == best_score:
                best_candidates.append(p)
        
        # Randomly pick from best candidates
        p = rng.choice(best_candidates)
        S.append(p)
        
        # Update forbidden
        # We already calculated new_forbidden for p in the loop above?
        # To avoid recalculation, we can store the best new_forbidden.
        # Or just recalculate. It's fast.
        forbidden = 0
        forbidden |= (1 << p)
        for q in S:
            if q != p:
                forbidden |= conflict_masks[p][q]
        # Wait, the previous forbidden set is still valid?
        # The new forbidden set is the union of old forbidden and conflicts involving p.
        # Yes.
        
    return S
