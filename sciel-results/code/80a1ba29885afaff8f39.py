<<<<<<< SEARCH
def _greedy_build(pair_mask, rng, deadline):
    """
    Greedy build: pick points that maximize blocked points.
    Returns a valid (but not necessarily maximal) set.
    """
    S = []
    U_mask = np.ones(V, dtype=bool)
    
    while U_mask.any():
        if time.time() > deadline:
            break
        
        # Find all candidates that can be added
        candidates = []
        for p in np.where(U_mask)[0]:
            if _can_add(p, S, pair_mask):
                candidates.append(p)
        
        if not candidates:
            break
        
        # Score each candidate by how many points in U it would block
        best_score = -1
        best_p = None
        
        # Sample up to 60 candidates for speed if many
        if len(candidates) > 60:
            sampled = rng.sample(candidates, 60)
        else:
            sampled = candidates
        
        for p in sampled:
            # Compute how many points in U would be blocked by adding p
            blocked_count = 0
            for a in S:
                # Count U points that conflict with pair (a, p)
                conflicts = pair_mask[a, p]
                blocked_count += np.sum(conflicts & U_mask)
            
            if blocked_count > best_score:
                best_score = blocked_count
                best_p = p
            elif blocked_count == best_score and rng.random() < 0.3:
                best_p = p
        
        if best_p is None:
            break
        
        S.append(best_p)
        U_mask[best_p] = False
        # Update U_mask: remove points blocked by new pairs with best_p
        for a in S[:-1]:
            U_mask &= ~pair_mask[a, best_p]
    
    return S
=======
def _greedy_build(pair_mask, rng, deadline):
    """
    Greedy build: pick points that maximize blocked points.
    Returns a valid (but not necessarily maximal) set.
    """
    S = []
    U_mask = np.ones(V, dtype=bool)
    
    while U_mask.any():
        if time.time() > deadline:
            break
        
        # Find all candidates that can be added using vectorized check
        U_indices = np.where(U_mask)[0]
        can_add_mask = _can_add_all(U_indices, S, pair_mask)
        candidates = U_indices[can_add_mask]
        
        if len(candidates) == 0:
            break
        
        # Score each candidate by how many points in U it would block
        best_score = -1
        best_p = None
        
        # Sample up to 80 candidates for speed if many
        if len(candidates) > 80:
            sampled = rng.sample(candidates.tolist(), 80)
        else:
            sampled = candidates.tolist()
        
        for p in sampled:
            # Compute how many points in U would be blocked by adding p
            blocked_count = 0
            for a in S:
                conflicts = pair_mask[a, p]
                blocked_count += np.sum(conflicts & U_mask)
            
            if blocked_count > best_score:
                best_score = blocked_count
                best_p = p
            elif blocked_count == best_score and rng.random() < 0.3:
                best_p = p
        
        if best_p is None:
            break
        
        S.append(best_p)
        U_mask[best_p] = False
        # Update U_mask: remove points blocked by new pairs with best_p
        for a in S[:-1]:
            U_mask &= ~pair_mask[a, best_p]
    
    return S
>>>>>>> REPLACE

